import io
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader,DistributedSampler,Dataset
from . import config as k
from experiments import train_imagenet100_sit_flow as base


def prepare_indices():
    from experiments.raev2_training_core import DeterministicImageNetPacked
    result={}
    for model in k.MODELS:
        root=k.model_root(model)/'data';root.mkdir(parents=True,exist_ok=True)
        if model=='sit_small':
            groups={s:(np.arange(len(np.load(k.SIT_DATA/f'{s}_labels.npy'))),
                np.load(k.SIT_DATA/f'{s}_labels.npy')) for s in ('train','validation')}
        else:
            dataset=DeterministicImageNetPacked(k.JIT_DATA,image_size=256,horizontal_flip=False)
            labels=np.concatenate(dataset._labels)
            validation=np.load(k.EXPS/'jit_internal_readouts_20260908/validation_ids.npy')
            train=np.setdiff1d(np.arange(len(labels)),validation)
            groups={s:(ids,labels[ids]) for s,ids in (('train',train),('validation',validation))}
            assert not np.intersect1d(train,validation).size
        for split,(ids,labels) in groups.items():
            np.save(root/f'{split}_ids.npy',ids);np.save(root/f'{split}_labels.npy',labels)
        result[model]=dict(train_images=len(groups['train'][0]),validation_images=len(groups['validation'][0]),
            files={str(p):k.sha(p) for p in root.glob('*.npy')})
    return result


class RealDataset(Dataset):
    def __init__(self,model,split,exclude_fold=None):
        self.model,self.split=model,split
        self.ids=np.load(k.model_root(model)/'data'/f'{split}_ids.npy')
        self.positions=np.arange(len(self.ids))
        if exclude_fold is not None:self.positions=self.positions[self.positions%2!=exclude_fold]
        if model=='sit_small':self.dataset=base.NpyMomentsDataset(k.SIT_DATA,split)
        else:
            from experiments.raev2_training_core import DeterministicImageNetPacked
            self.dataset=DeterministicImageNetPacked(k.JIT_DATA,image_size=256,horizontal_flip=False)
    def __len__(self):return len(self.positions)
    def __getitem__(self,index):
        position=int(self.positions[index]);value=self.dataset[int(self.ids[position])]
        return value[0],int(value[1]),position


def loader(model,split,context,exclude_fold=None):
    dataset=RealDataset(model,split,exclude_fold)
    train=split=='train';seed=k.settings(model)['seed']+(0 if train else 1)
    sampler=DistributedSampler(dataset,num_replicas=context.world_size,rank=context.rank,
        shuffle=train,seed=seed,drop_last=train)
    result=DataLoader(dataset,batch_size=k.GLOBAL_BATCH//context.world_size,sampler=sampler,
        num_workers=4 if train else 2,pin_memory=True,drop_last=train,persistent_workers=True,
        prefetch_factor=4,generator=torch.Generator().manual_seed(seed+91337+context.rank))
    return result,sampler


class GeneratedBank:
    def __init__(self,model,source,split):
        self.model,self.source,self.split=model,source,split
        self.root=k.model_root(model)/'endpoints'/source/split
        self.receipt=k.read(self.root/'complete.json')
        assert self.receipt['complete'] and self.receipt['request_sha256']==k.sha(k.ROOT/'request.json')
        self.labels=np.load(k.model_root(model)/'data'/f'{split}_labels.npy')
        assert self.receipt['samples']==len(self.labels)
        self.by_class=[np.flatnonzero(self.labels==c) for c in range(k.settings(model)['classes'])]
        self.cache=OrderedDict()
        self.values=np.load(self.root/'clean.npy',mmap_mode='r') if model=='sit_small' else None
    def get(self,ids):
        if self.values is not None:return np.array(self.values[ids])
        from PIL import Image
        result=[];size=k.settings(self.model)['endpoint_batch']
        for idx in ids:
            start=int(idx)//size*size;key=f'{start:08d}'
            if key not in self.cache:
                with np.load(self.root/'batches'/(key+'.npz')) as index:
                    self.cache[key]=(index['offsets'].copy(),index['sizes'].copy())
                if len(self.cache)>128:self.cache.popitem(last=False)
            offsets,sizes=self.cache[key];slot=int(idx)-start
            with (self.root/'batches'/(key+'.bin')).open('rb') as f:
                f.seek(int(offsets[slot]));payload=f.read(int(sizes[slot]))
            with Image.open(io.BytesIO(payload)) as image:
                result.append(np.asarray(image.convert('RGB'),np.float32).transpose(2,0,1)/127.5-1)
        return np.stack(result)
    def draw(self,labels,generator,exclude_fold=None):
        ids=[]
        coins=torch.rand(len(labels),device='cuda',generator=generator).cpu().numpy()
        for label,coin in zip(labels,coins):
            pool=self.by_class[int(label)]
            if exclude_fold is not None:pool=pool[pool%2!=exclude_fold]
            ids.append(int(pool[min(int(coin*len(pool)),len(pool)-1)]))
        return torch.from_numpy(self.get(ids)).cuda(),torch.tensor(ids,device='cuda')


class Stream:
    def __init__(self,model,source,context,start_step=0,exclude_fold=None,validation=False):
        self.model,self.source,self.exclude_fold=model,source,exclude_fold
        self.validation=validation;split='validation' if validation else 'train'
        self.loader,self.sampler=loader(model,split,context,exclude_fold)
        self.iterator=iter(self.loader) if validation else base.infinite_train_batches(self.loader,self.sampler,start_step)
        sources=('strong','weak') if source=='weakmix' else () if source=='real' else (source,)
        self.banks={s:GeneratedBank(model,s,split) for s in sources}
    def draw(self,generator):
        values,labels,ids=next(self.iterator)
        values=values.cuda(non_blocking=True);labels=labels.cuda(non_blocking=True);ids=ids.cuda()
        n=len(labels)
        if self.model=='sit_small':
            posterior=torch.randn((n,4,32,32),device='cuda',generator=generator)
            positive=base.sample_sdvae_posterior(values,posterior)
        else:positive=values.mul(2).sub(1)
        if self.banks:
            negative,negative_ids=self.banks[next(iter(self.banks))].draw(labels.cpu().numpy(),generator,self.exclude_fold)
            if self.source=='weakmix':
                weak,weak_ids=self.banks['weak'].draw(labels.cpu().numpy(),generator,self.exclude_fold)
                choose=torch.rand(n,device='cuda',generator=generator)<.5
                negative=torch.where(choose[:,None,None,None],weak,negative)
                # ID parity is retained; component labels are not provided to the head.
                negative_ids=torch.where(choose,weak_ids,negative_ids)
        else:negative,negative_ids=positive,ids
        if self.model=='jit' and not self.validation:
            flip=torch.rand(n,device='cuda',generator=generator)<.5
            positive=torch.where(flip[:,None,None,None],positive.flip(-1),positive)
            negative=torch.where(flip[:,None,None,None],negative.flip(-1),negative)
        noise=torch.randn(positive.shape,device='cuda',generator=generator)
        if self.model=='sit_small':time=torch.rand(n,device='cuda',generator=generator)
        else:time=torch.sigmoid(torch.randn(n,device='cuda',generator=generator)*.8-.8)
        conditioned=labels
        if self.model=='jit' and not self.validation:
            conditioned=torch.where(torch.rand(n,device='cuda',generator=generator)<.1,1000,labels)
        source=(torch.randperm(n,device='cuda',generator=generator)%2).float()
        clean=torch.where(source[:,None,None,None].bool(),positive,negative)
        return dict(positive=positive,negative=negative,clean=clean,labels=conditioned,true_labels=labels,
            pos_id=ids,neg_id=negative_ids,fold=torch.where(source.bool(),ids,negative_ids)%2,
            t=time,noise=noise,source=source,coin=torch.rand(n,device='cuda',generator=generator),
            null_source=(torch.randperm(n,device='cuda',generator=generator)%2).float())
