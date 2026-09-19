import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from . import config as k
from .data import RealDataset,GeneratedBank
from .models import Adapter
from experiments import train_imagenet100_sit_flow as base


@torch.inference_mode()
def features(model,rank):
    root=k.model_root(model)/'weights';root.mkdir(parents=True,exist_ok=True)
    adapter=Adapter(model);generator=torch.Generator(device='cuda').manual_seed(adapter.cfg['seed']+8100+rank)
    files={}
    for split in ('train','validation'):
        dataset=RealDataset(model,split);bank=GeneratedBank(model,'strong',split)
        ids=list(range(rank,len(dataset),4))
        loader=DataLoader(Subset(dataset,ids),batch_size=64,num_workers=4,pin_memory=True)
        for source in (0,1):
            path=root/f'features_{split}_{source}_{rank}.npz'
            if path.exists() and path.with_suffix('.json').exists():
                meta=k.read(path.with_suffix('.json'));assert meta['sha256']==k.sha(path)
                files[str(path)]=meta['sha256'];continue
            parts=[];positions=[];labels=[]
            for values,y,index in loader:
                k.check_stop()
                if source:clean=torch.from_numpy(bank.get(index.numpy())).cuda()
                elif model=='sit_small':
                    clean=base.sample_sdvae_posterior(values.cuda(),torch.randn((len(y),4,32,32),device='cuda',generator=generator))
                else:clean=values.cuda().mul(2).sub(1)
                with adapter.autocast(training=True):
                    feat=adapter.features(clean,torch.full((len(y),),.99,device='cuda'),y.cuda())['context'].float()
                parts.append(torch.cat([feat.mean(1),feat.std(1,correction=0)],1).cpu().numpy())
                positions.append(index.numpy());labels.append(y.numpy())
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez(f,features=np.concatenate(parts),ids=np.concatenate(positions),labels=np.concatenate(labels))
            tmp.replace(path);files[str(path)]=k.sha(path)
            k.atomic(path.with_suffix('.json'),dict(sha256=k.sha(path),request_sha256=k.sha(k.ROOT/'request.json')))
    k.atomic(root/f'features_rank{rank}.json',dict(complete=True,files=files,request_sha256=k.sha(k.ROOT/'request.json')))


def fit(model):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from experiments.weak_reference_loss_20260914.objectives import excess_weights
    root=k.model_root(model)/'weights';groups={}
    for split in ('train','validation'):
        rows=[]
        for source in (0,1):
            for rank in range(4):
                p=root/f'features_{split}_{source}_{rank}.npz'
                with np.load(p) as value:
                    rows.append((value['features'],value['ids'],value['labels'],np.full(len(value['ids']),source)))
        groups[split]=tuple(np.concatenate([row[i] for row in rows]) for i in range(4))
    probabilities={s:np.empty(len(v[0]),np.float64) for s,v in groups.items()};diagnostics=[]
    for fold in (0,1):
        features,ids,labels,source=groups['train'];mask=ids%2!=fold
        scaler=StandardScaler().fit(features[mask]);classifier=LogisticRegression(C=1.,max_iter=2000,random_state=40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            classifier.fit(scaler.transform(features[mask]),source[mask])
        for split,(features,ids,labels,source) in groups.items():
            heldout=ids%2==fold
            probabilities[split][heldout]=classifier.predict_proba(scaler.transform(features[heldout]))[:,1]
        np.savez(root/f'classifier_fold{fold}.npz',coef=classifier.coef_,intercept=classifier.intercept_,mean=scaler.mean_,scale=scaler.scale_)
        diagnostics.append(dict(fold=fold,training_examples=int(mask.sum()),converged=not caught))
    normalizers=None;rng=np.random.default_rng(41)
    for split in ('train','validation'):
        _,ids,labels,source=groups[split];mask=source==1
        order=np.argsort(ids[mask]);y=labels[mask][order]
        raw=excess_weights(torch.from_numpy(probabilities[split][mask][order])).numpy()
        if split=='train':normalizers=np.array([raw[y==c].mean() for c in range(k.settings(model)['classes'])])
        weights=(raw/normalizers[y]).astype(np.float32);shuffled=weights.copy()
        for label in range(k.settings(model)['classes']):
            selected=np.flatnonzero(y==label);shuffled[selected]=rng.permutation(shuffled[selected])
        np.save(root/f'excess_{split}.npy',weights);np.save(root/f'shuffled_{split}.npy',shuffled)
    files={str(p):k.sha(p) for p in root.glob('*.npy')}
    files.update({str(p):k.sha(p) for p in root.glob('classifier_fold*.npz')})
    k.atomic(root/'complete.json',dict(complete=True,request_sha256=k.sha(k.ROOT/'request.json'),
        files=files,folds=diagnostics,full_training_data=True,validation_excluded_from_fitting=True))
