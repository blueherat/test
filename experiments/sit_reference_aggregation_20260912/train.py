import argparse
import torch
from . import catalog as c,data,models
from experiments.sit_reference_compilation_20260912 import train as previous
from experiments.lifting_scale_sweep_20260909 import atomic,read,sha

state_sha=previous.state_sha


def prepare():
    data.verify()
    if (c.ROOT/'training_request.json').exists():return verify()
    receipt=read(c.ROOT/'data/complete.json')
    assert receipt['passed'] and receipt['paired_index_time_exact']
    assert receipt['data_request_sha256']==sha(c.ROOT/'data_request.json')
    request=dict(data.verify());request['input_files']=dict(request['input_files'])
    request['input_files'].update(receipt['files'])
    request.update(steps=c.STEPS,batch=c.BATCH,learning_rate=c.LR,ema=c.EMA,seed=c.TRAIN_SEED,
        initialization='previous shallow EMA, optimizer reset identically',old_new_state_ratio='1:1',selected_by_validation=False)
    atomic(c.ROOT/'training_request.json',request);return request


def verify():
    request=read(c.ROOT/'training_request.json')
    for key in ('sources','assets','input_files','old_stop_markers'):
        for path,digest in request[key].items():assert sha(path)==digest,(key,path)
    return request


class Rollouts:
    def __init__(self,method):
        keys=('index','f4','context','teacher','strong')
        def load(root,prefix):
            rows={k:[] for k in keys}
            for rank in range(4):
                path=root/'data'/f'{prefix}_rank{rank}.pt';value=torch.load(path,map_location='cpu',weights_only=False)
                for k in keys:rows[k].append(value[k])
                del value
            return {k:torch.cat(v) for k,v in rows.items()}
        old=load(c.PARENT_ROOT,'rollout');fresh=load(c.ROOT,'teacher' if method=='ig_teachercontinuation' else 'student')
        self.train={k:torch.cat([old[k][old['index']<800],fresh[k][fresh['index']<800]]).cuda() for k in keys}
        # Both validation sets are the same fresh student-visited heldout states.
        valid=fresh if method=='ig_studentaggregation' else load(c.ROOT,'student')
        self.valid={k:v[valid['index']>=800].cuda() for k,v in valid.items()};self.feature='f4'


def train(method):
    previous.c=c;previous.data=data;previous.models=models
    previous.Rollouts=lambda depth:Rollouts(method)
    previous.verify=verify
    previous.train(method)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',choices=c.METHODS,required=True);train(p.parse_args().train)
