"""Offline endpoint weights; clean validation endpoints never fit a classifier."""
import time
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config as k
from . import components as x
from experiments.weak_reference_loss_20260914.objectives import excess_weights


def fit():
    k.verify()
    out=k.ROOT/'weights'
    path=out/'endpoint_features.npz'
    assert k.read(out/'features_complete.json')['files'][str(path)]==k.sha(path)
    with np.load(path) as d:
        features,source,labels,ids=[d[key] for key in ('features','source','labels','endpoint_index')]
    probability=np.full(len(features),np.nan)
    records=[]
    start=time.perf_counter()
    for fold in (0,1):
        train=(ids%2 != fold)&(ids<18)
        valid=ids%2==fold
        assert all(np.sum(train&(source==s)&(labels==c))==9 for s in (0,1) for c in range(100))
        scaler=StandardScaler().fit(features[train])
        model=LogisticRegression(C=1.,max_iter=2000,random_state=k.SEED+40)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(scaler.transform(features[train]),source[train])
        probability[valid]=model.predict_proba(scaler.transform(features[valid]))[:,1]
        np.savez(out/f'classifier_fold{fold}.npz',coef=model.coef_,intercept=model.intercept_,
            mean=scaler.mean_,scale=scaler.scale_,training_ids=np.flatnonzero(train),validation_ids=np.flatnonzero(valid))
        records.append(dict(fold=fold,training=int(train.sum()),predicted=int(valid.sum()),
            auc=float(roc_auc_score(source[valid],probability[valid])),log_loss=float(log_loss(source[valid],probability[valid]))))
    assert np.isfinite(probability).all()
    raw=excess_weights(torch.from_numpy(probability[source==1])).numpy().reshape(100,20)
    weights=(raw/raw[:,:18].mean(1,keepdims=True)).astype(np.float32)
    shuffled=weights.copy()
    rng=np.random.default_rng(k.SEED+41)
    # Preserve the train/validation endpoint split as well as each class.
    for row in shuffled:
        rng.shuffle(row[:18]);rng.shuffle(row[18:])
    for name,array in [('probability',probability),('raw_excess',raw),('excess',weights),('shuffled',shuffled)]:
        np.save(out/(name+'.npy'),array)
    files=[path,*out.glob('*.npy'),*out.glob('classifier_fold*.npz')]
    k.atomic(out/'complete.json',dict(complete=True,degenerate=bool(weights[:,:18].std()<1e-3),
        normalized_weight_std=float(weights[:,:18].std()),raw_weight_min=float(raw.min()),raw_weight_max=float(raw.max()),
        source_label_one='generated',feature_ratio_only=True,cross_fitted=True,clean_validation_held_out=True,
        degeneracy_policy='still train all authorized 50K controls; no promotion if weights degenerate',
        folds=records,seconds=time.perf_counter()-start,files={str(p):k.sha(p) for p in files},
        request_sha256=k.sha(k.ROOT/'request.json')))
