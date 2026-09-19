"""Curved-manifold counterexample: sparse memorization versus continuous ring."""
from pathlib import Path
import json
import numpy as np
from scipy.special import i0e,logsumexp
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.weak_reference_20260914.toy import OUT,write_csv


def log_sparse(x,y):
    angle=np.arange(12)*2*np.pi/12
    dx=x[...,None]-2*np.cos(angle);dy=y[...,None]-2*np.sin(angle)
    var=.12**2
    return logsumexp(-(dx*dx+dy*dy)/(2*var)-np.log(2*np.pi*var*12),axis=-1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(-3,3,241);dx=axis[1]-axis[0];x,y=np.meshgrid(axis,axis)
    r=np.hypot(x,y);var=.12**2;arg=2*r/var
    logtrue=-(r*r+4)/(2*var)+np.log(i0e(arg))+arg-np.log(2*np.pi*var)
    norm=lambda logp:np.exp(logp-logp.max())/(np.exp(logp-logp.max()).sum()*dx*dx)
    truth=norm(logtrue);base=log_sparse(x,y);tau=.09
    a=.15;c,s=np.cos(a),np.sin(a)
    angular=.5*(log_sparse(c*x-s*y,s*x+c*y)+log_sparse(c*x+s*y,-s*x+c*y))
    logweak=gaussian_filter(base,np.sqrt(tau)/dx,mode='nearest',truncate=8)
    densityweak=np.log(gaussian_filter(norm(base),np.sqrt(tau)/dx,mode='nearest',truncate=8).clip(1e-200))
    pots={'truth':truth,'baseline':norm(base),'density':norm(2*base-densityweak),
          'log':norm(2*base-logweak),'angular':norm(2*base-angular)}
    nearest=np.min((x[...,None]-2*np.cos(np.arange(12)*2*np.pi/12))**2+
                   (y[...,None]-2*np.sin(np.arange(12)*2*np.pi/12))**2,axis=-1)
    rows=[]
    for kind,p in pots.items():
        rows.append(dict(method=kind,kl_true_to_potential=float(np.sum(truth*(np.log(truth.clip(1e-200))-np.log(p.clip(1e-200))))*dx*dx),
            mass_near_12_training_centers=float(np.sum(p*(nearest<.15**2))*dx*dx),
            off_ring_mass=float(np.sum(p*(np.abs(r-2)>.3))*dx*dx),
            radial_mse=float(np.sum(p*(r-2)**2)*dx*dx)))
    write_csv(OUT/'curved.csv',rows);np.savez(OUT/'curved_fields.npz',axis=axis,**pots)
    fig,axes=plt.subplots(1,5,figsize=(13,3),layout='constrained')
    for ax,(name,p) in zip(axes,pots.items()):
        ax.imshow(p,origin='lower',extent=(-3,3,-3,3),vmin=0,vmax=2,cmap='Greys')
        ax.set_title(name);ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_aspect('equal')
    fig.savefig(OUT/'curved.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
