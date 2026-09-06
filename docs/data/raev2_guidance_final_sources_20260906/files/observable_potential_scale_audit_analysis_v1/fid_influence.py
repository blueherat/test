"""CPU-only local FID influence approximation for fixed-reference paired arms."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
from scipy.linalg import eigh


def spd_eigh(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError(f'{name}: expected finite square matrix')
    eps = np.finfo(np.float64).eps
    symmetry_error = np.linalg.norm(matrix - matrix.T, 'fro')
    scale = np.linalg.norm(matrix, 'fro')
    if symmetry_error > 100 * eps * len(matrix) * scale:
        raise ValueError(f'{name}: materially asymmetric')
    # Symmetrization removes arithmetic antisymmetry only; never add a ridge.
    matrix = (matrix + matrix.T) * .5
    values, vectors = eigh(matrix, check_finite=True, driver='evd')
    tolerance = eps * len(matrix) * float(np.max(np.abs(values)))
    if values[0] <= tolerance:
        raise ValueError(f'{name}: not numerically SPD; min={values[0]}, rank_tolerance={tolerance}; no ridge/truncation allowed')
    return values, vectors, {'minimum_eigenvalue': float(values[0]),
        'maximum_eigenvalue': float(values[-1]), 'condition_number': float(values[-1]/values[0]),
        'numerical_rank_tolerance': tolerance}


def fid_and_gradients(mean, covariance, reference_mean, reference_covariance):
    mean, reference_mean = np.asarray(mean, np.float64), np.asarray(reference_mean, np.float64)
    covariance, reference_covariance = np.asarray(covariance, np.float64), np.asarray(reference_covariance, np.float64)
    if mean.ndim != 1 or mean.shape != reference_mean.shape or covariance.shape != (len(mean),len(mean)) or reference_covariance.shape != covariance.shape:
        raise ValueError('mean/covariance dimensions differ')
    if not np.isfinite(mean).all() or not np.isfinite(reference_mean).all():
        raise ValueError('nonfinite mean')
    lam, vec, sample_audit = spd_eigh(covariance, 'sample covariance')
    _, _, reference_audit = spd_eigh(reference_covariance, 'reference covariance')
    square = (vec * np.sqrt(lam)) @ vec.T
    inverse_square = (vec * (1/np.sqrt(lam))) @ vec.T
    middle = square @ reference_covariance @ square
    gamma, basis, middle_audit = spd_eigh(middle, 'S^(1/2) R S^(1/2)')
    middle_root = (basis * np.sqrt(gamma)) @ basis.T
    transport = inverse_square @ middle_root @ inverse_square
    transport = (transport + transport.T) * .5
    residual = np.linalg.norm(transport @ covariance @ transport - reference_covariance, 'fro') / np.linalg.norm(reference_covariance, 'fro')
    if not np.isfinite(residual) or residual > 1e-8:
        raise ValueError(f'numerically inaccurate transport square root: relative residual {residual}; no repair allowed')
    delta = mean - reference_mean
    mean_term = float(delta @ delta)
    covariance_term = float(np.trace(covariance) + np.trace(reference_covariance) - 2*np.sqrt(gamma).sum())
    fid = mean_term + covariance_term
    if not np.isfinite(fid) or fid <= 0:
        raise ValueError(f'FID is nonpositive/nonfinite ({fid}); ratio/first-order inference is not supported at equality')
    return fid, 2*delta, np.eye(len(mean)) - transport, {
        'mean_term': mean_term, 'covariance_term': covariance_term,
        'sample_spd': sample_audit, 'reference_spd': reference_audit,
        'middle_spd': middle_audit, 'transport_relative_residual': float(residual)}


def features_influence(features, reference_mean, reference_covariance):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) <= x.shape[1]:
        raise ValueError('features must be finite [N,d] with N>d; no ridge or rank reduction allowed')
    n = len(x)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / (n-1)
    fid, gm, gs, audit = fid_and_gradients(mean, covariance, reference_mean, reference_covariance)
    alpha = n/(n-1)
    linear = centered @ gm
    quadratic = np.einsum('ij,ij->i', centered @ gs, centered)
    trace_gradient_covariance = float(np.sum(gs * covariance))
    influence = linear + alpha*quadratic - trace_gradient_covariance
    # The empirical mean of the centered influence must be zero without repair.
    tolerance = 1000*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(influence))))*x.shape[1]
    if abs(influence.mean()) > tolerance:
        raise ValueError('influence centering identity failed')
    return {'fid':fid,'influence':influence,'mean':mean,'covariance':covariance,
        'mean_gradient':gm,'covariance_gradient':gs,'alpha':alpha,'audit':audit}


def stratified_covariance(influences, labels):
    """Covariance of paired scalar estimators under fixed class allocation.

    Row i is the complete vector of arm/contrast influences for the same noise ID.
    Class c has weight m_c/N. Variance contribution is (m_c/N)^2 S_c/m_c.
    """
    values = np.asarray(influences, np.float64)
    if values.ndim == 1:
        values = values[:,None]
    labels = np.asarray(labels)
    if labels.shape != (len(values),) or not np.isfinite(values).all():
        raise ValueError('invalid influence/label alignment')
    classes, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError('each fixed stratum requires at least two noise replicates')
    covariance = np.zeros((values.shape[1],values.shape[1]),np.float64)
    within_variances = []
    for label,count in zip(classes,counts):
        block = values[labels==label]
        center = block - block.mean(0)
        within = center.T @ center / (count-1)
        covariance += count/len(values)**2 * within
        within_variances.append(np.diag(within))
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance)<0):
        raise ValueError('invalid stratified variance')
    return covariance, classes, counts, np.asarray(within_variances)


def paired_contrast(candidate, baseline, labels):
    fa,fb = float(candidate['fid']),float(baseline['fid'])
    if fb<=0:
        raise ValueError('relative gain requires positive baseline FID')
    difference_if = candidate['influence'] - baseline['influence']
    relative_if = -candidate['influence']/fb + fa*baseline['influence']/(fb*fb)
    matrix, classes, counts, variances = stratified_covariance(np.column_stack([difference_if,relative_if]),labels)
    result = {}
    for index,(name,value) in enumerate((('fid_difference_candidate_minus_baseline',fa-fb),('relative_improvement_one_minus_candidate_over_baseline',1-fa/fb))):
        se=float(np.sqrt(matrix[index,index]))
        result[name]={'estimate':value,'stratified_first_order_se':se,
            'local_normal_95_interval':[value-1.96*se,value+1.96*se]}
    return result, difference_if, relative_if, variances
