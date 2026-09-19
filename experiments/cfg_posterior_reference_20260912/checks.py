"""Posterior averaging versus independent-label averaging in an exact mixture."""
import numpy as np
from experiments.sit_strong_reference_20260912.checks import analytic_checks as projection_checks


def analytic_checks():
    records=[]
    for mean,sigma in ((2.,1.),(.8,1.),(1.,.3)):
        z=np.linspace(-5,5,201)
        logits=np.stack((-(z+mean)**2/(2*sigma*sigma),-(z-mean)**2/(2*sigma*sigma)))
        weights=np.exp(logits-logits.max(0));weights/=weights.sum(0)
        scores=np.stack((-(z+mean)/(sigma*sigma),-(z-mean)/(sigma*sigma)))
        posterior=(weights*scores).sum(0)
        marginal=-(z-mean*np.tanh(mean*z/(sigma*sigma)))/(sigma*sigma)
        independent=scores.mean(0);geometric=-z/(sigma*sigma)
        np.testing.assert_allclose(posterior,marginal,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(independent,geometric,rtol=1e-12,atol=1e-12)
        assert np.max(np.abs(posterior-independent))>.1
        records.append(dict(class_mean=mean,sigma=sigma,
            posterior_identity_max_error=float(np.max(np.abs(posterior-marginal))),
            prior_vs_posterior_max_difference=float(np.max(np.abs(posterior-independent)))))
    return dict(passed=True,mixture_checks=records,projection=projection_checks(),
        true_label_given_state_is_posterior_distributed=True,
        independent_labels_do_not_estimate_arithmetic_mixture_score=True,
        note='Exact conditional scores and population label-blind regression only; not a claim about all ICG models or generated quality.')
