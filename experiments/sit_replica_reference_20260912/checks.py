"""Population risk identities, independent of image quality evaluation."""
import numpy as np


def analytic_checks():
    rng = np.random.default_rng(2026121234)
    x = np.array([[-1.3, -.2], [.4, 1.6], [1.1, -.9]])
    p = np.array([.2, .5, .3])
    # A finite corruption kernel permits an exact, nonparametric risk check.
    K = rng.uniform(.01, 1., (3, 11)); K /= K.sum(1, keepdims=True)
    q = p@K
    posterior = (p[:, None]*K/q).T
    T = posterior@K
    m = posterior@x
    d = np.linalg.solve(2*np.eye(11)-T, m)
    a = np.linalg.solve(np.eye(11)+T, 2*m)
    residual = max(np.max(np.abs((2*np.eye(11)-T)@d-m)),
                   np.max(np.abs((np.eye(11)+T)@a-2*m)))
    detailed_balance = np.max(np.abs(q[:, None]*T-(q[:, None]*T).T))
    symmetric = np.sqrt(q)[:, None]*T/np.sqrt(q)[None, :]
    eigenvalues = np.linalg.eigvalsh(symmetric)
    # Direct enumeration of both replica observations, with finite differences.
    def risk(f, method):
        r = f[None, :, :]-x[:, None, :]
        native = .5*np.einsum('i,ij,ijk,ijk->', p, K, r, r)
        difference = f[:, None, :]-f[None, :, :]
        joint = np.einsum('i,ij,ik->jk', p, K, K)
        penalty = .25*np.einsum('ij,ijk,ijk->', joint, difference, difference)
        return native+penalty if method=='consistent' else native-.5*penalty
    finite_difference = 0.
    for f, method in ((d, 'consistent'), (a, 'average')):
        for _ in range(6):
            direction = rng.normal(size=f.shape); step = 1e-5
            derivative = (risk(f+step*direction, method)-risk(f-step*direction, method))/(2*step)
            finite_difference = max(finite_difference, abs(derivative))
    # Gaussian denoisers have prior covariance C/2 and 2C, respectively.
    C = np.array([[1.2, .3], [.3, .8]])
    sigma2 = .7; rho = C@np.linalg.inv(C+sigma2*np.eye(2))
    consistent = np.linalg.solve(2*np.eye(2)-rho, rho)
    average = np.linalg.solve(np.eye(2)+rho, 2*rho)
    gaussian_error = max(np.max(np.abs(consistent-(C/2)@np.linalg.inv(C/2+sigma2*np.eye(2)))),
                         np.max(np.abs(average-(2*C)@np.linalg.inv(2*C+sigma2*np.eye(2)))))
    # Smooth functions need not be densities: J_d involves Cov(shifted X, X),
    # not a symmetric covariance. An exact finite corruption test establishes
    # the risk identities only; a separate Gaussian-kernel check handles curl.
    e1, e2 = rng.normal(size=(2, 17, 3))
    native = .25*(np.mean(e1**2)+np.mean(e2**2))
    average_risk = .5*np.mean(((e1+e2)/2)**2)
    loss_identity = abs(average_risk-(native-.125*np.mean((e1-e2)**2)))
    assert residual<1e-12 and detailed_balance<1e-12 and finite_difference<1e-8
    assert eigenvalues.min()>-1e-12 and eigenvalues.max()<1+1e-12
    assert gaussian_error<1e-12 and loss_identity<1e-12
    return dict(passed=True, resolvent_max_error=float(residual),
        detailed_balance_max_error=float(detailed_balance),
        direct_risk_derivative_max=float(finite_difference),
        operator_eigenvalue_range=[float(eigenvalues.min()), float(eigenvalues.max())],
        gaussian_covariance_identity_max_error=float(gaussian_error),
        averaged_loss_identity_max_error=float(loss_identity),
        no_general_density_smoothing_claim=True)
