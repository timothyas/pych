
def get_nice_attrs(fldname):
    bigdict = {
    'Nx': {
        'label': r'$N_x$',
        'description': 'integer correlation length in terms of '+\
                       'number of neighboring grid cells'},
    'xi': {
        'label': r'$\xi$',
        'description': 'factor amplifying correlation lengths in horizontal, '+\
                       'relative to vertical'},
    'sigma': {
        'label': r'$\sigma$',
        'description': 'parameter uncertainty, standard deviation'},
    'betaSq': {
        'label': r'$\beta^2$',
        'description': 'regularization parameter'},

    # --- Interpolation
    'F': {
        'label': r'$F$',
        'description': 'linear interpolation operator from parameter '+\
                       'to observation space'},

    # --- matern prior
    'delta': {
        'label': r'$\delta$',
        'description': 'spatially varying term in Matern PDE operator'},
    'Phi': {
        'label': r'$\Phi$',
        'description': 'jacobian of mapping from isotropic, stationary space to '+\
                       'the real world'},
    'randNorm': {
        'label': r'det$(\Phi(\mathbf{x}))^{1/2}$',
        'description': 'normalization for white noise right hand side in Matern '+\
                       'PDE operator'}, 
    'Kux': {
        'label': r'$\kappa_{ux}$',
        'description': r'x-term in Laplacian tensor Kappa, defined through $\Phi$'},
    'Kvy': {
        'label': r'$\kappa_{vy}$',
        'description': r'y-term in Laplacian tensor Kappa, defined through $\Phi$'},
    'Kwz': {
        'label': r'$\kappa_{wz}$',
        'description': r'z-term in Laplacian tensor Kappa, defined through $\Phi$'},
    'filternorm': {
        'label': r'$X$',
        'description': 'normalization factor: '+\
                       'inverse of point-wise sample standard deviation of '+\
                       'inverse differential operator'},

    # --- REVD
    'Y': {
        'label': r'$Y$',
        'description': 'range approximation matrix'},
    'Q': {
        'label': r'$Q$',
        'description': 'orthonormal basis of matrix range (parameter space)'},
    'V': {
        'label': r'$V$',
        'description': 'Eigenvectors of prior preconditioned misfit Hessian, '+\
                       'forms basis for reduced subspace in control space'},
    'Dorig': {
        'label': r'$\hat\lambda$',
        'description': r'Eigenvalues of prior preconditioned misfit Hessian, '+\
                        'not scaled by $\sigma^2$ (parameter uncertainty)'},
    'D': {
        'label': r'$\lambda$',
        'description': 'Eigenvalues of prior preconditioned misfit Hessian'},
    'Dinv': {
        'label': r'$D$',
        'description': r'$\dfrac{\lambda_i}{\lambda_i + 1}$, '+\
                        'used to get inverse Hessian'},

    # --- MAP point and misfits
    'm_map': {
        'label': r'$\mathbf{m}_{MAP}$',
        'description': r'Maximum a Posteriori solution for control parameter $\mathbf{m}$'},
    'reg_norm': {
        'label': r'$||\mathbf{m}_{MAP} - \mathbf{m}_0||_{\Gamma_{prior}^{-1}}$',
        'label2': r'$||\Gamma_{prior}^{-1/2}(\mathbf{m}_{MAP} - \mathbf{m}_0)||_2$',
        'description': 'Normed difference between initial and MAP solution, '+\
                       'weighted by prior uncertainty'},
    'misfits': {
        'label': r'$F\mathbf{m}_{MAP} - \mathbf{d}$',
        'description': 'Difference between MAP solution and observations'},
    'misfits_normalized': {
        'label': r'$\dfrac{F\mathbf{m}_{MAP} - \mathbf{d}}{\sigma_{obs}}$',
        'description': 'Difference between MAP solution and observations, '+\
                       'normalized by observation uncertainty'},
    'misfit_norm': {
        'label': r'$||F\mathbf{m}_{MAP} - \mathbf{d}||_{\Gamma_{obs}^{-1}}$',
        'label2': r'$||\Gamma_{obs}^{-1/2}(F\mathbf{m}_{MAP} - \mathbf{d})||_2$',
        'description': 'Normed difference between MAP solution and observations, '+\
                       'weighted by observational uncertainty'},
    'misfits_model_space': {'label': r'$\mathbf{m}_{MAP} - F^T\mathbf{d}$',
            'description': 'Difference between MAP solution and observations, '+\
                           'in model domain'},
    }
    return bigdict[fldname] if fldname in bigdict.keys() else {}
