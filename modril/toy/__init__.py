from modril.toy.toy_tasks import *

TASK_REGISTRY = {
    # 1D
    'sine': Sine1D,
    'multi_sine': MultiSine1D,
    'gauss_sine': GaussSine1D,
    'poly': Poly1D,
    # 2D
    'gaussian_hill': GaussianHill2D,
    'mexican_hat': MexicanHat2D,
    'saddle': Saddle2D,
    'ripple': SinusoidalRipple2D,
    'bimodal_gaussian': BimodalGaussian2D,
}
TASK_LIST = list(TASK_REGISTRY.keys())

MEHTOD_LIST = [
    "gail",
    "drail",
    "mine",
    "nwj",
    "ebil",
    "fm",
    "flowril"
]
ENV_LIST = [
    "static",
    "dynamic"
]