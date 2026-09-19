import copy
from experiments.sit_reference_compilation_20260912.models import Capture,patchify,unpatchify,categorical_delta
from experiments.sit_reference_compilation_20260912 import models as previous,core as previous_core
from . import catalog as c


def make_head(rt,method,codewords=256):
    if method not in c.METHODS:return previous.make_head(rt,method,codewords)
    module=copy.deepcopy(previous_core.get_head(rt,'ig_shallow'))
    for layer in module.modules():
        layer._forward_hooks.clear();layer._forward_pre_hooks.clear()
    return module.eval()
