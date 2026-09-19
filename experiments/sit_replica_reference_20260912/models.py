from experiments.sit_reference_compilation_20260912.models import Capture,patchify,unpatchify
from experiments.sit_reference_compilation_20260912 import models as previous
from . import catalog as c


def make_head(rt,method,codewords=256):
    return previous.make_head(rt,'ig_shallow' if method in c.METHODS else method,codewords)
