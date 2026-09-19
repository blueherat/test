"""Original ADM evaluator with explicit CUDA FP32 execution (TF32 disabled)."""

from pathlib import Path
import runpy
import sys
import tensorflow as tf


devices = tf.config.list_physical_devices('GPU')
if len(devices) != 1:
    raise RuntimeError('Expected one explicitly assigned evaluation GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.experimental.enable_tensor_float_32_execution(False)
source = Path(__file__).resolve().parents[1] / 'compute_adm_fid.py'
sys.argv[0] = str(source)
runpy.run_path(str(source), run_name='__main__')
