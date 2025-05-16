import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import build_info as tf_build_info
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")

import tensorflow as tf
print("✅ GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
print(device_lib.list_local_devices())  
print("CUDA version:", tf_build_info.cuda_version_number)
print("cuDNN version:", tf_build_info.cudnn_version_number)