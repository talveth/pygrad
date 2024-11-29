
"""
Module storing library-wide global constants. Define the precision of .value and .grad here.
"""

# define what data type you want
import numpy as np

PRECISION = np.float64
# one of np.float32, np.float64, np.float128
# if not using softmax .grad for 3D tensors, can use np.float16
# np.float16 will have too many precision errors to run the transformer model, use >32.
