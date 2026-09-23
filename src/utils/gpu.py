"""GPU configuration utilities for TensorFlow."""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger(__name__)


def configure_gpu(memory_growth: bool = True, memory_limit: Optional[int] = None) -> bool:
    """Configure GPU settings for TensorFlow.

    This function should be called at the start of any script before
    creating models or loading data.

    Args:
        memory_growth: If True, enables memory growth to avoid allocating
                      all GPU memory at once. Recommended for most use cases.
        memory_limit: Optional memory limit in MB. If set, restricts TensorFlow
                     to use only this much GPU memory.

    Returns:
        True if GPU is available and configured, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("No GPU detected. Running on CPU.")
        return False

    try:
        for gpu in gpus:
            if memory_growth:
                # Enable memory growth to prevent TF from allocating all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Enabled memory growth for {gpu.name}")

            if memory_limit is not None:
                # Set a hard memory limit
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                logger.info(f"Set memory limit to {memory_limit}MB for {gpu.name}")

        logger.info(f"GPU configuration complete. {len(gpus)} GPU(s) available.")
        return True

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(f"GPU configuration failed: {e}")
        return False


def get_device_info() -> dict:
    """Get information about available compute devices.

    Returns:
        Dictionary with device information.
    """
    info = {
        'tensorflow_version': tf.__version__,
        'cuda_available': tf.test.is_built_with_cuda(),
        'gpu_available': tf.config.list_physical_devices('GPU'),
        'num_gpus': len(tf.config.list_physical_devices('GPU')),
    }

    # Get detailed GPU info if available
    if info['num_gpus'] > 0:
        try:
            from tensorflow.python.client import device_lib
            local_devices = device_lib.list_local_devices()
            gpu_details = [
                {
                    'name': d.name,
                    'device_type': d.device_type,
                    'memory_limit': d.memory_limit,
                }
                for d in local_devices if d.device_type == 'GPU'
            ]
            info['gpu_details'] = gpu_details
        except Exception:
            pass

    return info


def log_device_placement(enable: bool = True):
    """Enable or disable logging of device placement for operations.

    Useful for debugging to see which device (CPU/GPU) each operation runs on.

    Args:
        enable: Whether to enable device placement logging.
    """
    tf.debugging.set_log_device_placement(enable)
