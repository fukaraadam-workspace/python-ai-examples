import time
from typing import Any


def check_opencv() -> dict[str, Any]:
    """
    Check OpenCV availability
    """
    import cv2

    try:
        cv2.namedWindow("test")
        is_headless = False
        cv2.destroyWindow("test")
    except cv2.error:
        is_headless = True

    return {
        "opencv_version": cv2.__version__,
        "is_headless": is_headless,
    }


def check_tensorflow_cuda() -> dict[str, Any]:
    """
    Check TensorFlow CUDA availability and GPU information
    """
    import tensorflow as tf

    # Simulate image tensor
    start_time = time.time()
    tf.reduce_sum(tf.random.normal([1000, 1000]))
    end_time = time.time()
    gpu_devices = tf.config.list_physical_devices("GPU")
    test_tensor_time = f"{(end_time - start_time) * 1000:.2f} ms"

    if gpu_devices:
        device_info = []
        for i, device in enumerate(gpu_devices):
            device_info.append({"index": i, "name": device.name})
        return {
            "is_available": True,
            "tensorflow_version": tf.__version__,
            "device_count": len(gpu_devices),
            "test_tensor_time": test_tensor_time,
            "devices": device_info,
        }
    else:
        return {
            "is_available": False,
            "tensorflow_version": tf.__version__,
            "device_count": 0,
            "test_tensor_time": test_tensor_time,
            "devices": [],
        }


def check_torch_cuda() -> dict[str, Any]:
    """
    Check Torch CUDA availability and GPU information
    """
    import torch

    # Simulate image tensor
    dummy_img = torch.randn(3, 224, 224)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_img = dummy_img.unsqueeze(0).to(device)
    end_time = time.time()
    test_tensor_time = f"{(end_time - start_time) * 1000:.2f} ms"

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = []
        for i in range(device_count):
            device_info.append({"index": i, "name": torch.cuda.get_device_name(i)})
        return {
            "is_available": True,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": device_count,
            "current_device_index": torch.cuda.current_device(),
            "memory_allocated": torch.cuda.memory_allocated(0)
            / 1024**2,  # Convert to MB
            "memory_cached": torch.cuda.memory_reserved(0) / 1024**2,
            "test_tensor_time": test_tensor_time,
            "devices": device_info,
        }
    else:
        return {
            "is_available": False,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device_index": None,
            "test_tensor_time": test_tensor_time,
            "devices": [],
        }


if __name__ == "__main__":
    # print("OpenCV Check:", check_opencv())
    print("TensorFlow CUDA Check:", check_tensorflow_cuda())
    print("Torch CUDA Check:", check_torch_cuda())
