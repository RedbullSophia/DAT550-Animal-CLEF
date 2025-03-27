def get_max_batch_size(model_name="resnet18", m=2, memory_limit_gb=7.5):
    """
    Estimate a safe batch size based on model type, M value, and available GPU VRAM.
    
    Args:
        model_name (str): Backbone you're using
        m (int): MPerClassSampler value
        memory_limit_gb (float): How much of GPU memory to safely use (out of 8GB)

    Returns:
        int: Suggested batch size
    """

    # Base memory usage per image for common backbones (approximate, in MB)
    per_image_memory = {
        "resnet18": 10,
        "mobilenetv3_small_100": 6,
        "efficientnet_b0": 12,
        "resnet50": 20,
        "convnext_tiny": 24
    }

    # Fallback default
    base_usage = per_image_memory.get(model_name, 12)

    # Total memory available in MB
    total_gpu_mem_mb = memory_limit_gb * 1024

    # Estimate how many images we can safely fit
    est_batch_size = int(total_gpu_mem_mb / base_usage)

    # Round down to be divisible by m * 2
    est_batch_size = (est_batch_size // (m * 2)) * (m * 2)

    return max(8, est_batch_size)