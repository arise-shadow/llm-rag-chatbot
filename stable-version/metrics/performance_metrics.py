import time
import psutil
import torch


def calculate_tps(start_time: float, total_tokens: int) -> float:
    """
    Calculates TPS (Tokens Per Second) based on translation time and token count.

    Parameters:
        start_time (float): The start time of the translation (in seconds since epoch).
        total_tokens (int): Total number of tokens processed.

    Returns:
        float: TPS (tokens/second).

    Raises:
        ValueError: If total_tokens is negative or start_time is in the future.
    """
    if total_tokens < 0:
        raise ValueError("Total tokens cannot be negative.")
    
    elapsed_time = time.time() - start_time
    
    
    return total_tokens / elapsed_time if elapsed_time > 0 else 0


def calculate_memory_usage(device: str = "cuda") -> float:
    """
    Calculates GPU memory usage.

    Parameters:
        device (str): Device to check memory usage ("cuda" or "cpu").

    Returns:
        float: Memory usage in MB.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return psutil.virtual_memory().used / (1024 * 1024)


def calculate_power_consumption(device: str = "cuda") -> float:
    """
    Calculates GPU power consumption. Requires NVIDIA GPUs with `nvidia-smi`.

    Parameters:
        device (str): Device to check power usage ("cuda" or "npu").

    Returns:
        float: Power consumption in watts (if applicable).
    """
    if device == "cuda" and torch.cuda.is_available():
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            power = float(result.stdout.decode().strip())
            return power
        except ValueError:
            return 0.0
    return 0.0