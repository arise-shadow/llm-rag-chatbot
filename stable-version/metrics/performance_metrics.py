import time
import psutil
import torch
import subprocess


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
    Calculates memory usage for CUDA or NPU devices.

    Parameters:
        device (str): Device to check memory usage. Examples:
                      - "cuda" for GPU
                      - "npu:0:*" for NPU (Furiosa runtime format)

    Returns:
        float: Memory usage in MB.
    """
    if "cuda" in device:  # GPU (CUDA)
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            print("CUDA device not available.")
            return 0.0
    elif "npu" in device:  # NPU (Furiosa)
        import subprocess
        result = subprocess.run(
            ["furiosa-smi", "info", "--device", device],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            output = result.stdout.decode()
            # Parse Furiosa memory usage (example parsing logic)
            memory_line = [line for line in output.split("\n") if "Memory Usage" in line][0]
            memory_used = float(memory_line.split(":")[1].strip().split()[0])  # Assuming "Memory Usage: 512 MB"
            return memory_used
        except (IndexError, ValueError):
            print(f"Failed to parse memory usage for device {device}.")
            return 0.0
    else:
        print(f"Unsupported device type: {device}")
        return 0.0



def calculate_power_consumption(device: str) -> float:
    """
    Calculates power consumption for CUDA or NPU devices.

    Parameters:
        device (str): Device to check power usage. Examples:
                      - "cuda" for GPU
                      - "npu:0:*" for NPU (Furiosa runtime format)

    Returns:
        float: Power consumption in watts.
    """
    if "cuda" in device:  # GPU (CUDA)
        try:
            # Extract GPU ID from the device string (e.g., "cuda:0" -> 0)
            gpu_id = device.split(":")[1]
            
            # Run nvidia-smi command for power usage
            result = subprocess.run(
                ["nvidia-smi", "--id", gpu_id, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse the power consumption value
            power_str = result.stdout.strip()  # Remove extra whitespace
            if power_str:
                return float(power_str)  # Convert string to float
            else:
                raise ValueError("Power consumption value not found.")
        except Exception as e:
            print(f"Error calculating power consumption for device {device}: {e}")
            return 0.0

    elif "npu" in device:  # NPU (Furiosa)
        try:
            result = subprocess.run(
                ["furiosa-smi", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            output = result.stdout
            # Filter lines related to the target NPU device
            target_line = [line for line in output.split("\n") if device in line]
            if not target_line:
                raise ValueError(f"Device {device} not found in furiosa-smi info output.")
            # Extract Power value (example: "| rngd | npu0   | ... | 41.00 W | ... |")
            power_str = target_line[0].split("|")[5].strip()  # Power is the 6th column
            power_value = float(power_str.split()[0])  # Extract numeric value (e.g., "41.00")
            return power_value
        except Exception as e:
            print(f"Error calculating power consumption for NPU {device}: {e}")
            return 0.0
    else:
        print(f"Unsupported device type: {device}")
        return 0.0