import subprocess


def detect_environment() -> str:
    """
    Checks the current system environment to determine if it's GPU-based or Furiosa RNGD.

    Returns:
        str: The detected environment ('gpu', 'furiosa', or 'unknown').
    """
    try:
        # Check for GPU using NVIDIA-smi command
        gpu_check = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if gpu_check.returncode == 0:
            return "gpu"
    except FileNotFoundError:
        pass

    try:
        # Check for Furiosa RNGD using a specific command (e.g., checking for RNGD binaries)
        furiosa_check = subprocess.run(
            ["furiosa-smi", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if furiosa_check.returncode == 0:
            return "furiosa"
    except FileNotFoundError:
        pass

    return "unknown"

if __name__ == "__main__":
    environment = check_system_environment()
    # print(environment)
