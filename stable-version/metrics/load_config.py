import yaml

def load_active_metrics(config_path: str):
    """
    Load active metrics (set to True) from the config file.

    Parameters:
        config_path (str): Path to the config YAML file.

    Returns:
        list: A list of active metrics.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract metrics from the config
    metrics_config = config.get("evaluation", {}).get("metrics", {})
    active_metrics = [metric for metric, is_active in metrics_config.items() if is_active]

    return active_metrics