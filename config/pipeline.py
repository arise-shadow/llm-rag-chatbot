# pipeline.py
import yaml

# YAML 파일 읽기
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 실행용 파이프라인 생성
pipeline = {
    "device": config["device"]["type"],
    "model": config["model"]["name"],
    "quantization": config["model"]["quantization"],
    "input_dataset": config["data"]["input_dataset"],
    "calib_dataset": config["data"]["calib_dataset"],
    "task": config["evaluation"]["task"],
    "metrics": config["evaluation"]["metrics"],
    "output_dir": config["evaluation"]["output_dir"]
}