import yaml

# YAML 파일 읽기
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 평가 실행
def performance_test(pipeline, task, yaml_config):
    # 기본 메트릭 평가
    results = {
        "metrics": {}, 
        "performance": {}
    }
    
    # 기존 메트릭 실행 (예: BLEU, BERTScore)
    if task == "translation":
        results["metrics"]["BLEU"] = compute_bleu(pipeline)
        results["metrics"]["BERTScore"] = compute_bertscore(pipeline)

    # 성능 메트릭 계산
    if yaml_config["evaluation"]["performance_metrics"]["power_consumption"]:
        results["performance"]["power_consumption"] = measure_power(pipeline)

    if yaml_config["evaluation"]["performance_metrics"]["memory_usage"]:
        results["performance"]["memory_usage"] = measure_memory(pipeline)

    if yaml_config["evaluation"]["performance_metrics"]["fps"]:
        results["performance"]["fps"] = measure_fps(pipeline)

    return results

# 성능 측정 함수 (가상 구현)
def measure_power(pipeline):
    # 전력 소모량 측정 로직 (Watt 단위)
    return "150W"

def measure_memory(pipeline):
    # 메모리 사용량 측정 로직 (GB 단위)
    return "8GB"

def measure_fps(pipeline):
    # FPS 측정 로직
    return "60 FPS"

# 실행
pipeline = {
    "device": config["device"],
    "model": config["model"],
    "data": config["data"]
}
task = config["evaluation"]["task"]
results = performance_test(pipeline, task, config)

# 출력 결과
print(results)