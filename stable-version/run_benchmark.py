from time import time
from translate.one_trans import initialize_translation_environment, translate_text, batch_translate_text
from common.load_config import load_all_configurations
from metrics.evaluation_transition import evaluate_translation
import os
import json


config_path = "/home/dudaji/joonwon/llm-rag-chatbot/config/config_gpu_translate.yaml"



# Load configuration
all_configs = load_all_configurations(config_path)

# Set ENABLE_MONITORING=true in the environment to enable monitoring
os.environ["ENABLE_MONITORING"] = "true" if "power_consumption" in all_configs["active_metrics"] else "false"
    
# Print the loaded configurations
print("Configurations Loaded Successfully:")
print("Active Metrics:", all_configs["active_metrics"])
print("Device Config:", all_configs["device_config"])
print("Model Config:", all_configs["model_config"])
print("Evaluation Settings:", all_configs["evaluation_settings"])

# Look up imtermediate result folder to avoid redundancy (performance leaderboard)
name_config = (
    f'{all_configs["device_config"]["type"]}-'
    f'{all_configs["device_config"]["model"]}_'
    f'{all_configs["model_config"]["name"]}-'
    f'{all_configs["model_config"]["quantization"]}_'
    f'calib-{all_configs["model_config"].get("calibration", "none")}'
)
result_folder = os.path.join(all_configs["evaluation_settings"]["output_dir"], "ko2en")
result_file = os.path.join(result_folder, f"{name_config}.json")

# Ensure result folder exists
os.makedirs(result_folder, exist_ok=True)

print("----------------------- \n")
if os.path.exists(result_file):
    print(f"Result file already exists at: {result_file}. Skipping computation.")
    proceed_calc = "true"
else:
    print(f"Result file does not exist. Proceeding with computation...")
    proceed_calc = "false"


# Load the CSV file
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# FLORES-200 데이터 위치 
data_dir = "../data/flores"

data_eng = load_text_file(f"{data_dir}/devtest.eng_Latn")
data_kor = load_text_file(f"{data_dir}/devtest.kor_Hang")



# Initialize the translation environment
try:
    initialize_translation_environment(llm_model=all_configs["model_config"]["name"])
except EnvironmentError as e:
    print(f"Failed to initialize translation environment: {e}")
    exit(1)
    
# 한 -> 영 번역 
num_benchmark = 100
start_time = time() # start time
batch_results = batch_translate_text(data_kor[:num_benchmark], source_language="Korean", target_language="English")
batch_results["elapsed_time"] = time() - start_time  # Add elapsed time to the results
result_folder = os.path.join(all_configs["evaluation_settings"]["output_dir"], "ko2en")
result_file = os.path.join(result_folder, f"{name_config}.json")
print('ko2en translation done')

# Save intermediate result 
with open(result_file, "w") as file:
    json.dump(batch_results, file, indent=4)  # Save the batch results in JSON format
    print(f"Results saved to {result_file}.")
    
# 영 -> 한 번역 
start_time = time()
batch_results = batch_translate_text(data_eng[:num_benchmark], source_language="English", target_language="Korean")
batch_results["elapsed_time"] = time() - start_time  # Add elapsed time to the results
result_folder = os.path.join(all_configs["evaluation_settings"]["output_dir"], "en2ko")
result_file = os.path.join(result_folder, f"{name_config}.json")
print('en2ko translation done')

# Save intermediate result 
with open(result_file, "w") as file:
    json.dump(batch_results, file, indent=4)  # Save the batch results in JSON format
    print(f"Results saved to {result_file}.")
