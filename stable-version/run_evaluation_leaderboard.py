import pandas as pd
import os
import json
import numpy as np
from metrics.evaluation_transition import evaluate_translation

# 평가를 위해 reference text 가져오기 
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



# Configurations
translation_config = 'ko2en'
target_lang = "English"  # 목표 언어

# 과제 이름 설정
task_names = ["KO→EN Translation", "EN→KO Translation", "Classification Task 1", "Classification Task 2"]


data_dir = f"../data/flores/"
result_dir = f"../result/translate/{translation_config}/"
leaderboard_file = f"../result/translate/{translation_config}/leaderboard.csv"

# 기존 리더보드 파일 로드 또는 새로 생성
if os.path.exists(leaderboard_file):
    leaderboard = pd.read_csv(leaderboard_file)
else:
    leaderboard = pd.DataFrame(columns=[
        "device-type", "device-name", "llm", "quantization", "calibration",
        "BLEU", "METEOR", "BERTScore", "tps"
    ])
    
# Load reference data
data_eng = load_text_file(f"{data_dir}/devtest.eng_Latn")
data_kor = load_text_file(f"{data_dir}/devtest.kor_Hang")

active_metrics = ['BLEU', 'METEOR', 'BERTScore', 'tps']

# Process each JSON file
for filename in os.listdir(result_dir):
    if not filename.endswith(".json"):
        continue

    # Parse metadata from filename
    name_head = filename.replace(".json", "")
    metadata = {
        "device-type": name_head.split("-")[0],
        "device-name": name_head.split("_")[0].split("-")[1],
        "llm": name_head.split("_")[1].split('-')[0],
        "quantization": name_head.split("_calib")[0].split('-')[-1],
        "calibration": name_head.split("_calib-")[1],
    }

    # Skip if already in leaderboard
    if ((leaderboard["device-type"] == metadata["device-type"]) &
        (leaderboard["device-name"] == metadata["device-name"]) &
        (leaderboard["llm"] == metadata["llm"]) &
        (leaderboard["quantization"] == metadata["quantization"]) &
        (leaderboard["calibration"] == metadata["calibration"])).any():
        print(f"Skipping {filename}, already in leaderboard.")
        continue
    
    # Load translation results
    with open(os.path.join(result_dir, filename), "r", encoding="utf-8") as file:
        json_data = json.load(file)
        translations = json_data.get("translations", [])

    # Evaluate translations
    num_metrics = 4  # BLEU, METEOR, BERTScore, TPS
    metrics = np.full((num_metrics, len(translations)), np.nan)

    for i, result in enumerate(translations):
        translation = result.get("translation", "")
        elapsed_time = result.get("elapsed_time", 1e-6)  # Default time if not provided
        ref_text = data_eng[i] if i < len(data_eng) else ""

        # Evaluate translation
        metric_result = evaluate_translation(
            translation, ref_text, target_lang, elapsed_time, active_metrics
        )
        for j, metric_name in enumerate(active_metrics):
            metrics[j, i] = metric_result.get(metric_name, np.nan)

    # Calculate averages
    avg_metrics = {metric: np.nanmean(metrics[j, :]) for j, metric in enumerate(active_metrics)}

    # Add to leaderboard
    new_entry = pd.DataFrame([{
        **metadata,
        **avg_metrics
    }])  # Create a DataFrame for the new entry

    # Concatenate the new entry to the leaderboard
    leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
    print(f"Processed and added {filename} to leaderboard.")

# 리더보드 CSV 파일로 저장
leaderboard.to_csv(leaderboard_file, index=False)
print(f"Leaderboard updated and saved to {leaderboard_file}")


# CSV 파일 경로 설정
file_paths = [
    "../result/translate/ko2en/leaderboard.csv",
    "../result/translate/en2ko/leaderboard.csv",
    "../result/translate/ko2en/leaderboard.csv",
    "../result/translate/en2ko/leaderboard.csv",
]

# 테이블들을 HTML로 연속 추가
tables_content = ""
for i, (file_path, task_name) in enumerate(zip(file_paths, task_names)):
    # CSV 데이터 로드 및 소수점 세 자리 반올림
    data = pd.read_csv(file_path).round(3)

    # 테이블 헤더와 행 생성
    headers = "".join(f"<th>{col}</th>" for col in data.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in data.values
    )

    # 테이블 섹션 추가
    tables_content += f"""
    <section>
        <h2>{task_name}</h2>
        <table id="leaderboard-{i}" class="display">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </section>
    """

# HTML 파일 생성
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Leaderboard</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: #E0E0E0;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #B0B0B0;
            margin: 0;
        }}
        .logo {{
            height: 50px;
        }}
        section {{
            margin-bottom: 40px;
        }}
        h2 {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 10px;
            color: #B0B0B0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background-color: #323232;
            color: #FFFFFF;
            padding: 10px;
            text-align: center;
        }}
        td {{
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #2C2C2C;
        }}
        tr:hover {{
            background-color: #424242;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Service Leaderboard</h1>
        <img src="ac6c6784959948e1aa377e8b01cfed51.webp" alt="Furiosa Logo" class="logo">
    </div>
    {tables_content}

    <!-- DataTables Script -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script>
        $(document).ready(function() {{
            // Initialize DataTables for all tables
            {''.join([f'$("#leaderboard-{i}").DataTable({{ "paging": true, "pageLength": 10, "info": false, "lengthChange": false, "searching": false, "order": [[7, "desc"]] }});' for i in range(len(file_paths))])}
        }});
    </script>
</body>
</html>
"""

# HTML 파일 저장
with open("../docs/leaderboard_recent.html", "w", encoding="utf-8") as file:
    file.write(html_content)