import pandas as pd

# CSV 파일 로드
file_path = "/home/dudaji/Jun/llm-rag-chatbot/result/translate/ko2en/leaderboard.csv"
data = pd.read_csv(file_path)

# HTML 테이블 생성
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sortable Leaderboard</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        h1 {{
            text-align: center;
        }}
        .download-btn {{
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #007BFF;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .download-btn:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <h1>AI Model Leaderboard</h1>
    <a href="leaderboard.csv" download class="download-btn">Download CSV</a>
    <table id="leaderboard" class="display">
        <thead>
            <tr>
                {headers}
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <!-- DataTables Script -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script>
        $(document).ready(function() {{
            $('#leaderboard').DataTable({{
                "order": [[7, "desc"]],  // Default sort by TPS column in descending order
            }});
        }});
    </script>
</body>
</html>
"""

# 테이블 헤더와 데이터 생성
headers = "".join(f"<th>{col}</th>" for col in data.columns)
rows = "".join(
    "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
    for row in data.values
)

# HTML 파일 저장
with open("sortable_leaderboard1.html", "w", encoding="utf-8") as file:
    file.write(html_content.format(headers=headers, rows=rows))