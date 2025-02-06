import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# XMLファイルのパス
xml_file_path = r"E:\apple_health_export\export.xml"  # 適宜変更

# 今日から3ヶ月前の日付を取得
three_months_ago = datetime.now() - timedelta(days=90)

# 逐次読み込み & フィルタ処理
context = ET.iterparse(xml_file_path, events=("end",))  # ← ここでタプルを明示

for event, elem in context:
    if elem.tag == "Record":
        record_attrib = elem.attrib

        # 'creationDate' を取得してパース
        creation_date_str = record_attrib.get("creationDate", "")
        try:
            creation_date = datetime.strptime(creation_date_str, "%Y-%m-%d %H:%M:%S %z")
        except ValueError:
            continue  # パースに失敗したらスキップ

        # 過去3ヶ月以内のデータのみ出力
        if creation_date >= three_months_ago:
            print(f"Tag: {elem.tag}")
            print(f"Attributes: {record_attrib}")
            print("-" * 50)

        # メモリ節約のため、処理後に要素を削除
        elem.clear()
