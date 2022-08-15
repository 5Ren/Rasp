import requests
import json
import pprint

API_TOKEN = "b66621f4e663cb3813b4e3ad773c0f10"

if __name__ == "__main__":
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            ## 緯度・軽度を指定する場合
            # "lat": "35.68944",
            # "lon": "139.69167",

            ## 都市名で取得する場合
            "q": "tokyo",

            "appid": API_TOKEN,
            "units": "metric",
            "lang": "ja",
        },
    )
    ret = json.loads(response.text)
    pprint.pprint(ret)