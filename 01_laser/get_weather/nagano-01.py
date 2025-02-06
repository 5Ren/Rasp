import requests
import json

area = '200000'
list = []
base_url = "https://www.jma.go.jp/bosai/forecast/data/forecast/"
url = base_url + area + ".json"

res = requests.get(url)

print(res)