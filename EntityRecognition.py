import json


with open("train-v2.0.json") as f:
    data = json.load(f)
for i in range(20):
    print(data["data"][i]["paragraphs"])