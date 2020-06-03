import json
import spacy
from spacy import displacy


nlp = spacy.load("en_core_web_lg")
with open("/Users/leo/PycharmProjects/EECS4080/train-v2.0.json") as f:
    data = json.load(f)
count = 0
result = []
for i in range(2):
    for d in data["data"][i]["paragraphs"]:
        if count == 20:
            break
        if "context" in d.keys():
            print(d["context"])
            result.append(d["context"])
            print("\n---------------------------------------------------------------------------\n")
            count += 1

print(count)