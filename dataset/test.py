import json

with open('dataset/dummy_data.jsonl', 'r', encoding='utf8') as jsnl:
    v = jsnl.readlines(2.2e9)

v = [json.loads(i) for i in v]

formatDict = {
    "Instruction": "",
    "Input": "",
    "Response": ""
}

li = []

print(v[0])
print(type(v[0]))
for data in v:
    for i in data['conversation']:
        li.append({
        "Instruction": i['human'],
        "Input": "",
        "Response": i['assistant']
    })

with open('dummy_after.jsonl', 'w', encoding='utf8') as jsnl:
    for i in li:
        json.dump(i, jsnl, ensure_ascii=False)
        jsnl.write('\n')