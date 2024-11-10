import json


def find_mrr(data):
    rr = 0
    n = len(data)
    for i, v in data.items():
        rr += 1 / (v + 1)
    mrr = rr / n
    return mrr


with open('../evaluation_results_bm25_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MMR = {find_mrr(data)}")

with open('../evaluation_results_bm25_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MMR = {find_mrr(data)}")