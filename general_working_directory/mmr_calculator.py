import json


def find_mrr(data_):
    rr = 0
    n = len(data_)
    for i, v in data_.items():
        rr += 1 / (v + 1)
    mrr = rr / n
    return mrr


print("==== Embedding-based model ====/n")

with open('evaluation_results_embedding-based_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")

with open('evaluation_results_embedding-based_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")

print("==== BM25 model ====/n")
with open('evaluation_results_bm25_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")

with open('evaluation_results_bm25_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
