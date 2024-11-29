import json


def find_mrr(data):
    rr = 0
    n = len(data)
    for i, v in data.items():
        rr += 1 / (v + 1)
    mrr = rr / n
    return mrr


print("==== Embedding-based model ====/n")


def find_hit_k(data, k):
    count = 0
    n = len(data)
    for i, v in data.items():
        if v + 1 < k:
            count += 1
    return count / n


k = 20
with open('evaluation_results_embedding-based_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k} = {find_hit_k(data, k)}")

with open('evaluation_results_embedding-based_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k} = {find_hit_k(data, k)}")

print("==== BM25 model ====/n")
with open('evaluation_results_bm25_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k} = {find_hit_k(data, k)}")

with open('evaluation_results_bm25_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k} = {find_hit_k(data, k)}")

