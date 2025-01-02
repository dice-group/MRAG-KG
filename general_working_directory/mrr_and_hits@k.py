import json


def find_mrr(data):
    rr = 0
    n = len(data)
    for i, v in data.items():
        rr += 1 / (v + 1)
    mrr = rr / n
    return mrr


def find_hit_k(data, k):
    count = 0
    n = len(data)
    for i, v in data.items():
        if v + 1 < k:
            count += 1
    return count / n


k1 = 10
k2 = 20
k3 = 50
k4 = 100

print("==== embedding-based model b1====/n")
with open('evaluation_results_embedding-based_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG Hits@{k4} = {find_hit_k(data, k4)}")

with open('evaluation_results_embedding-based_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG Hits@{k4} = {find_hit_k(data, k4)}")

print("==== BM25 model b1====/n")
with open('evaluation_results_bm25_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG Hits@{k4} = {find_hit_k(data, k4)}")

with open('evaluation_results_bm25_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG Hits@{k4} = {find_hit_k(data, k4)}")

print("==== embedding-based model b2====/n")
with open('evaluation_results2_embedding-based_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG Hits@{k4} = {find_hit_k(data, k4)}")

with open('evaluation_results2_embedding-based_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG Hits@{k4} = {find_hit_k(data, k4)}")

print("==== BM25 model b2====/n")
with open('evaluation_results2_bm25_on_second_KG.json', 'r') as file:
    data = json.load(file)

print(f"Second KG MRR = {find_mrr(data)}")
print(f"Second KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG Hits@{k4} = {find_hit_k(data, k4)}")

with open('evaluation_results2_bm25_on_third_KG.json', 'r') as file:
    data = json.load(file)

print(f"Third KG MRR = {find_mrr(data)}")
print(f"Third KG Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG Hits@{k4} = {find_hit_k(data, k4)}")

print("==== CEL-based 20-20 lp ====/n")
with open('rag2_eval_second_kg.json', 'r') as file:
    data = json.load(file)

print(f"Second KG + CEL 20-20 MRR = {find_mrr(data)}")
print(f"Second KG + CEL 20-20 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG + CEL 20-20 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG + CEL 20-20 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG + CEL 20-20 Hits@{k4} = {find_hit_k(data, k4)}")

with open('rag2_eval_third_kg.json', 'r') as file:
    data = json.load(file)

print(f"Third KG + CEL 20-20 MRR = {find_mrr(data)}")
print(f"Third KG + CEL 20-20 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG + CEL 20-20 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG + CEL 20-20 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG + CEL 20-20 Hits@{k4} = {find_hit_k(data, k4)}")

print("==== CEL-based 10-10 lp ====/n")

with open('rag2_eval_second_kg-10-10.json', 'r') as file:
    data = json.load(file)

print(f"Second KG + CEL 10-10 MRR = {find_mrr(data)}")
print(f"Second KG + CEL 10-10 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG + CEL 10-10 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG + CEL 10-10 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG + CEL 10-10 Hits@{k4} = {find_hit_k(data, k4)}")

with open('rag2_eval_third_kg-10-10.json', 'r') as file:
    data = json.load(file)

print(f"Third KG + CEL 10-10 MRR = {find_mrr(data)}")
print(f"Third KG + CEL 10-10 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG + CEL 10-10 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG + CEL 10-10 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG + CEL 10-10 Hits@{k4} = {find_hit_k(data, k4)}")

print("==== CEL-based 5-5 lp ====/n")

with open('rag2_eval_second_kg-5-5.json', 'r') as file:
    data = json.load(file)

print(f"Second KG + CEL 5-5 MRR = {find_mrr(data)}")
print(f"Second KG + CEL 5-5 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Second KG + CEL 5-5 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Second KG + CEL 5-5 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Second KG + CEL 5-5 Hits@{k4} = {find_hit_k(data, k4)}")

with open('rag2_eval_third_kg-5-5.json', 'r') as file:
    data = json.load(file)

print(f"Third KG + CEL 5-5 MRR = {find_mrr(data)}")
print(f"Third KG + CEL 5-5 Hits@{k1} = {find_hit_k(data, k1)}")
print(f"Third KG + CEL 5-5 Hits@{k2} = {find_hit_k(data, k2)}")
print(f"Third KG + CEL 5-5 Hits@{k3} = {find_hit_k(data, k3)}")
print(f"Third KG + CEL 5-5 Hits@{k4} = {find_hit_k(data, k4)}")
