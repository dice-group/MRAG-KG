import numpy as np
import pandas as pd
from declarations import evaluation_samples
from random import sample
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
import json

df = pd.read_csv("benchmark_knn.csv", index_col=0, nrows=None)
iris = df.index.values.tolist()
knn = df.values.tolist()

k = 213  # k in knn
kb = KnowledgeBase(path="fashionpedia-second-generation-v2.owl")
model = TDL(knowledge_base=kb, use_nominals=True, max_runtime=10)


def get_list_of_iris(target_iri):
    target_iri_index = iris.index(target_iri)
    neighbors_indexes = knn[target_iri_index]
    neighbors_iris = list()
    for i in neighbors_indexes:
        neighbor_iri = iris[i]
        neighbors_iris.append(neighbor_iri)
    return neighbors_iris


def get_random_examples(examples_to_avoid):
    iris_to_consider = [iri for iri in iris if iri not in examples_to_avoid]
    random_samples = sample(iris_to_consider, k)
    return random_samples


def get_performance_measurements(individuals, pos, neg):
    assert type(individuals) == type(pos) == type(neg), f"Types must match:{type(individuals)},{type(pos)},{type(neg)}"
    tp = len(pos.intersection(individuals))
    tn = len(neg.difference(individuals))
    fp = len(neg.intersection(individuals))
    fn = len(pos.difference(individuals))
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0
    if precision == 0 or recall == 0:
        return 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f_1 = 2 * ((precision * recall) / (precision + recall))
    return f_1, acc, precision, recall


results = dict()

for iri in evaluation_samples.keys():
    pos_examples = get_list_of_iris(iri)
    neg_examples = get_random_examples(pos_examples)

    typed_pos = set(map(OWLNamedIndividual, pos_examples))
    typed_neg = set(map(OWLNamedIndividual, neg_examples))

    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    prediction = model.fit(learning_problem=lp).best_hypotheses(n=1)
    f1_score, accuracy, precision, recall = get_performance_measurements(
        individuals=set({i for i in kb.individuals(prediction)}),
        pos=typed_pos, neg=typed_neg)
    results[iri] = [f1_score, accuracy, precision, recall]

with open("cel_evaluation.json", "w") as outfile:
    json.dump(results, outfile)

values = np.array(list(results.values()))
mean_values = np.mean(values, axis=0)
print(list(mean_values))
