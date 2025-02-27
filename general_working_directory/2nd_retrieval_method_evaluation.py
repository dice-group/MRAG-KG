import argparse
import csv
from openai import OpenAI
import pandas as pd
import numpy as np
from numpy.linalg import norm
import json
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import TDL
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_property import OWLObjectProperty
from owlapy.render import DLSyntaxObjectRenderer
from declarations import evaluation_samples
import random

def run(args):

    manager = OntologyManager()
    onto_iri = "file://" + args.kg_path
    ontology = manager.load_ontology(IRI.create(onto_iri))
    reasoner = StructuralReasoner(ontology=ontology)
    dprop1 = OWLDataProperty(IRI.create("http://example.org/hasDescription"))
    dprop2 = OWLDataProperty(IRI.create("http://example.org/hasLLMDescription"))
    kb = KnowledgeBase(path="fashionpedia-second-generation-v2.owl")

    llm_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")
    half_lp = int(args.lp_size) / 2
    if "third" in args.kg_path:
        output = f"evaluation_results_2nd_retrieval_method_KG3_{half_lp}_{half_lp}.json"
        third = True
        path_embeddings = "embeddings_third_kg.csv"
    elif "second" in args.kg_path:
        output = f"evaluation_results_2nd_retrieval_method_KG2_{half_lp}_{half_lp}.json"
        third = False
        path_embeddings = "embeddings_second_kg.csv"
    else:
        raise Exception("Please use the path of a file generated by "
                        "'second_kg_generation.py' script or 'third_kg_generation.py' script")


    def verbalize(expression):
        return llm_client.chat.completions.create(
            model="tentris",
            messages=[
                {"role": "system",
                 "content": "You are a description logics translator. Your task is to explain in natural language the description logic expression that is given to you."},
                {"role": "user", "content": [{"type": "text",
                                              "text": f"Explain the following expression in natural language: {expression}. Provide only the explenation in a single sentence."}]}
            ],
            temperature=0.1,
            seed=1
        ).choices[0].message.content


    def get_result(query, docs):
        return llm_client.chat.completions.create(
            model="tentris",
            messages=[
                {"role": "system",
                 "content": "You are a apparel-loving AI and your focus is to give information about apparels. You should find similar points on the assisting information provided to you and present them in a short paragraph tailored to the user's query."},
                {"role": "user", "content": [{"type": "text", "text": f"{query}"}]},
                {"role": "assistant", "content": f"{docs}"},
            ],
            temperature=0.1,
            seed=1
        ).choices[0].message.content


    def find_cosine_similarities(doc, docs):
        docs_norms = docs / norm(docs, axis=1, keepdims=True)
        doc_norms = doc / norm(doc)
        cosine_similarities = (docs_norms @ doc_norms).flatten()
        return cosine_similarities


    def get_LLM_textual_summary(list_of_iris):
        merged_documents = ""
        for iri in list_of_iris:
            image_ind = OWLNamedIndividual(iri)
            if third:
                llm_description = str(list(reasoner.data_property_values(image_ind, dprop2))[0].get_literal())
                if len(llm_description) > 800:
                    llm_description = llm_description[:800]

            all_descriptions = ""
            for d in list(reasoner.data_property_values(image_ind, dprop1)):
                all_descriptions = all_descriptions + d.get_literal() + "\n"

            if third:
                merged_documents += llm_description + " \n" + all_descriptions
            else:
                merged_documents = all_descriptions

        textual_result = get_result(query, merged_documents)
        return textual_result

    df = pd.read_csv(path_embeddings, index_col=0, nrows=None)
    iris = df.index.values

    embbeding_client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")

    docs = np.array(df.values)

    render = DLSyntaxObjectRenderer()
    model = TDL(knowledge_base=kb, use_nominals=True, max_runtime=10)

    df2 = pd.read_csv("second_benchmark.csv", index_col=0, nrows=None)
    knn = df2.values.tolist()
    results = dict()
    iris_list = iris.tolist()
    for iri, query in evaluation_samples.items():
        qr = np.array(embbeding_client.embeddings.create(input=[query], model="tentris").data[0].embedding)
        cosine_similarities = find_cosine_similarities(qr, docs)
        K = 20
        top_scores_indexes = np.argpartition(cosine_similarities, -K)[-K:]
        bottom_scores_indexes = np.argpartition(cosine_similarities, K)[:K]

        top_iris = {iris[i] for i in top_scores_indexes}
        bottom_iris = {iris[i] for i in top_scores_indexes}

        typed_pos = set(map(OWLNamedIndividual, top_iris))
        typed_neg = set(map(OWLNamedIndividual, bottom_iris))

        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

        h = model.fit(learning_problem=lp).best_hypotheses()

        classified_instances = np.array([i.str for i in list(kb.individuals(h))])

        indexes_of_classified_instances = np.where(np.isin(iris, classified_instances))[0]
        irs = iris[indexes_of_classified_instances]
        classified_instances_docs = docs[indexes_of_classified_instances]
        textual_summary = get_LLM_textual_summary(iris[top_scores_indexes])
        textual_result_embedding = embbeding_client.embeddings.create(input=[textual_summary], model="tentris").data[
            0].embedding

        cs = find_cosine_similarities(textual_result_embedding, classified_instances_docs)
        vk = dict()
        for i in range(0, len(irs) - 1):
            vk[irs[i]] = cs[i]
        sorted_vk = dict(sorted(vk.items(), key=lambda item: item[1], reverse=True))

        cs = cs.tolist()
        cs.sort(reverse=True)
        best_score = -1
        for neighbor_index in knn[iris_list.index(iri)]:
            neigbor_iri = iris_list[neighbor_index]
            if neigbor_iri in sorted_vk:
                neighbor_score = sorted_vk[neigbor_iri]
                if neighbor_score > best_score:
                    best_score = neighbor_score
        if best_score != -1:
            placement = cs.index(best_score)
        else:
            placement = 45623
        results[iri] = placement
        print(f"==================== {iri}: {placement} ====================")

    with open(output, "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lp_size", help="Size of the learning problem where lp_size= |E+| + |E-| and |E+| = |E-|",
                        default=20)
    parser.add_argument('kg_path', help="Path of the knowledge graph to rank documents",
                        default="fashionpedia-third-generation.owl")
    args = parser.parse_args()
    run(args)



