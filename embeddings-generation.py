from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
import torch
from transformers import AutoModel
from numpy.linalg import norm
import json

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-second-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
has_description = OWLDataProperty(IRI.create("http://example.org/hasDescription"))

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)

embeddings_final = {}
for image in ontology.individuals_in_signature():
    descriptions = list(reasoner.data_property_values(image, has_description))
    desc_counter = 1
    for description in descriptions:
        embeddings = model.encode(description.get_literal())
        embeddings_final[image.str.split("/")[-1] + f"_{desc_counter}"] = embeddings.tolist()
        desc_counter += 1

with open("fashionpedia-embeddings", 'w') as f:
    json.dump(embeddings_final, f)
