import base64

import aiohttp
import asyncio
import time
from openai import OpenAI
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
from rdflib import Graph, URIRef, Literal, BNode, RDFS, OWL, Namespace, RDF
from rdflib.namespace import XSD
from owlapy.owl_individual import OWLNamedIndividual

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-third-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
dprop1 = OWLDataProperty(IRI.create("http://example.org/hasFileName"))
dprop2 = OWLDataProperty(IRI.create("http://example.org/hasDescription"))
dprop3 = OWLDataProperty(IRI.create("http://example.org/hasLLMDescription"))
image_iri_as_str = "http://example.org/image_25521"
image_ind = OWLNamedIndividual(image_iri_as_str)

image_filename = "images/" + str(list(reasoner.data_property_values(image_ind, dprop1))[0].get_literal())
llm_description = str(list(reasoner.data_property_values(image_ind, dprop3))[0].get_literal())
base64_image = encode_image(image_filename)
all_descriptions = ""

# "Consider you are a user that is looking for clothes and other apparels in an online recommandation system."
#                             "Formulate a query of a prompt-like structure that the user would use in such a way that the attached image would be returned. To generate the query you can take in consideration the following auxiliary information about the image:"
#                             f"{all_descriptions}"
#                             f"{llm_description}"

# "Only write the query which should be a question and always end with a questionmark."
for d in list(reasoner.data_property_values(image_ind,dprop2)):
    all_descriptions = all_descriptions + d.get_literal() + "\n"

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")
print(client.chat.completions.create(
    model="tentris",
    messages=[
        {
        "role": "user",
        "content":
            [
                {
                    "type": "text",
                    "text": "Consider you are a user that is looking for clothes/apparels in an online recommandation system."
                            "Formulate a query of a prompt-like structure that the you would ask in such a way that the attached image would be recommended to you. To generate the query you can take in consideration the following auxiliary information about the image:"
                            f"{all_descriptions}"
                            f"{llm_description}" 
                            "Only write the query which should be a question and always end with a questionmark."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.1,
    seed=1
).choices[0].message.content)
