import base64
import time
from openai import OpenAI
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
from rdflib import Graph, URIRef, Literal, RDFS, OWL, RDF
from rdflib.namespace import XSD

api_key = "token-tentris-upb"
api_base = "http://tentris-ml.cs.upb.de:8501/v1"
client = OpenAI(api_key=api_key, base_url=api_base)


manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-second-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
g = Graph()
g.parse("fashionpedia-second-generation.owl")

has_llm_description = URIRef("http://example.org/hasLLMDescription")

class_image = URIRef("http://example.org/Image")
g.add((has_llm_description, RDF.type, OWL.DatatypeProperty))
g.add((has_llm_description, RDFS.domain, class_image))
g.add((has_llm_description, RDFS.range, XSD.string))

def get_message(base64_image):
    return client.chat.completions.create(
    model="tentris",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are a fashion expert."
                        "Your task is to give a short description of the apparel provided in the attached image."
                        "You should focus only on the apparel presented to you. Don't describe the background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }],
    temperature=0.1,
    seed=1
).choices[0].message.content


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def start_generation():
    tasks = []
    count = 0
    for i in ontology.individuals_in_signature():
        try:
            image_ind = URIRef(i.str)
            dprop1 = OWLDataProperty(IRI.create("http://example.org/hasFileName"))
            image = "images/" + str(list(reasoner.data_property_values(i, dprop1))[0].get_literal())
            base64_image = encode_image(image)
            llm_description = get_message(base64_image)
            g.add((image_ind, has_llm_description, Literal(llm_description, datatype=XSD.string)))
            count += 1
            print(f"{count:,}/45,623")
        except Exception as e:
            print(e)
            g.serialize(destination="fashionpedia-third-generation.owl", format="xml")
            exit(0)

    g.serialize(destination="fashionpedia-third-generation.owl", format="xml")

# Run the async loop
if __name__ == "__main__":
    start_time = time.time()
    start_generation()
    print(time.time() - start_time)
