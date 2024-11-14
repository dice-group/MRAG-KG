import csv
from openai import OpenAI
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-third-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
dprop2 = OWLDataProperty(IRI.create("http://example.org/hasDescription"))
dprop3 = OWLDataProperty(IRI.create("http://example.org/hasLLMDescription"))

with open('embeddings.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["IRI"] + [f"{i}" for i in range(4095)])
    client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8502/v1", api_key="token-tentris-upb")
    count = 0
    for image_ind in ontology.individuals_in_signature():
        llm_description = str(list(reasoner.data_property_values(image_ind, dprop3))[0].get_literal())
        if len(llm_description) > 4000:
            llm_description = llm_description[:4000]
        all_descriptions = ""
        for d in list(reasoner.data_property_values(image_ind, dprop2)):
            all_descriptions = all_descriptions + d.get_literal() + "\n"
        image_iri = image_ind.str
        responses = client.embeddings.create(input=[all_descriptions + "\n " + llm_description], model="tentris")
        writer.writerow([image_iri] + responses.data[0].embedding)
        count += 1
        print(f"{image_iri}: {count:,}/45,623")
