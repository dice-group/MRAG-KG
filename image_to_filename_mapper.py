from owlapy.class_expression import OWLDataHasValue
from owlapy.iri import IRI
from owlapy.owl_literal import OWLLiteral
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
import json
import os

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-second-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
has_description = OWLDataProperty(IRI.create("http://example.org/hasDescription"))

mappings = {}
for filename in os.listdir('images'):
    iri = IRI.create("http://example.org/hasFileName")
    dp = OWLDataProperty(iri)
    dhvf = OWLDataHasValue(dp, OWLLiteral(filename))
    image = list(reasoner.instances(dhvf))[0]
    mappings[image.iri.get_short_form()] = filename

with open("image-filename-mappings", 'w') as f:
    json.dump(mappings, f)
