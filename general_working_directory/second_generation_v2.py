from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectHasValue
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner
from rdflib import Graph, URIRef, Literal, BNode, RDFS, OWL, Namespace, RDF
from rdflib.namespace import XSD
import json

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-first-generation.owl"))
reasoner = StructuralReasoner(ontology=ontology)

images = reasoner.instances(OWLClass(IRI.create("http://example.org/Image")))
g = Graph()
ex = Namespace("http://example.org/")
g.bind("ex", ex)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)

class_image = URIRef("http://example.org/Image")
class_category = URIRef("http://example.org/AnnotationCategory")
class_super_category = URIRef("http://example.org/AnnotationSupercategory")
class_attribute = URIRef("http://example.org/Attribute")
class_attribute_category = URIRef("http://example.org/AttributeCategory")

g.add((class_image, RDF.type, OWL.Class))
g.add((class_category, RDF.type, OWL.Class))
g.add((class_super_category, RDF.type, OWL.Class))
g.add((class_attribute, RDF.type, OWL.Class))
g.add((class_attribute_category, RDF.type, OWL.Class))

hasAnnotationCategory = URIRef("http://example.org/hasAnnotationCategory")
g.add((hasAnnotationCategory, RDF.type, OWL.ObjectProperty))
g.add((hasAnnotationCategory, RDFS.domain, class_image))
g.add((hasAnnotationCategory, RDFS.range, class_category))

hasAnnotationSupercategory = URIRef("http://example.org/hasAnnotationSupercategory")
g.add((hasAnnotationSupercategory, RDF.type, OWL.ObjectProperty))
g.add((hasAnnotationSupercategory, RDFS.domain, class_image))
g.add((hasAnnotationSupercategory, RDFS.range, class_super_category))

hasAttribute = URIRef("http://example.org/hasAttribute")
g.add((hasAttribute, RDF.type, OWL.ObjectProperty))
g.add((hasAttribute, RDFS.domain, class_image))
g.add((hasAttribute, RDFS.range, class_attribute))

hasAttributeCategory = URIRef("http://example.org/hasAttributeCategory")
g.add((hasAttributeCategory, RDF.type, OWL.ObjectProperty))
g.add((hasAttributeCategory, RDFS.domain, class_attribute))
g.add((hasAttributeCategory, RDFS.range, class_attribute_category))

oprop = OWLObjectProperty(IRI.create("http://example.org/hasImage"))
oprop2 = OWLObjectProperty(IRI.create("http://example.org/hasCategory"))
oprop3 = OWLObjectProperty(IRI.create("http://example.org/hasAttribute"))

dprop4 = OWLDataProperty(IRI.create("http://example.org/hasName"))
dprop5 = OWLDataProperty(IRI.create("http://example.org/hasSupercategory"))

with open('instances_attributes_train2020.json') as json_file:
    data = json.load(json_file)

for category in data["categories"]:
    cat_name = category["name"].replace(" ", "_")
    supercat_name = category["supercategory"].replace(" ", "_")
    if (URIRef(f"http://example.org/annotation_supercategory_{supercat_name}"), None, class_super_category) not in g:
        ind = URIRef(f"http://example.org/annotation_supercategory_{supercat_name}")
        g.add((ind, RDF.type, OWL.NamedIndividual))
        g.add((ind, RDF.type, class_super_category))
    ind = URIRef(f"http://example.org/annotation_category_{cat_name}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_category))
print("Annotations: Done!")

for attribute in data["attributes"]:
    attribute_name = attribute["name"].replace(" ", "_")
    attribute_cat_name = attribute["supercategory"].replace(" ", "_")
    if (URIRef(f"http://example.org/attribute_category_{attribute_cat_name}"), None, class_attribute_category) not in g:
        ind = URIRef(f"http://example.org/attribute_category_{attribute_cat_name}")
        g.add((ind, RDF.type, OWL.NamedIndividual))
        g.add((ind, RDF.type, class_attribute_category))
    ind = URIRef(f"http://example.org/attribute_{attribute_name}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_attribute))
print("Attributes: Done!")

for image in images:
    ind = URIRef(image.str)
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_image))
    annotations = reasoner.instances(OWLObjectHasValue(oprop, image))
    for annotation in annotations:
        cat_ind = list(reasoner.object_property_values(annotation, oprop2))[0]

        cat_name = list(reasoner.data_property_values(cat_ind, dprop4))[0].get_literal().replace(" ", "_")
        cat_supercat = list(reasoner.data_property_values(cat_ind, dprop5))[0].get_literal().replace(" ", "_")

        annotation_cat_ind = URIRef(f"http://example.org/annotation_category_{cat_name}")
        annotation_supercat_ind = URIRef(f"http://example.org/annotation_supercategory_{cat_supercat}")

        g.add((ind, hasAnnotationCategory, annotation_cat_ind))
        g.add((ind, hasAnnotationSupercategory, annotation_supercat_ind))

        attrs = list(reasoner.object_property_values(annotation, oprop3))
        if len(attrs) > 0:
            for attr in attrs:
                attr_name = list(reasoner.data_property_values(attr, dprop4))[0].get_literal().replace(" ", "_")
                attr_cat = list(reasoner.data_property_values(attr, dprop5))[0].get_literal().replace(" ", "_")

                attribute_ind = URIRef(f"http://example.org/attribute_{attr_name}")
                attribute_cat_ind = URIRef(f"http://example.org/attribute_category_{attr_cat}")

                g.add((ind, hasAttribute, attribute_ind))
                g.add((ind, hasAttributeCategory, attribute_cat_ind))

g.serialize(destination="fashionpedia-second-generation-v2.owl", format="xml")