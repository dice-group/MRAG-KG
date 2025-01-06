from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectHasValue
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_reasoner import StructuralReasoner
from rdflib import Graph, URIRef, Literal, BNode, RDFS, OWL, Namespace, RDF
from rdflib.namespace import XSD


manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://../fashionpedia-first-generation.owl"))
reasoner = StructuralReasoner( ontology=ontology)

images = reasoner.instances(OWLClass(IRI.create("http://example.org/Image")))
g = Graph()
ex = Namespace("http://example.org/")
g.bind("ex", ex)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)

class_image = URIRef("http://example.org/Image")
has_width = URIRef("http://example.org/hasWidth")
g.add((has_width, RDF.type, OWL.DatatypeProperty))
g.add((has_width, RDFS.domain, class_image))
g.add((has_width, RDFS.range, XSD.integer))

has_height = URIRef("http://example.org/hasHeight")
g.add((has_height, RDF.type, OWL.DatatypeProperty))
g.add((has_height, RDFS.domain, class_image))
g.add((has_height, RDFS.range, XSD.integer))

has_filename = URIRef("http://example.org/hasFileName")
g.add((has_filename, RDF.type, OWL.DatatypeProperty))
g.add((has_filename, RDFS.domain, class_image))
g.add((has_filename, RDFS.range, XSD.string))

has_description = URIRef("http://example.org/hasDescription")

g.add((has_description, RDF.type, OWL.DatatypeProperty))
g.add((has_description, RDFS.domain, class_image))
g.add((has_description, RDFS.range, XSD.string))

oprop = OWLObjectProperty(IRI.create("http://example.org/hasImage"))
oprop2 = OWLObjectProperty(IRI.create("http://example.org/hasCategory"))
oprop3 = OWLObjectProperty(IRI.create("http://example.org/hasAttribute"))

dprop1 = OWLDataProperty(IRI.create("http://example.org/hasFileName"))
dprop2 = OWLDataProperty(IRI.create("http://example.org/hasWidth"))
dprop3 = OWLDataProperty(IRI.create("http://example.org/hasHeight"))
dprop4 = OWLDataProperty(IRI.create("http://example.org/hasName"))
dprop5 = OWLDataProperty(IRI.create("http://example.org/hasSupercategory"))
dprop6 = OWLDataProperty(IRI.create("http://example.org/hasOriginalUrl"))


for image in images:
    ind = URIRef(image.str)
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_image))
    g.add((ind, has_filename, Literal(list(reasoner.data_property_values(image,dprop1))[0].get_literal(), datatype=XSD.string)))
    g.add((ind, has_width, Literal(int(list(reasoner.data_property_values(image, dprop2))[0].get_literal()), datatype=XSD.integer)))
    g.add((ind, has_height, Literal(int(list(reasoner.data_property_values(image, dprop3))[0].get_literal()), datatype=XSD.integer)))
    annotations = reasoner.instances(OWLObjectHasValue(oprop, image))
    for annotation in annotations:
        cat_ind = list(reasoner.object_property_values(annotation, oprop2))[0]
        cat_name = list(reasoner.data_property_values(cat_ind, dprop4))[0].get_literal()
        cat_supercat = list(reasoner.data_property_values(cat_ind, dprop5))[0].get_literal()
        description = f"Contains {cat_name} from supercategory '{cat_supercat}'"
        attrs = list(reasoner.object_property_values(annotation, oprop3))
        if len(attrs) > 0:
            description += " with the following attributes: "
            for attr in attrs:
                attr_name = list(reasoner.data_property_values(attr, dprop4))[0].get_literal()
                attr_supercat = list(reasoner.data_property_values(attr, dprop5))[0].get_literal()
                description += f"{attr_name} of supercategory '{attr_supercat}', "
            g.add((ind, has_description, Literal(description[:-2], datatype=XSD.string)))
        else:
            g.add((ind, has_description, Literal(description, datatype=XSD.string)))

g.serialize(destination="../fashionpedia-second-generation.owl", format="xml")