from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectHasValue
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
from rdflib import Graph, URIRef, Literal, BNode, RDFS, OWL, Namespace, RDF
from rdflib.namespace import XSD

manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://fashionpedia-first-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)

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

# has_category = URIRef("http://example.org/hasCategory")

# g.add((has_category, RDF.type, OWL.DatatypeProperty))
# g.add((has_category, RDFS.domain, class_image))
# g.add((has_category, RDFS.range, XSD.string))

# has_category_supercat = URIRef("http://example.org/hasCatSuperCategory")

# g.add((has_category_supercat, RDF.type, OWL.DatatypeProperty))
# g.add((has_category_supercat, RDFS.domain, class_image))
# g.add((has_category_supercat, RDFS.range, XSD.string))

# has_attribute_supercat = URIRef("http://example.org/hasAttrSuperCategory")

# g.add((has_attribute_supercat, RDF.type, OWL.DatatypeProperty))
# g.add((has_attribute_supercat, RDFS.domain, class_image))
# g.add((has_attribute_supercat, RDFS.range, XSD.string))

# has_attribute = URIRef("http://example.org/hasAttribute")

# g.add((has_attribute, RDF.type, OWL.DatatypeProperty))
# g.add((has_attribute, RDFS.domain, class_image))
# g.add((has_attribute, RDFS.range, XSD.string))

has_description = URIRef("http://example.org/hasDescription")

g.add((has_description, RDF.type, OWL.DatatypeProperty))
g.add((has_description, RDFS.domain, class_image))
g.add((has_description, RDFS.range, XSD.string))

# has_original_url = URIRef("http://example.org/hasOriginalUrl")
# g.add((has_original_url, RDF.type, OWL.DatatypeProperty))
# g.add((has_original_url, RDFS.domain, class_image))
# g.add((has_original_url, RDFS.range, XSD.string))

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
    # g.add((ind, has_original_url, Literal(list(reasoner.data_property_values(image, dprop6))[0], datatype=XSD.string)))
    g.add((ind, has_width, Literal(int(list(reasoner.data_property_values(image, dprop2))[0].get_literal()), datatype=XSD.integer)))
    g.add((ind, has_height, Literal(int(list(reasoner.data_property_values(image, dprop3))[0].get_literal()), datatype=XSD.integer)))
    annotations = reasoner.instances(OWLObjectHasValue(oprop, image))
    for annotation in annotations:
        cat_ind = list(reasoner.object_property_values(annotation, oprop2))[0]
        cat_name = list(reasoner.data_property_values(cat_ind, dprop4))[0].get_literal()
        cat_supercat = list(reasoner.data_property_values(cat_ind, dprop5))[0].get_literal()
        # g.add((ind, has_category, Literal(cat_name, datatype=XSD.string)))
        # g.add((ind, has_category_supercat, Literal(cat_supercat, datatype=XSD.string)))
        description = f"Contains {cat_name} from supercategoty '{cat_supercat}'"
        attrs = list(reasoner.object_property_values(annotation, oprop3))
        if len(attrs) > 0:
            description += " with the following attributes: "
            for attr in attrs:
                attr_name = list(reasoner.data_property_values(attr, dprop4))[0].get_literal()
                attr_supercat = list(reasoner.data_property_values(attr, dprop5))[0].get_literal()
                # g.add((ind, has_attribute, Literal(cat_name, datatype=XSD.string)))
                # g.add((ind, has_attribute_supercat, Literal(cat_supercat, datatype=XSD.string)))
                description += f"{attr_name} of supercategory '{attr_supercat}', "
        g.add((ind, has_description, Literal(description[:-1], datatype=XSD.string)))

g.serialize(destination="fashionpedia-second-generation.owl", format="xml")