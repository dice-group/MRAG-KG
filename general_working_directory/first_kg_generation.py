import json

from rdflib import Graph, URIRef, Literal, BNode, RDFS, OWL, Namespace, RDF
from rdflib.collection import Collection
from rdflib.namespace import XSD

with open('../instances_attributes_train2020.json') as json_file:
    data = json.load(json_file)

g = Graph()

ex = Namespace("http://example.org/")
g.bind("ex", ex)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)
g.bind("owl", OWL)

class_category = URIRef("http://example.org/Category")
class_attribute = URIRef("http://example.org/Attribute")
class_image = URIRef("http://example.org/Image")
class_annotation = URIRef("http://example.org/Annotation")
class_license = URIRef("http://example.org/License")

g.add((class_category, RDF.type, OWL.Class))
g.add((class_attribute, RDF.type, OWL.Class))
g.add((class_image, RDF.type, OWL.Class))
g.add((class_annotation, RDF.type, OWL.Class))
g.add((class_license, RDF.type, OWL.Class))

# Object properties

# --- Has Image ---
has_image = URIRef("http://example.org/hasImage")

g.add((has_image, RDF.type, OWL.ObjectProperty))
g.add((has_image, RDFS.domain, class_annotation))
g.add((has_image, RDFS.range, class_image))

# --- Has Category ---
has_category = URIRef("http://example.org/hasCategory")

g.add((has_category, RDF.type, OWL.ObjectProperty))
g.add((has_category, RDFS.domain, class_annotation))
g.add((has_category, RDFS.range, class_category))

# --- Has Attribute ---
has_attribute = URIRef("http://example.org/hasAttribute")

g.add((has_attribute, RDF.type, OWL.ObjectProperty))
g.add((has_attribute, RDFS.domain, class_annotation))
g.add((has_attribute, RDFS.range, class_attribute))

# --- Has License ---
has_license = URIRef("http://example.org/hasLicense")

g.add((has_attribute, RDF.type, OWL.ObjectProperty))
g.add((has_attribute, RDFS.domain, class_image))
g.add((has_attribute, RDFS.range, class_license))


# Data properties
# --- Has File Name ---
has_filename = URIRef("http://example.org/hasFileName")
g.add((has_filename, RDF.type, OWL.DatatypeProperty))
g.add((has_filename, RDFS.domain, class_image))
g.add((has_filename, RDFS.range, XSD.string))

union_cat_attr = BNode()
g.add((union_cat_attr, RDF.type, OWL.Class))
class_list = BNode()
g.add((union_cat_attr, OWL.unionOf, class_list))
Collection(g, class_list, [class_category, class_attribute, class_license])

# --- Has Name ---
has_name = URIRef("http://example.org/hasName")
g.add((has_name, RDF.type, OWL.DatatypeProperty))
g.add((has_name, RDFS.domain, union_cat_attr))
g.add((has_name, RDFS.range, XSD.string))

# --- Has Supercategory ---
has_supercategory = URIRef("http://example.org/hasSupercategory")
g.add((has_supercategory, RDF.type, OWL.DatatypeProperty))
g.add((has_supercategory, RDFS.domain, union_cat_attr))
g.add((has_supercategory, RDFS.range, XSD.string))

# --- Has Level ---
has_level = URIRef("http://example.org/hasLevel")
g.add((has_level, RDF.type, OWL.DatatypeProperty))
g.add((has_level, RDFS.domain, union_cat_attr))
g.add((has_level, RDFS.range, XSD.integer))

# --- Has Taxonomy ---
has_taxonomy = URIRef("http://example.org/hasTaxonomy")
g.add((has_taxonomy, RDF.type, OWL.DatatypeProperty))
g.add((has_taxonomy, RDFS.domain, union_cat_attr))
g.add((has_taxonomy, RDFS.range, XSD.string))

has_width = URIRef("http://example.org/hasWidth")
g.add((has_width, RDF.type, OWL.DatatypeProperty))
g.add((has_width, RDFS.domain, class_image))
g.add((has_width, RDFS.range, XSD.integer))

has_height = URIRef("http://example.org/hasHeight")
g.add((has_height, RDF.type, OWL.DatatypeProperty))
g.add((has_height, RDFS.domain, class_image))
g.add((has_height, RDFS.range, XSD.integer))

time_captured = URIRef("http://example.org/hasTimeCaptured")
g.add((time_captured, RDF.type, OWL.DatatypeProperty))
g.add((time_captured, RDFS.domain, class_image))
g.add((time_captured, RDFS.range, XSD.string))

has_original_url = URIRef("http://example.org/hasOriginalUrl")
g.add((has_original_url, RDF.type, OWL.DatatypeProperty))
g.add((has_original_url, RDFS.domain, class_image))
g.add((has_original_url, RDFS.range, XSD.string))

is_static = URIRef("http://example.org/isStatic")
g.add((is_static, RDF.type, OWL.DatatypeProperty))
g.add((is_static, RDFS.domain, class_image))
g.add((is_static, RDFS.range, XSD.integer))

has_kaggle_id = URIRef("http://example.org/hasKaggleId")
g.add((has_kaggle_id, RDF.type, OWL.DatatypeProperty))
g.add((has_kaggle_id, RDFS.domain, class_image))
g.add((has_kaggle_id, RDFS.range, XSD.string))

has_url = URIRef("http://example.org/hasUrl")
g.add((has_url, RDF.type, OWL.DatatypeProperty))
g.add((has_url, RDFS.domain, class_license))
g.add((has_url, RDFS.range, XSD.string))

has_area = URIRef("http://example.org/hasArea")
g.add((has_area, RDF.type, OWL.DatatypeProperty))
g.add((has_area, RDFS.domain, class_annotation))
g.add((has_area, RDFS.range, XSD.integer))

is_crowd = URIRef("http://example.org/isCrowd")
g.add((is_crowd, RDF.type, OWL.DatatypeProperty))
g.add((is_crowd, RDFS.domain, class_annotation))
g.add((is_crowd, RDFS.range, XSD.integer))

for license in data["licenses"]:
    ind = URIRef(f"http://example.org/license_{license['id']}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_license))
    g.add((ind, has_name, Literal(license["name"], datatype=XSD.string)))
    g.add((ind, has_url, Literal(license["url"], datatype=XSD.string)))


for image in data["images"]:
    ind = URIRef(f"http://example.org/image_{image['id']}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_image))
    g.add((ind, has_license, URIRef(f"http://example.org/license_{image['license']}")))
    g.add((ind, has_filename, Literal(image["file_name"], datatype=XSD.string)))
    if image["width"]:
        g.add((ind, has_width, Literal(image["width"], datatype=XSD.integer)))
    if image["height"]:
        g.add((ind, has_height, Literal(image["height"], datatype=XSD.integer)))
    if image["time_captured"]:
        g.add((ind, time_captured, Literal(image["time_captured"], datatype=XSD.string)))
    if image["original_url"]:
        g.add((ind, has_original_url, Literal(image["original_url"], datatype=XSD.string)))
    g.add((ind, is_static, Literal(image["isstatic"], datatype=XSD.string)))
    g.add((ind, has_kaggle_id, Literal(image["kaggle_id"], datatype=XSD.string)))


for category in data["categories"]:
    ind = URIRef(f"http://example.org/category_{category['id']}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_category))
    g.add((ind, has_name, Literal(category["name"], datatype=XSD.string)))
    g.add((ind, has_supercategory, Literal(category["supercategory"], datatype=XSD.string)))
    g.add((ind, has_level, Literal(category["level"], datatype=XSD.integer)))
    g.add((ind, has_taxonomy, Literal(category["taxonomy_id"], datatype=XSD.string)))


for attribute in data["attributes"]:
    ind = URIRef(f"http://example.org/attribute_{attribute['id']}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_attribute))
    g.add((ind, has_name, Literal(attribute["name"], datatype=XSD.string)))
    g.add((ind, has_supercategory, Literal(attribute["supercategory"], datatype=XSD.string)))
    g.add((ind, has_level, Literal(attribute["level"], datatype=XSD.integer)))
    g.add((ind, has_taxonomy, Literal(attribute["taxonomy_id"], datatype=XSD.string)))


for annotation in data["annotations"]:
    ind = URIRef(f"http://example.org/annotation_{annotation['id']}")
    g.add((ind, RDF.type, OWL.NamedIndividual))
    g.add((ind, RDF.type, class_annotation))
    g.add((ind, has_image, URIRef(f"http://example.org/image_{annotation['image_id']}")))
    g.add((ind, has_category, URIRef(f"http://example.org/category_{annotation['category_id']}")))
    for attribute_id in annotation["attribute_ids"]:
        g.add((ind, has_attribute, URIRef(f"http://example.org/attribute_{attribute_id}")))
    if annotation["area"]:
        g.add((ind, has_area, Literal(annotation["area"], datatype=XSD.integer)))
    if annotation["iscrowd"]:
        g.add((ind, is_crowd, annotation["iscrowd"]))


g.serialize(destination="../fashionpedia-first-generation.owl", format="xml")