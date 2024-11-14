import json
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from owlapy.class_expression import OWLObjectHasValue
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner, CELOE
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_property import OWLObjectProperty, OWLDataProperty

kb = KnowledgeBase(path="../fashionpedia-first-generation.owl")


def add_namespace(ind):
    return "http://example.org/" + ind


pos = {'image_21007': "Contains dress from supercategory 'wholebody' with the following attributes: a-line of supercategory 'silhouette', mini (length) of supercategory 'length'", 'image_45433': "Contains dress from supercategory 'wholebody' with the following attributes: normal waist of supercategory 'waistline', gown of supercategory 'nickname', trumpet of supercategory 'silhouette', asymmetrical of supercategory 'silhouette', lining of supercategory 'textile finishing, manufacturing techniques', maxi (length) of supercategory 'length', plastic of supercategory 'non-textile material type', zip-up of supercategory 'opening type'", 'image_19012': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_42863': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_8113': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_26892': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_32017': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_42632': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'"}

neg = {'image_34250': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'", 'image_20344': "Contains neckline from supercategory 'garment parts' with the following attributes: straight across (neck) of supercategory 'neckline type'"}

with open('../image-filename-mappings.json', 'r') as file:
    filename_of = json.load(file)
oprop = OWLObjectProperty(IRI.create("http://example.org/hasImage"))
oprop2 = OWLObjectProperty(IRI.create("http://example.org/hasCategory"))
oprop3 = OWLObjectProperty(IRI.create("http://example.org/hasAttribute"))
dprop4 = OWLDataProperty(IRI.create("http://example.org/hasName"))
dprop5 = OWLDataProperty(IRI.create("http://example.org/hasSupercategory"))


typed_pos = set()
typed_neg = set()

for i in set(pos.keys()).union(set(neg.keys())):
    ind = OWLNamedIndividual(add_namespace(i))
    annotations = kb.individuals(OWLObjectHasValue(oprop, ind))
    final_description = ""
    for annotation in annotations:
        cat_ind = list(kb.get_object_property_values(annotation, oprop2))[0]
        cat_name = list(kb.get_data_property_values(cat_ind, dprop4))[0].get_literal()
        cat_supercat = list(kb.get_data_property_values(cat_ind, dprop5))[0].get_literal()
        description = f"Contains {cat_name} from supercategory '{cat_supercat}'"
        attrs = list(kb.get_object_property_values(annotation, oprop3))
        if len(attrs) > 0:
            description += " with the following attributes: "
            for attr in attrs:
                attr_name = list(kb.get_data_property_values(attr, dprop4))[0].get_literal()
                attr_supercat = list(kb.get_data_property_values(attr, dprop5))[0].get_literal()
                description += f"{attr_name} of supercategory '{attr_supercat}', "
            final_description = description[:-2]
        else:
            final_description = description

        if i in pos.keys() and final_description in pos[i]:
            typed_pos.add(annotation)
        elif i in neg.keys() and final_description in neg[i]:
            typed_neg.add(annotation)

print(typed_pos)
print(typed_neg)
lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

model = CELOE(knowledge_base=kb, max_runtime=120)
model.fit(lp, verbose=False)

hypotheses = model.best_hypotheses(n=1)
print(hypotheses)
for ind in list(kb.individuals(hypotheses))[:10]:
    print(ind)
    filename = filename_of[ind.iri.reminder]
    img = np.asarray(Image.open(f'../images/{filename}'))
    plt.imshow(img)
    plt.show()
try:
    print(":gdf")
except Exception as e:
    print(e)
    with open("questions.json", "w") as outfile:
        json.dump({"dsad": "dasd"}, outfile)