import json
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner, CELOE
from ontolearn.learning_problem import PosNegLPStandard

kb = KnowledgeBase(path="../fashionpedia-second-generation.owl")


def add_namespace(ind):
    return "http://example.org/" + ind


pos = {'image_16361','image_17484'}

neg = {'image_27498', 'image_13378', 'image_15301', 'image_14748', 'image_14509', 'image_24968', 'image_11384'}

with open('../image-filename-mappings.json', 'r') as file:
    filename_of = json.load(file)

typed_pos = set(map(OWLNamedIndividual, map(IRI.create, map(add_namespace, pos))))
typed_neg = set(map(OWLNamedIndividual, map(IRI.create, map(add_namespace, neg))))

lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

model = CELOE(knowledge_base=kb, max_runtime=120)
model.fit(lp, verbose=False)

hypotheses = model.best_hypotheses(n=1)
print(hypotheses)
for ind in list(kb.individuals(hypotheses))[:10]:
    print(ind)
    filename = filename_of[ind.iri.get_short_form()]
    img = np.asarray(Image.open(f'images/{filename}'))
    plt.imshow(img)
    plt.show()


# image_16361: -7641c2fb25c33cfa17350704ebb5d5c0.jpg
# image_15301: 46bf121c6a6ec0f552697fde21f941fb.jpg
# image_24968: 52888b80a35e3183b6e7f601496ae95d.jpg
# image_13378: 649575404695e7beee8545587b2cc904.jpg
# image_14748: 4f29f775d533dcbda07247010f3e8171.jpg
# image_27498: ac1c3d5a9c672b5b9e585901ada643a6.jpg
# image_14509: 7ef52ba6e7f3d10ab1d02b70f1394f8c.jpg
# image_17484: -b58cc4cd8b961fa2ec842b6feb779509.jpg
# image_11384: a1e19fedb33336ef1469b44bbdd06db6.jpg
