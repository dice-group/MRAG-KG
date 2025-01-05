import base64
import json
import time
from openai import OpenAI
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner


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

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")

def get_message(base64, desc, llm_desc):
    return client.chat.completions.create(
        model="tentris",
        messages=[
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": "Consider you are a user that is looking for clothes/apparel in an online recommandation system."
                                    "Formulate a query of a prompt-like structure that you would ask in such a way that the attached image would be recommended to you. To generate the query, you can take into consideration the following auxiliary information about the image:"
                                    f"{desc}"
                                    f"{llm_desc}"
                                    "Only write the query, which should be a question, and always end with a question mark."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64}"
                            }
                        }
                    ]
            }
        ],
        temperature=0.1,
        seed=1
    ).choices[0].message.content


image_full_question_dict = dict()


def start_generation():
    count = 0
    for image_ind in ontology.individuals_in_signature():
        image_filename = "images/" + str(list(reasoner.data_property_values(image_ind, dprop1))[0].get_literal())
        base64_image = encode_image(image_filename)
        llm_description = str(list(reasoner.data_property_values(image_ind, dprop3))[0].get_literal())
        if len(llm_description) > 4000:
            llm_description = llm_description[:4000]
        all_descriptions = ""
        for d in list(reasoner.data_property_values(image_ind, dprop2)):
            all_descriptions = all_descriptions + d.get_literal() + "\n"
        question = get_message(base64_image, all_descriptions, llm_description)
        image_iri = image_ind.str
        image_full_question_dict[image_iri] = question
        count += 1
        print(f"{image_iri}: {count:,}/45,623")

    with open("questions.json", "w") as outfile:
        json.dump(image_full_question_dict, outfile)


if __name__ == "__main__":
    start_time = time.time()
    try:
        start_generation()
    except Exception as e:
        print(e)
        with open("questions_uncompleted.json", "w") as outfile:
            json.dump(image_full_question_dict, outfile)
    print(time.time() - start_time)