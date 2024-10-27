import base64

import aiohttp
import asyncio
import time
from openai import OpenAI
from owlapy.iri import IRI
from owlapy.owl_ontology_manager import OntologyManager
from owlapy.owl_property import OWLDataProperty
from owlapy.owl_reasoner import OntologyReasoner, FastInstanceCheckerReasoner
from rdflib import Graph, URIRef, RDF, OWL, RDFS, XSD

api_key = "token-tentris-upb"
api_base = "http://tentris-ml.cs.upb.de:8501/v1"
client = OpenAI(api_key=api_key, base_url=api_base)


manager = OntologyManager()
ontology = manager.load_ontology(IRI.create("file://../fashionpedia-second-generation.owl"))
base_reasoner = OntologyReasoner(ontology)
reasoner = FastInstanceCheckerReasoner(base_reasoner=base_reasoner, ontology=ontology)
g = Graph()
g.parse("../fashionpedia-second-generation.owl")

has_llm_description = URIRef("http://example.org/hasLLMDescription")

class_image = URIRef("http://example.org/Image")
g.add((has_llm_description, RDF.type, OWL.DatatypeProperty))
g.add((has_llm_description, RDFS.domain, class_image))
g.add((has_llm_description, RDFS.range, XSD.string))


def tentris_ensemble_llm():
    completion = client.chat.completions.create(
      model="tentris",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Which LLM would you use to answer the user's question?"
                                          "We cannot effort to use largest model for every questions."},
            {"role": "user", "content": "What is the capital of Germany"}],
        extra_body={"guided_choice": ["Qwen2.5-0.5B-Instruct",
                                      "Qwen2.5-1.5B-Instruct"
                                      "Qwen2.5-3B-Instruct",
                                      "Qwen2.5-7B-Instruct",
                                      "Llama-3.1-3B-Instruct",
                                      "Llama-3.2-3B-Instruct"]})
    return completion.choices[0].message.content


headers = {
    'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}


# Asynchronous function to send a single request
async def send_async_command(payload, ind):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{api_base}/chat/completions', headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"Result from {ind.str}: {result['content']}")
            else:
                print(f"Error in query about {ind.str}: {response.status}")
                print(await response.text())


# Define a function to create the payload for each query
def create_payload(base64_image):
    return {
        "model": "tentris",
        "messages": [
            {"role": "user",
             "content": [{
                            "type": "text",
                            "text": "You are a fashion expert."
                                    "Your task is to give a short description of the apparel shown in the attached image."
                         },
                         {
                             "type": "image_url",
                             "image_url": {
                                 "url": f"data:image/jpeg;base64,{base64_image}"
                             }
                         }
                         ]
             }
        ], "temperature": 0.1
    }


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Asynchronous function to send 10 queries concurrently
async def send_multiple_queries():
    tasks = []
    count = 0
    for i in ontology.individuals_in_signature():
        dprop1 = OWLDataProperty(IRI.create("http://example.org/hasFileName"))
        image = "../images/" + str(list(reasoner.data_property_values(i, dprop1))[0].get_literal())
        base64_image = encode_image(image)
        payload = create_payload(base64_image)
        tasks.append(send_async_command(payload, i))
        count += 1
        if count > 3:
            break

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


# Run the async loop
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(send_multiple_queries())
    print(time.time() - start_time)
