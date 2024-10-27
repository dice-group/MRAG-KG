import base64
from openai import OpenAI
from owlapy.owl_individual import OWLNamedIndividual


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image = "000b3a87508b0fa185fbd53ecbe2e4c6.jpg"
# Getting the base64 string
base64_image = encode_image(image)

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")
print(client.chat.completions.create(
    model="tentris",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are a fashion expert."
                        "Your task is to give a short description of the apparel provided in the attached image."
                        "You should focus only on the apparel presented to you. Don't describe the background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }],
    temperature=0.1,
    seed=1
).choices[0].message.content)
