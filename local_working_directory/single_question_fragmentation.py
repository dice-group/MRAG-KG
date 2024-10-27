from openai import OpenAI

client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")
print(client.chat.completions.create(
    model="tentris",
    messages=[
        {
        "role": "user",
        "content":
            [
                {
                    "type": "text",
                    "text": "Can you separate each part of the following question into self-contained questions:"
                             "What are some clothing options that include a black leather jacket, a light blue denim shirt with a white collar, and black pants, suitable for a casual yet edgy style?"
                },
            ]
        }
    ],
    temperature=0.1,
    seed=1
).choices[0].message.content)
