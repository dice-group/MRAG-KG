import json
import time
from openai import OpenAI


def get_message(query):
    client = OpenAI(base_url="http://tentris-ml.cs.upb.de:8501/v1", api_key="token-tentris-upb")
    return client.chat.completions.create(
        model="tentris",
        messages=[
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": "Separate each part of the following question into self-contained questions. Separate them in new line without using ordering numbers. Here is the question:"
                                    f"{query}"
                        },
                    ]
            }
        ],
        temperature=0.1,
        seed=1
    ).choices[0].message.content


image_full_question_fragments_dict = dict()


def start_generation():
    count = 0
    with open('questions.json') as json_file:
        data = json.load(json_file)

    images_iris = data.keys()

    for image in images_iris:
        separated_questions = get_message(data[image])
        image_full_question_fragments_dict[image] = separated_questions
        count += 1
        print(f"{image}: {count:,}/45,623")
        if count > 10:
            break

    with open("first_benchmark.json", "w") as outfile:
        json.dump(image_full_question_fragments_dict, outfile)


if __name__ == "__main__":
    start_time = time.time()
    try:
        start_generation()
    except Exception as e:
        print(e)
        with open("fragmented_questions_uncompleted.json", "w") as outfile:
            json.dump(image_full_question_fragments_dict, outfile)
    print(time.time() - start_time)