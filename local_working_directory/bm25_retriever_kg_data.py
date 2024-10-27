import os

os.environ["OPENAI_API_KEY"] = "token-tentris-upb"
import rdflib
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, Document
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import Stemmer


Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

g = rdflib.Graph()
g.parse("fashionpedia-second-generation.owl", format="xml")

# Extract triplets
triplets = []
for subj, pred, obj in g:
    triplets.append((str(subj), str(pred), str(obj)))

# Index the data (convert triplets to text format)
text_data = ["\n".join(triplet) for triplet in triplets]

with open('fashionpedia-second-generation.txt', 'w') as file:
    file.writelines(text_data)
