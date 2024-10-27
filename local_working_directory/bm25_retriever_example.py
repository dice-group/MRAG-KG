import os

os.environ["OPENAI_API_KEY"] = "token-tentris-upb"

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response.notebook_utils import display_source_node
import Stemmer

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

# load documents
documents = SimpleDirectoryReader("./data/text-from-kg").load_data()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
# We can pass in the index, docstore, or list of nodes to create the retriever
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

retrieved_nodes = bm25_retriever.retrieve(
    "What are some clothing options that include a black leather jacket with a zip-up front, above-the-hip length, asymmetrical silhouette, and biker style?"
)
for node in retrieved_nodes:
    print(node.text)


