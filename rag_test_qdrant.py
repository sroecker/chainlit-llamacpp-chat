import os
import sys
import logging

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from qdrant_client import QdrantClient
# Qdrant local
client = QdrantClient(
    url="http://localhost:6334",
    prefer_grpc=True,
)

# embeddings
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3", request_timeout=120.0)


Settings.llm = llm
Settings.embed_model = ollama_embedding

vector_store = QdrantVectorStore(client=client, collection_name="llamacpp", prefer_grpc=True)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

queries = [
    "How to add a new model arch?",
    "Why is the first input token repeated?",
    "How to fix the Llama3 tokenizer?"
]

#retriever = index.as_retriever()
retriever = VectorIndexRetriever(
    index=index,
    similarity_topk=5,
)

response = retriever.retrieve(queries[0])
for node in response:
    print("node", node.score)
    print("node", node.text)
    print("node", node.metadata)
    print("#####\n\n")

"""
STREAMING = True
query_engine = index.as_query_engine(
    streaming=STREAMING,
    verbose=True,
)


for query_text in queries:
    response = query_engine.query(query_text)
    print(response)
"""
