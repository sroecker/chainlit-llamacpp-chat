import os
import sys
import logging

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.readers.github import (
    GitHubRepositoryIssuesReader,
    GitHubIssuesClient,
)

github_client = GitHubIssuesClient(github_token=os.getenv("GITHUB_TOKEN"), verbose=True)

from llama_index.vector_stores.qdrant import QdrantVectorStore

# To connect to the same event-loop,
# allows async events to run on notebook
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from qdrant_client import QdrantClient
# Qdrant local
client = QdrantClient(
    #location=":memory:"
    # Async upsertion does not work
    # on 'memory' location and requires
    # Qdrant to be deployed somewhere.
    url="http://localhost:6334",
    prefer_grpc=True,
)

# TODO: You will need to set a Github token

reader = GitHubRepositoryIssuesReader(
    github_client=github_client,
    owner="ggerganov",
    repo="llama.cpp",
    verbose=True,
)

documents = reader.load_data(
    state=GitHubRepositoryIssuesReader.IssueState.ALL,
    #labelFilters=[("bug", GitHubRepositoryIssuesReader.FilterType.INCLUDE)],
)

# embeddings
from llama_index.embeddings.ollama import OllamaEmbedding

ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# not really needed except for the service context
from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3", request_timeout=120.0)


Settings.llm = llm
Settings.embed_model = ollama_embedding
#Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
#Settings.num_output = 512
#Settings.context_window = 3900

vector_store = QdrantVectorStore(    
    client=client, collection_name="llamacpp", 
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    # FIXME asyncio throws error
    #use_async=True,
    show_progress=True,
)
# FIXME add refresh example
