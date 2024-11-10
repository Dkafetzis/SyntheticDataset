import os
import pandas as pd
from datasets.utils.py_utils import glob_pattern_to_regex

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
import json
from continuous_eval.generators import SimpleDatasetGenerator
from continuous_eval.llm_factory import LLMFactory

os.environ["OPENAI_API_KEY"] = "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"
pd.set_option("display.max_colwidth", None)

loader = DirectoryLoader("/home/dkafetzis/Documents/langchain4j/docs/docs", glob="*.md")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
documents = loader.load_and_split(text_splitter)


# Load chunks into a vector database
vectorstore = Milvus(
    embedding_function=OpenAIEmbeddings(),
    connection_args={
        "uri": "milvus_vectorstore.db",
    },
    auto_id=True,
    drop_old=True,
)

vectorstore.add_documents(documents)

print("Done vectorizing")


generator = SimpleDatasetGenerator(
    vector_store_index = vectorstore,
    generator_llm = LLMFactory("gpt-4o"),
)

synthetic_dataset = generator.generate(
    embedding_vector_size = 1536,
    num_questions = 20,
)

with open('synthetic_dataset.json', 'w') as outfile:
    json.dump(synthetic_dataset, outfile, indent=4)
print(synthetic_dataset)
print("Done")


