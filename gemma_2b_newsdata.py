import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from pprint import pprint
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

repo_id = "google/gemma-1.1-2b-it"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=200, temperature=0.7, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)


from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path="/content/drive/MyDrive/GSoC_RAG/output.json",
    jq_schema='.content',
    text_content=False
)
data = loader.load()


# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from bs4 import BeautifulSoup as Soup
# url = "https://transcripts.cnn.com/date/2024-04-13"

# loader = RecursiveUrlLoader(
#     url=url, max_depth=5, extractor=lambda x: Soup(x, "html.parser").text
# )
# data = loader.load()



pprint(data)
# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# create the open-source embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")# all-MiniLM-L6-v2

# load it into Chroma
# db = Chroma.from_documents(docs, embedding_function)
# create a qdrant collection - a vector based index of all resumes
qdrant_collection = Qdrant.from_documents(
    docs,
    embedding_function,
    location=":memory:", # Local mode with in-memory storage only
    collection_name="resumes",
)

# construct a retriever on top of the vector store
qdrant_retriever = qdrant_collection.as_retriever()

memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original English.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

conversational_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            chain_type="stuff",
            retriever=qdrant_retriever,
            memory = memory,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT
        )

# pprint(conversational_chain.invoke({"question": "donald trump has proved what"}))
# pprint(conversational_chain.invoke({"question": "what happened in 2015"}))
# pprint(conversational_chain.invoke({"question": "how was 2015"}))
# pprint(conversational_chain.invoke({"question": "2015 had what challenges"}))
# pprint(conversational_chain.invoke({"question": "What are people talking about trump?"}))
pprint(conversational_chain.invoke({"question": "you designed a car to cheat. it is really"}))