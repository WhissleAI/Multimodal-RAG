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

repo_id = "google/gemma-1.1-2b-it"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=100, temperature=0.1, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)


from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path="jsons/output.json",
    jq_schema='.messages[].content',
    # text_content=False
)
data = loader.load()
pprint(data)
# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

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
            retriever=db.as_retriever(),
            memory = memory,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT
        )

# print(conversational_chain.invoke({"question":"who is gojo?"}))

# print(conversational_chain.invoke({"question":"what is his power?"}))