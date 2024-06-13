import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader

class ConversationalChainWrapper:
    def __init__(self, repo_id, token, json_file_path, collection_name="resumes"):
        self.repo_id = repo_id
        self.token = token
        self.json_file_path = json_file_path
        self.collection_name = collection_name

        # Initialize the LLM
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id, max_length=200, temperature=0.7, token=self.token
        )

        # Load documents from the JSON file
        loader = JSONLoader(
            file_path=self.json_file_path,
            jq_schema='.content[]',
            text_content=False
        )
        data = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        # Create the embedding function
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Create a Qdrant collection
        self.qdrant_collection = Qdrant.from_documents(
            docs,
            embedding_function,
            location=":memory:", # Local mode with in-memory storage only
            collection_name=self.collection_name,
        )

        # Construct a retriever on top of the vector store
        self.qdrant_retriever = self.qdrant_collection.as_retriever()

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        # Create a custom prompt template
        custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original English.
                            Chat History:
                            {chat_history}
                            Follow-Up Input: {question}
                            Standalone question:"""
        self.CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

        # Create the conversational retrieval chain
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.qdrant_retriever,
            memory=self.memory,
            condense_question_prompt=self.CUSTOM_QUESTION_PROMPT
        )

    def invoke(self, question):
        return self.conversational_chain.invoke({"question": question})

# Usage
if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    repo_id = "google/gemma-1.1-2b-it"
    json_file_path = "./jsons/2016-01-01_0000_US_CNN_Erin_Burnett_OutFront.json"

    conversational_chain = ConversationalChainWrapper(repo_id, os.environ["HUGGINGFACEHUB_API_TOKEN"], json_file_path)
    conversational_chain2 = ConversationalChainWrapper(repo_id, os.environ["HUGGINGFACEHUB_API_TOKEN"], json_file_path)

    response = conversational_chain.invoke("What is also called conflict nuts?")
    print(response)
    response = conversational_chain2.invoke("Tell me about kim davis")
    print(response)