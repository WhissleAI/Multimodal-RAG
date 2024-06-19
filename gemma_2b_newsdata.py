import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from langchain.chains import LLMChain
import json
import time

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
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})

        # Create a Qdrant collection
        self.qdrant_collection = Qdrant.from_documents(
            docs,
            embedding_function,
            location=":memory:", # Local mode with in-memory storage only
            collection_name=self.collection_name,
        )

        # Construct a retriever on top of the vector store
        self.qdrant_retriever = self.qdrant_collection.as_retriever()

        # Create a custom prompt template
        custom_template = """
            "role": "user",
            "content": 
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query}"
            """
        self.CUSTOM_QUESTION_PROMPT = PromptTemplate(template=custom_template, input_variables=['context_str', 'query'])

        self.llm_chain = LLMChain(prompt=self.CUSTOM_QUESTION_PROMPT, llm=self.llm)


    def invoke(self, query):
        context = self.qdrant_retriever.invoke(query)
        context = [c.page_content for c in context]
        context_str = "\n\n".join(context)
        return context, self.llm_chain.run(query=query, context_str=context_str)

# Usage
if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    repo_id = "google/gemma-1.1-2b-it"
    json_file_path = "./jsons/2016-01-01_0000_US_CNN_Erin_Burnett_OutFront.json"

    conversational_chain = ConversationalChainWrapper(repo_id, os.environ["HUGGINGFACEHUB_API_TOKEN"], json_file_path)

    with open('dataset/format_QAdata.json') as f:
        json_data = json.load(f)

    questions = json_data['question']
    for question in questions:
        context, response = conversational_chain.invoke(question)
        json_data['contexts'].append(context)
        json_data['answer'].append(response)
        # break

    name = "gemma-2b"
    time = time.strftime("%Y-%m-%d-%H-%M-%S")
    with open(f"out/{name}-{time}.json", 'a+') as f:
        json.dump(json_data, f)



    
