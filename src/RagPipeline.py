import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils import log_execution

import os
from dotenv import load_dotenv
load_dotenv()



class RagPipeline:
    def __init__(self, config):
        self.config = config
        self.repo_id = config['llm']['model_id']
        self.token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.context_metadata_filename = config['context_loader']['file_path']
        self.collection_name = config['vectordb']['qdrant']['collection_name']

        self.load_context_metadata()
        self.create_vectordb()
        self.init_LLM()

        prompt_template = PromptTemplate(
            template=config['prompt']['template'],
            input_variables=config['prompt']['input_variables']
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)
   
    @log_execution
    def init_LLM(self):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.repo_id,
            task=self.config['llm']['task'],
            batch_size=self.config['llm']['batch_size'],
            pipeline_kwargs={
                "max_new_tokens": self.config['llm']['max_new_tokens'],
                "temperature": self.config['llm']['temperature'],
                "top_p": self.config['llm']['top_p'],
                "repetition_penalty": self.config['llm']['repetition_penalty'],
                "do_sample": self.config['llm']['do_sample'],
                "return_full_text": self.config['llm']['return_full_text']
            },
            device=self.config['llm']['device'],
        )

    @log_execution
    def load_context_metadata(self):
        loader = CSVLoader(
            file_path=self.config['context_loader']['file_path'], 
            csv_args=self.config['context_loader']['csv_args']
        )
        self.data = loader.load()

    @log_execution
    def create_vectordb(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['vectordb']['splitter']['chunk_size'], 
            chunk_overlap=self.config['vectordb']['splitter']['chunk_overlap']
        )
        docs = splitter.split_documents(self.data)

        embedding_function = HuggingFaceEmbeddings(
            model_name=self.config['vectordb']['embedding_function']['model_name'],
            model_kwargs=self.config['vectordb']['embedding_function']['model_kwargs']
        )

        self.qdrant_collection = Qdrant.from_documents(
            docs,
            embedding_function,
            location=self.config['vectordb']['qdrant']['location'],
            collection_name=self.config['vectordb']['qdrant']['collection_name']
        )
        
        self.retriever = self.qdrant_collection.as_retriever()
        torch.cuda.empty_cache()
