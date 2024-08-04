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
from langchain_huggingface import HuggingFaceEndpoint

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

        if self.config['use_rag']:
            self.load_context_metadata()
            self.create_vectordb()
            self.init_LLM()

            if self.config['dataset']['language'] == 'en':
                prompt_template = PromptTemplate(
                    template=config['prompt']['template_en_rag'],
                    input_variables=config['prompt']['input_variables_rag']
                )
            elif self.config['dataset']['language'] == 'fr':
                prompt_template = PromptTemplate(
                    template=config['prompt']['template_fr_rag'],
                    input_variables=config['prompt']['input_variables_rag']
                )
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            self.conversation_chain = RunnableParallel(
                {"context": self.retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)
        else:
            self.init_LLM()
            if self.config['dataset']['language'] == 'en':
                prompt_template = PromptTemplate(
                    template=config['prompt']['template_en_without_rag'],
                    input_variables=config['prompt']['input_variables_without_rag']
                )
            elif self.config['dataset']['language'] == 'fr':
                prompt_template = PromptTemplate(
                    template=config['prompt']['template_fr_without_rag'],
                    input_variables=config['prompt']['input_variables_without_rag']
                )
            chain = (
                prompt_template | self.llm | StrOutputParser()
                )
            self.conversation_chain = RunnableParallel(
                {"question": RunnablePassthrough()}
            ).assign(answer=chain)
            
    @log_execution
    def init_LLM(self):
        if not self.config['llm']['use_endpoint']:
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.repo_id,
                task=self.config['llm']['task'],
                batch_size=self.config['llm']['batch_size'],
                pipeline_kwargs={
                    "max_new_tokens": self.config['llm']['max_new_tokens'],
                    "temperature": self.config['llm']['temperature'],
                    "repetition_penalty": self.config['llm']['repetition_penalty'],
                    "do_sample": self.config['llm']['do_sample'],
                    "return_full_text": self.config['llm']['return_full_text']
                },
                device=self.config['llm']['device'],
            )
        else:
            self.llm = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                max_new_tokens=self.config['llm']['max_new_tokens'],
                temperature=self.config['llm']['temperature']
            )

    @log_execution
    def load_context_metadata(self):
        loader = CSVLoader(
            file_path=self.config['context_loader']['file_path'], 
            csv_args=self.config['context_loader']['csv_args'],
            metadata_columns = self.config['context_loader']['metadata_columns']
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
