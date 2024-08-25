import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import csv
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils import log_execution
from langchain_huggingface import HuggingFaceEndpoint
from langchain.retrievers import EnsembleRetriever

from langchain.output_parsers import GuardrailsOutputParser
# from langserve.client import RemoteRunnable


import os
from dotenv import load_dotenv
load_dotenv()

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

class RagPipeline:
    def __init__(self, config):
        self.config = config
        self.repo_id = config['llm']['model_id']
        self.token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

        if self.config['use_guardrails']:
            rail_str = """
                <rail version="0.1">
                <output>
                    <string 
                        description="Profanity-free translation" 
                        format="is-profanity-free" 
                        name="translated_statement" 
                        on-fail-is-profanity-free="fix">
                    </string>
                </output>
                <prompt>
                    Translate the given statement into English:

                    ${statement_to_be_translated}

                    ${gr.complete_json_suffix}
                </prompt>
                </rail>
                """
            # output_parser = GuardrailsOutputParser.from_rail_string(rail_str)
            # output_parser = RemoteRunnable("http://localhost:8000/guardrails-output-parser")
        else:
            output_parser = StrOutputParser()

        if self.config['use_rag']:
            self.load_context_metadata()
            self.create_vectordb()
            self.init_LLM()

            if self.config['dataset']['language'] == 'en':
                if self.config['llm']['model_id'] == "RedHenLabs/news-reporter-3b":
                    prompt_template = PromptTemplate(
                        template=config['prompt']['fine_tuned_rag'],
                        input_variables=config['prompt']['input_variables_rag']
                    )
                else:
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
                context_size = config['context_size']
                return "\n\n".join(doc.page_content for doc in docs[:context_size])
            
            rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt_template
                | self.llm
                | output_parser
            )
            
            self.conversation_chain = RunnableParallel(
                {"context": self.retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)
        else:
            self.init_LLM()
            if self.config['dataset']['language'] == 'en':
                if self.config['llm']['model_id'] == "RedHenLabs/news-reporter-3b":
                    prompt_template = PromptTemplate(
                        template=config['prompt']['fine_tuned_without_rag'],
                        input_variables=config['prompt']['input_variables_without_rag']
                    )
                else:
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
                prompt_template | self.llm | output_parser
                )
            self.conversation_chain = RunnableParallel(
                {"question": RunnablePassthrough()}
            ).assign(answer=chain)
        
        # if self.config['use_guardrails']:
        #     self.conversation_chain = self.conversation_chain.with_types(output_type=dict)

        # guardrails_config = RailsConfig.from_path(config['guardrails']['config_path'])
        # self.guardrails = RunnableRails(guardrails_config)

        # self.chain_with_guardrails =  self.conversation_chain | self.guardrails

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
                device_map=self.config['llm']['device_map'],
            )
        else:
            self.llm = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                max_new_tokens=self.config['llm']['max_new_tokens'],
                temperature=self.config['llm']['temperature']
            )

    @log_execution
    def load_context_metadata(self):
        if self.config['vectordb']['create_new_collection']:
            csv.field_size_limit(10**6)
            loader = CSVLoader(
                file_path=self.config['context_loader']['file_path'], 
                csv_args=self.config['context_loader']['csv_args'],
                metadata_columns = self.config['context_loader']['metadata_columns']
            )
            self.data = loader.load()

    @log_execution
    def create_vectordb(self):
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.config['vectordb']['embedding_function']['model_name'],
            model_kwargs=self.config['vectordb']['embedding_function']['model_kwargs']
        )
        if self.config['vectordb']['create_new_collection']:
            print("Creating new collection ...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['vectordb']['splitter']['chunk_size'], 
                chunk_overlap=self.config['vectordb']['splitter']['chunk_overlap']
            )
            docs = splitter.split_documents(self.data)

            self.qdrant_collection = Qdrant.from_documents(
                docs,
                embedding_function,
                path=self.config['vectordb']['qdrant']['path'],
                collection_name=self.config['vectordb']['qdrant']['collection_name'],
            )
        else:
            print("Using existing collection ...")
            if self.config['vectordb']['qdrant']['use_12_month']:
                self.retrievers = []
                # import pdb; pdb.set_trace()
                for i, collection_name in enumerate(self.config['vectordb']['qdrant']['collection_name_12']):
                    qdrant_collection = Qdrant.from_existing_collection(
                        collection_name=collection_name,
                        embedding=embedding_function,
                        path=self.config['vectordb']['qdrant']['path_12'][i],
                    )
                    self.retrievers.append(qdrant_collection.as_retriever())
                self.retriever = EnsembleRetriever(retrievers=self.retrievers)

            else:
                self.qdrant_collection = Qdrant.from_existing_collection(
                    collection_name=self.config['vectordb']['qdrant']['collection_name'],
                    embedding=embedding_function,
                    path=self.config['vectordb']['qdrant']['path']
                )

                self.retriever = self.qdrant_collection.as_retriever()
        torch.cuda.empty_cache()
