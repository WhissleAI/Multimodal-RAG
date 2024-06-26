import os
from langchain_huggingface import HuggingFacePipeline
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json
import time
import torch

class ConversationalChainWrapper:
    def __init__(self, repo_id, token, context_metadata_filename, collection_name="resumes"):
        self.repo_id = repo_id
        self.token = token
        self.context_metadata_filename = context_metadata_filename
        self.collection_name = collection_name

        self.load_context_metadata()
        self.create_vectordb()
        self.init_LLM()

        import os

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = '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'

        custom_template = """
            "role": "user",
            "content": 
            "We have provided context information below. \n"
            "---------------------\n"
            "{context}"
            "\n---------------------\n"
            "Given this information, please answer the question: {question}"
            """
        prompt = PromptTemplate(template=custom_template, input_variables=['context', 'question'])

        # self.llm_chain = LLMChain(prompt=self.CUSTOM_QUESTION_PROMPT, llm=self.llm)

        # prompt = hub.pull("rlm/rag-prompt")
        # import pdb; pdb.set_trace()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        
        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)
        self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        import pdb; pdb.set_trace()
   
    def init_LLM(self):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.repo_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 150,  # to be confirmed 
                "top_k": 50,
                "temperature": 0.25,
                "do_sample": True
            },
            device=0
        )
        print(">>> Successfully initialize LLM!")

    def load_context_metadata(self):
        print(">>> Loading context and metadata ... ")
        loader = CSVLoader(file_path=f'dataset/{self.context_metadata_filename}')
        self.data = loader.load()
        print(">>> Successfully load context and metadata!")

    def create_vectordb(self):
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        print(">>> Creating vectorDB ...")
        
        # Split documents in parallel
        def split_document(doc):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            return text_splitter.split_documents([doc])
        
        with ThreadPoolExecutor() as executor:
            chunks = list(tqdm(executor.map(split_document, self.data), total=len(self.data), desc="Splitting documents"))
        
        # Flatten the list of lists
        docs = [chunk for sublist in chunks for chunk in sublist]

        # Initialize the embedding function
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={'device': 'cuda'}
        )

        self.qdrant_collection = Qdrant.from_documents(
            docs,
            embedding_function,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name=self.collection_name,
        )
        
        self.retriever = self.qdrant_collection.as_retriever()
        torch.cuda.empty_cache()

        print(">>> Successfully create qdrant collection!")

from torch.utils.data import Dataset, DataLoader

class QuestionsDataset(Dataset):
    def __init__(self, json_data):
        self.questions = json_data['question']
        self.ground_truth = json_data['ground_truth']
        self.question_ground_truth = [pair for pair in zip(self.questions, self.ground_truth)]

    def __len__(self):
        assert len(self.ground_truth)==len(self.questions)
        return len(self.questions)

    def __getitem__(self, idx):
        return self.question_ground_truth[idx]
    
# Usage
if __name__ == "__main__":
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # new token needed
    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    repo_id = "google/gemma-2b"
    context_metadata_filename = "2016_01_english_with_metadata_small.csv"

    conversational_chain = ConversationalChainWrapper(repo_id, os.environ["HUGGINGFACEHUB_API_TOKEN"], context_metadata_filename)

    with open('dataset/format_QAdata.json') as f:
        json_data = json.load(f)


    dataset = QuestionsDataset(json_data)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    name = "Phi-3-mini-4k-instruct"
    time = time.strftime("%Y-%m-%d-%H-%M-%S")

    
    for batch in data_loader:
        batch = zip(batch[0], batch[1])
        for question, ground_truth in batch:
            result = conversational_chain.rag_chain_with_source.invoke(question)
            context = [doc.page_content for doc in result['context']]
            response = result['answer']
            json_data['contexts'].append(context)
            json_data['answer'].append(response)

            print(f"    question: {question}\n\n    ground_truth: {ground_truth}\n\n    answer: {response}\n\n    context: {context}\n\n ---end--- \n\n")

        with open(f"out/{name}-{time}.json", 'a+') as f:
            json.dump(json_data, f)
            




    
