import os
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import HuggingFaceEndpoint
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
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
from torch.utils.data import Dataset, DataLoader
import huggingface_hub

class ConversationalChainWrapper:
    def __init__(self, repo_id, token, context_metadata_filename, collection_name="resumes"):
        self.repo_id = repo_id
        self.token = token
        self.context_metadata_filename = context_metadata_filename
        self.collection_name = collection_name

        self.load_context_metadata()
        self.create_vectordb()
        self.init_LLM()



        prompt = """
            <|user|>
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question} \n\nContext: {context} \n\nAnswer: <|end|>
            <|assistant|>
            """
        prompt = PromptTemplate(template=prompt, input_variables=['question', 'context'])
        
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
   
    def init_LLM(self):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.repo_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 200,  # to be confirmed 
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.5,
                "do_sample": True
            },
            device=0,
            
        )
        # self.llm = HuggingFaceEndpoint(
        #     repo_id=self.repo_id, max_length=200, temperature=0.7, token=self.token
        # )
        print(">>> Successfully initialize LLM!")

    def load_context_metadata(self):
        print(">>> Loading context and metadata ... ")
        loader = CSVLoader(file_path=f'dataset/{self.context_metadata_filename}', 
                            csv_args={
                            'delimiter': ',',
                            'quotechar': '"',
                            'fieldnames': ['context','metadata']
                        })
        self.data = loader.load()
        print(">>> Successfully load context and metadata!")

    def create_vectordb(self):
        print(">>> Creating vectorDB ...")
        # from concurrent.futures import ThreadPoolExecutor
        # from tqdm import tqdm

        # def split_document(doc):
        #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        #     return text_splitter.split_documents([doc])
        
        # with ThreadPoolExecutor() as executor:
            # chunks = list(tqdm(executor.map(split_document, self.data), total=len(self.data), desc="Splitting documents"))
        
        # docs = [chunk for sublist in chunks for chunk in sublist]

        splitter =  RecursiveCharacterTextSplitter(
                                chunk_size=200, 
                                chunk_overlap=50
                                )
        docs = splitter.split_documents(self.data)

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

        print(">>> Successfully create qdrant collection and retriever!")


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
    import os
    from dotenv import load_dotenv
    load_dotenv()
    langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
    langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
    huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')


    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    # repo_id = "google/gemma-2b"
    context_metadata_filename = "2016_01_english_with_metadata_small.csv"

    conversational_chain = ConversationalChainWrapper(repo_id, huggingfacehub_api_token, context_metadata_filename)

    with open('dataset/format_QAdata.json') as f:
        json_data = json.load(f)

    dataset = QuestionsDataset(json_data)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    name = "Phi-3-mini-4k-instruct"
    time = time.strftime("%Y-%m-%d-%H-%M-%S")

    collected_data = {
        'question': [],
        'ground_truth': [],
        'answer': [],
        'contexts': []
    }
    for i, batch in enumerate(data_loader):
        try:
            batch = zip(batch[0], batch[1])
            for question, ground_truth in batch:
                result = conversational_chain.rag_chain_with_source.invoke(question)
                context = [doc.page_content for doc in result['context']]
                response = result['answer']
                collected_data['contexts'].append(context)
                collected_data['answer'].append(response)
                collected_data['question'].append(question)
                collected_data['ground_truth'].append(ground_truth)

                print(f"    question: {question}\n\n    ground_truth: {ground_truth}\n\n    answer: {response}\n\n    context: {context}\n\n ---end--- \n\n")
            
            with open(f"out/{name}-{time}.json", 'a+') as f:
                json.dump(collected_data, f)
            print(f"batch {i} finished.")
            exit()
        except huggingface_hub.utils._errors.HfHubHTTPError:
            print(f"batch {i} failed.")
            
            




    
