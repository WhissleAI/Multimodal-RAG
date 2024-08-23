from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
month12db = Qdrant.from_existing_collection(
                collection_name="english_2016_01",
                embedding=embedding_function,
                path="data/db_english_2016_combined"
            )

file_paths = ["processed_2016_02_english.csv",
              "processed_2016_03_english.csv",
              "processed_2016_04_english.csv",
              "processed_2016_05_english.csv",
              "processed_2016_06_english.csv",
              "processed_2016_07_english.csv",
              "processed_2016_08_english.csv",
              "processed_2016_09_english.csv",
              "processed_2016_10_english.csv",
              "processed_2016_11_english.csv",
              "processed_2016_12_english.csv"]
for file_path in file_paths:             
        loader = CSVLoader(
            file_path=f"/home/yfg2/Multimodal-RAG/data/context_metadata/processed_eng_context_metadata/{file_path}",
            csv_args={"delimiter": ',', "quotechar": '"'},
            metadata_columns = self.config['context_loader']['metadata_columns']
        )
        self.data = loader.load()

        splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['vectordb']['splitter']['chunk_size'], 
                chunk_overlap=self.config['vectordb']['splitter']['chunk_overlap']
            )
        docs = splitter.split_documents(self.data)