# off the shelf model, with rag, 12-month vectorDB
python src/main.py --cfg-options \
vectordb.create_new_collection=true \
llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
use_rag=true \
context_loader.file_path="/home/featurize/work/Multimodal-RAG/data/context_metadata/processed_2016_10_english.csv" \
vectordb.qdrant.path="data/db_english_2016_10" \
vectordb.qdrant.collection_name="english_2016_10" \
llm.use_endpoint=true \
dataset.file="data/QA/format_QAdata_small.json" \

python src/main.py --cfg-options \
vectordb.create_new_collection=true \
llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
use_rag=true \
context_loader.file_path="/home/featurize/work/Multimodal-RAG/data/context_metadata/processed_2016_11_english.csv" \
vectordb.qdrant.path="data/db_english_2016_11" \
vectordb.qdrant.collection_name="english_2016_11" \
llm.use_endpoint=true \
dataset.file="data/QA/format_QAdata_small.json" \

python src/main.py --cfg-options \
vectordb.create_new_collection=true \
llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
use_rag=true \
context_loader.file_path="/home/featurize/work/Multimodal-RAG/data/context_metadata/processed_2016_12_english.csv" \
vectordb.qdrant.path="data/db_english_2016_12" \
vectordb.qdrant.collection_name="english_2016_12" \
llm.use_endpoint=true \
dataset.file="data/QA/format_QAdata_small.json" \