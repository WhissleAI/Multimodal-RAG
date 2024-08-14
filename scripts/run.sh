export HUGGINGFACE_TOKEN="<your-huggingface-token>"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="<your-langchain-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com


# off the shelf model, no rag
# python src/main.py --cfg-options \
#     llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
#     use_rag=false \
#     llm.use_endpoint=false \
#     dataset.file="data/QA/format_QAdata.json" \
#     context_loader.file_path="data/context_metadata/processed_2016_01_english_with_metadata.csv" \



# off the shelf model, with rag, 1-month vectorDB
python src/main.py --cfg-options \
llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
use_rag=true \
context_loader.file_path="data/context_metadata/processed_2016_01_english_with_metadata.csv" \
llm.use_endpoint=false \
dataset.file="data/QA/format_QAdata_small.json" \

# off the shelf model, with rag, 12-month vectorDB
# python src/main.py --cfg-options
# fine-tuned model, no rag
# python src/main.py --cfg-options \
#     llm.model_id="RedHenLabs/news-reporter-3b" \
#     use_rag=false \

# fine-tuned model, with rag, 1-month vectorDB
# python src/main.py --cfg-options \
#     llm.model_id="RedHenLabs/news-reporter-3b" \
#     use_rag=true \
#     vector_db_path="data/vector_db/2016_01_english_with_metadata.csv" \


# fine-tuned model, with rag, 12-month vectorDB
# python src/main.py --cfg-optionsf