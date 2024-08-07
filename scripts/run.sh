export HUGGINGFACE_TOKEN="<your-huggingface-token>"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="<your-langchain-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"


# off the shelf model, no rag
python src/main.py --cfg-options \
    llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
    use_rag=false \

# off the shelf model, with rag, 1-month vectorDB
python src/main.py --cfg-options \
    llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
    use_rag=true \
    vector_db_path="data/vector_db/2016_01_english_with_metadata.csv" \

# off the shelf model, with rag, 12-month vectorDB
# python src/main.py --cfg-options

# fine-tuned model, no rag
python src/main.py --cfg-options \
    llm.model_id="RedHenLabs/news-reporter-3b" \
    use_rag=false \

# fine-tuned model, with rag, 1-month vectorDB
python src/main.py --cfg-options \
    llm.model_id="RedHenLabs/news-reporter-3b" \
    use_rag=true \
    vector_db_path="data/vector_db/2016_01_english_with_metadata.csv" \


# fine-tuned model, with rag, 12-month vectorDB
# python src/main.py --cfg-optionsf