export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS=96

python src/main.py --cfg-options \
    vectordb.create_new_collection=false \
    vectordb.qdrant.use_12_month=true \
    llm.model_id="RedHenLabs/news-reporter-3b" \
    dataset.file="data/QA/format_QAdata.json" \
    dataset.size=120 \
    use_rag=true \
    use_guardrails=false \
    llm.use_endpoint=false \


# python src/main.py --cfg-options \
#     vectordb.create_new_collection=false \
#     vectordb.qdrant.use_12_month=true \
#     llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
#     llm.temperature=0.8 \
#     dataset.file="data/QA/format_QAdata.json" \
#     use_rag=true \
#     use_guardrails=false \
#     llm.use_endpoint=true \