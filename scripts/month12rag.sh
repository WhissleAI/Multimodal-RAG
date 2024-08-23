python src/main.py --cfg-options \
vectordb.create_new_collection=true \
vectordb.qdrant.use_12_month=true \
llm.model_id="microsoft/Phi-3-mini-4k-instruct" \
use_rag=true \
use_guardrails=false \
llm.use_endpoint=false \

python src/main.py --cfg-options \
vectordb.create_new_collection=true \
vectordb.qdrant.use_12_month=true \
llm.model_id="RedHenLabs/news-reporter-3b" \
use_rag=true \
use_guardrails=false \
llm.use_endpoint=false \
