from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf_pipeline = HuggingFacePipeline(pipeline=pipe)
from langchain_huggingface import HuggingFaceEndpoint

endpoint_url = "your_endpoint_url"  # 替换为实际的端点 URL
hf_endpoint = HuggingFaceEndpoint(
    repo_id=model_id,
    max_new_tokens = 10
)
input_text = "Once upon a time"

# 使用 HuggingFacePipeline 生成
output_pipeline = hf_pipeline(input_text)
print("Pipeline Output:", output_pipeline)

# 使用 HuggingFaceEndpoint 生成
output_endpoint = hf_endpoint(input_text)
print("Endpoint Output:", output_endpoint)
