from transformers import pipeline

pipe = pipeline('text-generation', model='lucifertrj/exp4', device=-1)

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180) 
    return outputs[0]['generated_text'][len(prompt):].strip()


if __name__ == '__main__':
    print(test_inference("Hello, tell me about Trump."))
