import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
from sentence_transformers import  util

# processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
# speech_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
def embed_documents(texts):
    model_name = "facebook/seamless-m4t-v2-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = SeamlessM4Tv2Model.from_pretrained(model_name)
    text_inputs = processor(text = texts, src_lang="eng", return_tensors="pt")
    encoder_inputs = text_inputs["input_ids"]
    decoder_inputs = torch.tensor([[processor.tokenizer.cls_token_id]] * len(texts))  # or other appropriate decodert token

    # Get the outputs from the model
    with torch.no_grad():
        outputs = model(input_ids=encoder_inputs, decoder_input_ids=decoder_inputs)
        
    embeddings = outputs.encoder_last_hidden_state[:,0,:]
        # print(outputs.encoder_last_hidden_state.shape)
    return embeddings.tolist()


# model_name="Sakil/sentence_similarity_semantic_search"
# model = SentenceTransformer(model_name)
# sentences = ['A man is eating food.',
#           'Please keep me informed.',
#           'Time is running out.',
#           'That makes no difference.',
#           'I am not satisfied with his answer.',
#           'Two men pushed carts through the woods.',
#           'You are too generous. ',
#           'He is so busy.',
#           'Someone in a gorilla costume is playing a set of drums.'
#           ]
sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]

#Encode all sentences
embeddings = embed_documents(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []

for i in range(len(cos_sim)-1):

    for j in range(i+1, len(cos_sim)):
    
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score

all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")

for score, i, j in all_sentence_combinations[0:5]:

    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))