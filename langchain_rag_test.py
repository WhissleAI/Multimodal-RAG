import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Load, chunk and index the contents of the blog.
bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2", 
    model_kwargs={'device': 'cuda'}
)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
# llm = HuggingFacePipeline.from_model_id(
#     model_id="google/gemma-2b",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 150,  # to be confirmed 
#         "top_k": 20,
#         "temperature": 0.7,
#         "do_sample": True
#     },
#     device=0
# )
from langchain_community.llms import HuggingFaceEndpoint
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b", max_length=200, temperature=0.7, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)
print(">>> Successfully initialize LLM!")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# custom_template = """
#     "role": "user",
#     "content": 
#     "We have provided context information below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given this information, please answer the question: {query}"
#     """
# CUSTOM_QUESTION_PROMPT = PromptTemplate(template=custom_template, input_variables=['context_str', 'query'])

# rag_chain = LLMChain(prompt=prompt, llm=llm)

query = "What is Task Decomposition?"
context = retriever.invoke(input=query)
context = [c.page_content for c in context]
context_str = "\n\n".join(context)
res = rag_chain.invoke(query)
print(res)
# print(rag_chain.invoke())

'''
Human: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know16:06:26 [8/1755$
 say that you don't know. Use three sentences maximum and keep the answer concise.                                                                                
Question: What is Task Decomposition?                                                                                                                             
Context: Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple 
thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search
) with each state evaluated by a classifier (via a prompt) or majority vote.                                                                                      
Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific 
instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.                                                                        
                                                                                                                                                                  
(3) Task execution: Expert models execute on the specific tasks and log results.                                                                                  
Instruction:                                                                                                                                                      
                                                                                                                                                                  
With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User I
nput }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a str
aightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results conta
in a file path, must tell the user the complete file path.                       

Fig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)                                                                                   
The system comprises of 4 stages:                                                
(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, de
pendencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.
Instruction:                            

Finite context length: The restricted context capacity limits the inclusion of historical information, detailed instructions, API call context, and responses. The
 design of the system has to work with this limited communication bandwidth, while mechanisms like self-reflection to learn from past mistakes would benefit a lot
 from long or infinite context windows. Although vector stores and retrieval can provide access to a larger knowledge pool, their representation power is not as p
owerful as full attention.              


Challenges in long-term planning and task decomposition: Planning over a lengthy history and effectively exploring the solution space remain challenging. LLMs str
uggle to adjust plans when faced with unexpected errors, making them less robust compared to humans who learn from trial and error. 
Answer:          

The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predi
ctions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results
 to the user in the first person. If inference results contain a file path, must tell the user the complete file path.

Fig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)                                                                                   
The system comprises of 4 stages:                                                
(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There 
'''