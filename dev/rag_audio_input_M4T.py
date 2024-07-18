# %%
from __future__ import annotations
from base import BaseConversationalRetrievalChain
import inspect
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain

class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    retriever: BaseRetriever
    """Retriever to use to fetch documents."""
    max_tokens_limit: Optional[int] = None
    """If set, enforces that the documents returned are less than this limit.
    This is only enforced if `combine_docs_chain` is of type StuffDocumentsChain."""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = self.retriever.invoke(
            question, config={"callbacks": run_manager.get_child()}
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = await self.retriever.ainvoke(
            question, config={"callbacks": run_manager.get_child()}
        )
        return self._reduce_tokens_below_limit(docs)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
from pprint import pprint
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from base import ConversationalRetrievalChain
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import JSONLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio

# %%
# repo_id = "google/gemma-1.1-2b-it"

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, max_length=200, temperature=0.7, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
# )

# %% [markdown]
# ## Prepare text data and create database. 

# %%
loader = JSONLoader(
    file_path="jsons/2016-01-01_0000_US_CNN_Erin_Burnett_OutFront.json",
    jq_schema='.content[]',
    text_content=False
)
data = loader.load()


# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
# from bs4 import BeautifulSoup as Soup
# url = "https://transcripts.cnn.com/date/2024-04-13"

# loader = RecursiveUrlLoader(
#     url=url, max_depth=5, extractor=lambda x: Soup(x, "html.parser").text
# )
# data = loader.load()

pprint(data)

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
docs = text_splitter.split_documents(data)


# %%
from typing import Any, Dict, List, Optional
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# embedding_function = HuggingFaceEmbeddings(model_name="bert-base-uncased", )  # all-MiniLM-L6-v2
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
class myHuggingFaceEmbeddings(BaseModel, Embeddings):
    processor: Any  #: :meta private:
    model: Any
    model_name: str = DEFAULT_MODEL_NAME
    cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    multi_process: bool = False
    show_progress: bool = False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(self.model_name)

    def embed_documents(self, texts):
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            # embeddings = self.client.encode(
            #     texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            # )
            text_inputs = self.processor(text = texts, src_lang="eng", return_tensors="pt")
            encoder_inputs = text_inputs["input_ids"]
            decoder_inputs = torch.tensor([[self.processor.tokenizer.cls_token_id]] * len(texts))  # or other appropriate decoder start token

            # Get the outputs from the model
            with torch.no_grad():
                outputs = self.model(input_ids=encoder_inputs, decoder_input_ids=decoder_inputs)
                
            embeddings = outputs.encoder_last_hidden_state.mean(dim=1)
            print(outputs.encoder_last_hidden_state.shape)
        return embeddings.tolist()
    def embed_query(self, text):
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

embedding_function =  myHuggingFaceEmbeddings(model_name="facebook/seamless-m4t-v2-large")

# embedding_function.embed_documents(['123', '456'])

# %%

# create a qdrant collection - a vector based index of all resumes
qdrant_collection = Qdrant.from_documents(
    docs,
    embedding_function,
    location=":memory:", # Local mode with in-memory storage only
    collection_name="resumes",
)

# %% [markdown]
# ## Audio processing utils.

# %%
# create the open-source embedding function
def load_audio(file_path, target_sr=16000):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

def get_audio_input(file_path):
    audio = load_audio(file_path)
    audio_tensor = torch.tensor(audio).float()  # Ensure data is float

    # Ensure the tensor is 1D
    if audio_tensor.dim() != 1:
        audio_tensor = audio_tensor.squeeze()  # Remove any singleton dimensions
    if audio_tensor.dim() == 0:
        audio_tensor = audio_tensor.unsqueeze(0)  # Handle rare case of a single sample
    audio_inputs_padded = pad_sequence([audio_tensor], batch_first=True, padding_value=0.0)
    return audio_inputs_padded

# Load audio processor and audio model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
speech_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# %% [markdown]
# ## Create a retriever with audio input and text candidates.
# 
# Note: We use a neural network to transform the dimension.

# %%
from langchain_core.retrievers import BaseRetriever

class MyRetriever(BaseRetriever):
    qdrant_collection: Qdrant=None

    class Config:
        arbitrary_types_allowed = True
    def __init__(self, qdrant_collection):
        super().__init__(qdrant_collection=qdrant_collection)
        self.qdrant_collection = qdrant_collection
    def get_relevant_documents(self, query_path):
        # audio_input = get_audio_input(query_path)
        audio, orig_freq =  torchaudio.load(query_path)
        audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
        
        # 预处理音频数据
        input_values = processor(audios=audio, return_tensors="pt", sampling_rate=16000)  # 返回tensor格式的输入值
        encoder_inputs = input_values["input_features"]
        decoder_inputs = torch.tensor([[processor.tokenizer.cls_token_id]] * len(audio))
        # 通过模型提取语音嵌入
        with torch.no_grad():
            outputs = speech_model(input_features=encoder_inputs, decoder_input_ids=decoder_inputs)
        embeddings = outputs.encoder_last_hidden_state.mean(dim=1).squeeze()
        docs = self.qdrant_collection.similarity_search_by_vector(embeddings.tolist())
        return docs
    # async def get_relevant_documents(self, query):
    #     return await self.qdrant_collection.asimilarity_search(query)
    
my_retriever = MyRetriever(qdrant_collection)  
# await my_retriever.invoke("how was 2015")
my_retriever.invoke("audio/synthetic_audio.wav")

# %%
# from torchaudio.utils import download_asset
# SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
# SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

# %% [markdown]
# Rewrite ConversationalRetrievalChain class, to make the input as audio

# %%



# %% [markdown]
# ## Usage
# 
# Note: The following has not be completed.

# %%
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original English.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)

# custom llm
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}
    
llm = CustomLLM(n=4)
llm("1234567")

conversational_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            chain_type="stuff",
            retriever=my_retriever,
            memory = memory,
            condense_question_prompt=CUSTOM_QUESTION_PROMPT
        )

# %%
# pprint(conversational_chain.invoke({"question": "donald trump has proved what"}))
# pprint(conversational_chain.invoke({"question": "what happened in 2015"}))
# pprint(conversational_chain.invoke({"question": "how was 2015"}))
# pprint(conversational_chain.invoke({"question": "2015 had what challenges"}))
# pprint(conversational_chain.invoke({"question": "What are people talking about trump?"}))

query_path = "audio/synthetic_audio2.wav"
pprint(conversational_chain.invoke({"question": query_path}))

# %%



