import locale 
def getpreferredencoding(do_setlocale = True): 
 return "UTF-8" 

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import AutoConfig,AutoTokenizer,BitsAndBytesConfig,AutoModelForCausalLM,pipeline
from langchain import HuggingFacePipeline
import torch
import torch.nn
#加载分词器
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=True,pad_token_id = 2)
#量化参数
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
#加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)

# input_tokenized = tokenizer.encode_plus("你好！你能给我提供一些有关gemini大模型的相关信息吗？",return_tensors="pt")['input_ids'].to('cuda')
# generated_ids = model.generate(input_tokenized,max_new_tokens=1000,do_sample=True)
# text =tokenizer.batch_decode(generated_ids)
# print(text)


from langchain.document_loaders import json_loader,PyPDFLoader
import nest_asyncio
from langchain.vectorstores import Chroma
from langchain.document_loaders.chromium import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings #使用不同的word2vec需要指定——sentence_transformer

from layout_recog import pdf_process,doc_process
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
from langchain_core.documents import Document

 

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    max_new_tokens=300
    # generation_config = Generation_config
)

llm = HuggingFacePipeline(
    pipeline=pipeline,
    )

prompt_template = """
### [INST] 
Instruction: 必须根据下面背景来回答这个问题:
​
{context}
​
### 问题:
{question} 
​
[/INST]
 """
from langchain.chains import LLMChain
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

def doc_pac(x):
  doc = Document(page_content=x, metadata={})
  return doc

llm_chain = LLMChain(llm=llm, prompt=prompt)
doc_mem = []
def get_file(input_file):
  temp_list = input_file.split('/')
  return temp_list[-1]

class Robot:
  def __init__(self):
    self.rag_chain = None
  def vec_doc(self,input_x):
    main_doc = []
    for i in input_x:
      if str(i).endswith('.pdf'):
        doc = pdf_process(str(i))
        split = text_splitter.split_text(doc)
        main_doc.extend(split)
      elif str(i).endswith('.doc') or str(i).endswith('.docx'):
        doc = doc_process(str(i))
        split = text_splitter.split_text(doc)
        main_doc.extend(split)
      else:
        continue
    split = [doc_pac(i) for i in main_doc]
    if split:
      vectordb = Chroma.from_documents(
      documents=split,
      embedding=embedding,
      )
      vec_retriever = vectordb.as_retriever(search_kwargs={"k": 2})
      bm25_retriever = BM25Retriever.from_documents(documents=split,
      embedding=embedding,)
      bm25_retriever.k = 2
      ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vec_retriever], weights=[0.9, 0.1])
 
    else:
      ensemble_retriever = {}
    self.rag_chain = ( 
 {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | llm_chain
)
  def chat(self,input):
    if self.rag_chain is None:
      return '请上传待解析文档'
    else:
      query = input
      answer = self.rag_chain.invoke(query)['text']
      return answer[answer.find('[/INST]')+7:]
