import locale 
def getpreferredencoding(do_setlocale = True): 
 return "UTF-8" 


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

from layout_recog import layoo
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
    max_new_tokens=200
    # generation_config = Generation_config
)

llm = HuggingFacePipeline(
    pipeline=pipeline,
    )

prompt_template = """
### [INST] 
Instruction: 根据你的了解和下面背景来回答这个问题，如果背景不相关你可以忽略。
这里有一些背景可以帮助你:
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
class Robot:
  def __init__(self):
    self.rag_chain = None
  def vec_doc(self,input_x):
    doc = layoo(input_x)
    print('**************')
    print(doc[:200])
    split = text_splitter.split_text(doc)
    split = [doc_pac(i) for i in split]
    vectordb = Chroma.from_documents(
    documents=split,
    embedding=embedding,
    )
    retriever = vectordb.as_retriever()
    self.rag_chain = ( 
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
  def chat(self,input):
    if self.rag_chain is None:
      return '请上传待解析文档'
    else:
      query = input
      answer = self.rag_chain.invoke(query)['text']
      return answer[answer.find('[/INST]')+7:]
