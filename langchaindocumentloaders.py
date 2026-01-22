import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq 
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings



     

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")   


speechloader= TextLoader(file_path="speech.txt", encoding="utf-8")
documents= speechloader.load()

# print(documents)

# Initialize LLMs
llm= ChatOpenAI(model_name="gpt-4o", temperature=0)

prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can identify the content and find details about the content "),
    ("human", "Content to identify is: {input}")])

parser= StrOutputParser()

chain= prompt | llm | parser
content = chain.invoke({"input": documents[0].page_content})

#pdf loader
loader = PyPDFLoader("sample-report.pdf")
reports =loader.load()

#Webbased Loader

web_loader= WebBaseLoader("https://langchain.com/docs/getting-started/introduction/")
loader.requests_kwargs = {'verify':False}
web_documents= web_loader.load()
# print(web_documents[0].page_content)    

 #Open AI embeddings
embeddings= OpenAIEmbeddings(model="text-embedding-3-large")

i=0
for report in reports:
    i=i+1;
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(report.page_content)
    print(f'Number of document {i} ',texts)
    vector = embeddings.embed_documents(texts)
    print(vector)

