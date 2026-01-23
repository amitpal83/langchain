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
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda




     

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")   


speechloader= TextLoader(file_path="speech.txt", encoding="utf-8")
documents= speechloader.load()

def splitStoreVectors(documents, vector_store):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = []

    for i, report in enumerate(documents, start=1):
        chunks = text_splitter.split_text(report.page_content)
        print(f"Document {i} → {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    # Add ALL chunks at once
    vector_store.add_texts(all_chunks)
    # Save ONCE
    vector_store.save_local("faiss_index")

def queryVectorStore(vector_store, query):
    docs = vector_store.similarity_search(
        query,
        k=5
    )   
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loader = PyPDFLoader("waisl.pdf")
loader.requests_kwargs = {'verify':False}
documents =loader.load()
print("Total pages" + str(len(documents)));

 #Open AI embeddings
embeddings= OpenAIEmbeddings(model="text-embedding-3-small")
faiss_index = faiss.IndexFlatL2(1536)  # Dimension for text-embedding-3-large is 1536
vector_store = FAISS(embeddings, faiss_index, InMemoryDocstore({}), {})

splitStoreVectors(documents, vector_store)
new_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
results = queryVectorStore(new_vector_store, "Revenue of WAISL in 2024-25") 

# Initialize the LLM ,  prompt and parser
llm= ChatOpenAI(model_name="gpt-4o", temperature=0)
prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant and have to read the text provided by the retriever in Context and answer the question asked by the user based on that text only. If you don't know the answer, just say that you don't know, don't try to make up an answer."),
    ("user", "Context: {context} \n\n Question: {question}")
])
parser= StrOutputParser()

# Create RAG chain
context_runnable = RunnableLambda(lambda _: format_docs(results))
rag_chain = (
    {
        "context": context_runnable,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

content = rag_chain.invoke(
    "What is the revenue of WAISL for 2024-25 and how would you rate the revenue growth compared to previous years?"
)

print(content)
