from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")   

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

chat_model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

def supervisor_call(state: AgentState) -> AgentState:
    print("Supervisor received message:")
    print(state['messages'][-1])
    # prompt = f"You are a Supervisor AI Assistant. Review the following response for accuracy and completeness: {state['messages'][-1].content}"
    # response = chat_model.invoke(state["messages"] + [{"role": "user", "content": prompt}])
    # return {"messages": [response]}

def router_call(state: AgentState) -> AgentState:
    print("Router received message:")
    print(state['messages'][-1])
    decision = "RAG"  # or "LLM" based on some criteria
    if decision == "RAG":
        return {"messages": [HumanMessage(content="RAG")]}
    else:
        return {"messages": [HumanMessage(content="LLM")]}    


def llm_call_with_graph(state: AgentState) -> AgentState:
    print(state['messages'][-1])
    prompt = f"You are an AI Assistant and please provide brief answer: {state['messages'][-1]}"
    response = chat_model.invoke(state["messages"] + [{"role": "user", "content": prompt}])
    state["messages"].append(response)
    return {"messages": [response]}

def rag_call_with_graph(state: AgentState) -> AgentState:
    print(state['messages'][-1])
    
    # Loading the documents
    loader = PyPDFLoader("waisl.pdf")
    loader.requests_kwargs = {'verify':False}
    documents=loader.load()
    print("Total pages" + str(len(documents)));

    # Splitting and Storing Vectors

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    print(f"Total chunks: {len(texts)}")
    all_chunks = []
    all_chunks.extend(texts)

    embeddings= OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
    collection_name="my_collection"
   )

    # Reloading Chromadb
    vector_store = Chroma(
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
    collection_name="my_collection"
    )
    
    # Similarity Search
    results = vector_store.similarity_search(state['messages'][-1].content, k=1)
    print(results[0].page_content)

    return state

class TopicSelector(BaseModel):
    topic: str = Field(description="The topic to search for")
    reason: str = Field(description="The reason for selecting this topic")  

pydanticParser = PydanticOutputParser(pydantic_object=TopicSelector)

workflow_agent_graph = StateGraph(AgentState)
workflow_agent_graph.add_node("rag node", rag_call_with_graph)
workflow_agent_graph.add_node("llm node", llm_call_with_graph)

workflow_agent_graph.add_node("supervisor node", supervisor_call)
workflow_agent_graph.set_entry_point("supervisor node")
workflow_agent_graph.add_conditional_edges(
    "supervisor node",
    router_call,
    {
        "RAG": "rag node",
        "LLM": "llm node"
    }
)

workflow_agent_graph.add_edge("llm node", END)
workflow_agent_graph.add_edge("rag node", END)


app = workflow_agent_graph.compile()
bytes = app.get_graph().draw_mermaid_png()
with open("workflow_agent_graph.png", "wb") as f:
    f.write(bytes)

print("Saved workflow_agent_graph.png in current folder")
# result = app.invoke({
#     "messages": [HumanMessage(content="Tell me about WAISL")]
# })

# print(result)