from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough      
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

class TopicSelector(BaseModel):
    topic: str = Field(description="The topic to search for")
    reason: str = Field(description="The reason for selecting this topic")  

pydanticParser = PydanticOutputParser(pydantic_object=TopicSelector)
chat_model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to build vector database if not exists

def build_vector_db():
    if os.path.exists("./chroma_langchain_db"):
        return

    loader = PyPDFLoader("waisl.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

 #   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_langchain_db",
        collection_name="my_collection"
    )

# Supervisor Node to check if query is related to WAISL 
def supervisor_call(state: AgentState) -> AgentState:
    print("Supervisor received message:")
    print(state['messages'][-1])
    user_query = state['messages'][-1]
    # prompt = f"You are a Supervisor AI Assistant. Review the following response for accuracy and completeness: {state['messages'][-1].content}"
    # response = chat_model.invoke(state["messages"] + [{"role": "user", "content": prompt}])
   
    # templateQuery="""WAISL is Wipro Airport Infrastructure Services Limited which provides Airport Infrastructure Services. You have to look at the user query {user_query}\n and return {topic} only as  either 'WAISL' or 'NOT RELATED' """

    templateQuery = """
    WAISL is Wipro Airport Infrastructure Services Limited which provides Airport Infrastructure Services. You have to look at the

    User Query: {user_query}
    Based on the user query, decide if it is related to WAISL or not
    Respond ONLY with:
    - WAISL
    - NOT RELATED

    {format_instructions}
    """
    prompt = PromptTemplate(
        input_variables=["user_query"],
        template=templateQuery,
        partial_variables={"format_instructions": pydanticParser.get_format_instructions()}
    ) 

    chain = prompt | chat_model | pydanticParser
    response = chain.invoke({"user_query": user_query.content})
    print(f"Supervisor decision: {response.topic} because {response.reason}")
    return {"messages": [AIMessage(content=response.topic)]}

# Router function to route calls to LLM or RAG based on supervisor decision
def router_call(state: AgentState) -> AgentState:
    print("Router received message:")
    print(state["messages"][-1])
    decision = state["messages"][-1].content  # or "LLM" based on some criteria
    if decision == "WAISL" :
        return "RAG"
    else:
        return "LLM"    

# LLM Node to return the response from LLM
def llm_call_with_graph(state: AgentState) -> AgentState:
    query = state['messages'][0].content

    prompt = PromptTemplate(
        template="You are an AI Assistant and please provide brief answer: {query}",
        input_variables=["query"]
    )

    chain = prompt | chat_model | StrOutputParser()
    response = chain.invoke({"query": query})

    return {"messages": [AIMessage(content=response)]}

# RAG Node to return the response from RAG chain
def rag_call_with_graph(state: AgentState) -> AgentState:
    print(f"User Query in RAG Call is {state['messages'][0].content}")
  
    # Similarity Search
    results = vector_store.similarity_search(state['messages'][0].content, k=1)
  

    # Implementing RAG CHain

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

    responsecontent = rag_chain.invoke(
    {
        "question":state['messages'][0].content
    })

    print(responsecontent)

    return {"messages": [AIMessage(content=responsecontent)]}

# buidling and loading vector db

build_vector_db()
vector_store = Chroma(
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# Creating Workflow Agent Graph
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
result = app.invoke({
     "messages": [HumanMessage(content="Tell me about WAISL and its revenue in 2024-25 and how is doing financially?") ]
 })

print(result)