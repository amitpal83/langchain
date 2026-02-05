from langchain.tools import tool
from langgraph.prebuilt import ToolNode    
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")   

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def add(a: int, b: int) -> int:
    '''Adds two integers and returns the result.'''
    print("In the tool add")
    return a + b

@tool
def divide(a: int, b: int) -> float:
    '''Divides two integers and returns the result.'''
    print("In the tool divide")
    return a / b

@tool
def multiply(a: int, b: int) -> int:
    '''Multiplies two integers and returns the result.'''
    print("In the tool multiply")
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    '''Subtracts two integers and returns the result.'''
    print("In the tool subtract")
    return a - b

search=DuckDuckGoSearchRun()

tool_graph = StateGraph(AgentState)


def llm_call_with_graph(state: AgentState) -> AgentState:
    print("In llm_call_with_graph")
    query = state["messages"][-1]
    response = chat_model_with_tool.invoke(
        state["messages"]
    )
    return {"messages": [response]}

def router_call(state: AgentState) :
    print("Router received message:")
    print(state["messages"][-1])
    last_message = state["messages"][-1]
    if last_message.tool_calls :
        print("Routing to tool node")
        return "tool node"
    else:
        return END

chat_model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
chat_model_with_tool = chat_model.bind_tools([add, divide, multiply, search])

tool_node = ToolNode([add, divide, multiply, search])

tool_graph.add_node("llm node", llm_call_with_graph)
tool_graph.add_node("tool node", tool_node)
tool_graph.add_conditional_edges("llm node", router_call, {
    "tool node": "tool node",
    END: END}
)
tool_graph.add_edge("tool node", "llm node")
tool_graph.set_entry_point("llm node")


app = tool_graph.compile()
print("Generating tool_graph.png in current folder")
bytes = app.get_graph().draw_mermaid_png()

with open("multiple_tool_graph.png", "wb") as f:
    f.write(bytes)

response = app.invoke({
     "messages": [HumanMessage(content="What is 2 times the age of Narendra Modi in years?") ]})    

print(response)
