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



load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")   

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]




@tool
def provideWeatherDetails(location: str) -> str:
    '''Provides weather details for a given location.'''
    # Dummy implementation for weather details
    print("In the tool provideWeatherDetails")

    if "delhi" in location.lower():
        return f"The current weather in Delhi is sunny with a temperature of 35°C."
    return f"The current weather is sunny with a temperature of 25°C."


# chat_model_with_tool = chat_model.bind_tools([provideWeatherDetails])
# response_with_tool = chat_model_with_tool.invoke("What is the temperature in delhi?")
# print(response_with_tool.content)



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
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "provideWeatherDetails":
        print("Routing to tool node")
        return "tool node"
    else:
        return END

       

chat_model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
chat_model_with_tool = chat_model.bind_tools([provideWeatherDetails])


tool_node = ToolNode([provideWeatherDetails])

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

with open("tool_graph.png", "wb") as f:
    f.write(bytes)

response = app.stream({
     "messages": [HumanMessage(content="What is weather in France?") ]})    

for res in response:
    for key, value in res.items():
        print(f"Key is {key} , Value is {value}")




