from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
import os
import getpass
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
from typing import TypedDict


load_dotenv()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

print(os.getenv("TAVILY_API_KEY"))

#initialize chat model
model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

#initialize Tavily search tool
toolTavily = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)


#initialize Youtube search tool
youtube_tool = YouTubeSearchTool(
    max_results=3,
    search_region="US",
    search_language="en"
)   
# Simple Toolcall
res= toolTavily.invoke({"query": "What won  the last wimbledon"})

# Tool call with Agent and LLM
def tool_call_with_agent(tool,model):
    agent = create_agent(model, [tool])
    user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."
    result = ""
    for step in agent.stream(
        {"messages": user_input},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        # Safely extract text
        if hasattr(msg, "content") and msg.content:
            result += msg.content

    return result

# response = tool_call_with_agent(toolTavily,model)
# print(response)

# Calling Youtube Search Tool
# res= youtube_tool.invoke({"query": "Ind Vs NZ 3rd T20 Highlights,2"})
# print(res)


# Creating custom tool
@tool
def multiply(a: int, b: int) -> int:
    """This code is for multiplying two numbers."""
    return a * b

resultmultiply = multiply.invoke({'a':5,'b':6})
print(f"Multiplication Result: {resultmultiply}")




# State Class

class GraphState(TypedDict):
    input: str


firstgraph = StateGraph(GraphState)

# Greeting Node
def greeting_node(state: GraphState) -> GraphState:
    print(state["input"])
    state["input"] = f"Hello, {state['input']} welcome to the LangGraph!"
    return state




#Farewell Node
def farewell_node(state: GraphState) -> GraphState:
    print(state["input"])
    state["input"] = f"Thanks for exploring..Goodbye! {state['input']} Have a great day!"
    print(state["input"])
    return state

# Creating Graph
firstgraph.add_node("greeting node", greeting_node)
firstgraph.add_node("farewell node", farewell_node)
firstgraph.add_edge("greeting node", "farewell node")
firstgraph.set_entry_point("greeting node")
firstgraph.set_finish_point("farewell node")
app= firstgraph.compile()
png_bytes = app.get_graph().draw_mermaid_png()


# Storing the graph as png
with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)
print("Saved workflow_graph.png in current folder")


#Invoking the graph
response = app.invoke({"input": "Amit" }) 
print(response)

for res in app.stream({"input": "Amit"}):
    for key, value in res.items():
        print(f"Key is {key} , Value is {value}")


# Adding workflow with llm

def llm_call_with_graph(state: GraphState) -> GraphState:
    chat_model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    prompt = f"You are an AI Assistant and please provide brief answer: {state['input']}"
    response = chat_model.invoke([{"role": "user", "content": prompt}])
    state["input"] = response
    print(state["input"])
    return state

def token_counter(state: GraphState) -> GraphState:
    token_count = state["input"].response_metadata["token_usage"]["total_tokens"]
    print(f"Total tokens in the quote: {token_count}")
    content = f"Response: {state['input'].content}\nTotal Tokens Used: {token_count}"
    state["input"] = content
    return state

workflow_llm_graph = StateGraph(GraphState)
workflow_llm_graph.add_node("llm node", llm_call_with_graph)
workflow_llm_graph.add_node("token counter node", token_counter)
workflow_llm_graph.add_edge("llm node", "token counter node")
workflow_llm_graph.set_entry_point("llm node")
workflow_llm_graph.set_finish_point("token counter node")    

app_llm = workflow_llm_graph.compile()

bytes = app_llm.get_graph().draw_mermaid_png()

with open("llm_workflow_graph.png", "wb") as f:
    f.write(bytes)

print("Saved llm_workflow_graph.png in current folder")

response_llm = app_llm.invoke({"input": "Who won the 2020 Wimbledon in Males?" })
print(f"response llm is {response_llm}");