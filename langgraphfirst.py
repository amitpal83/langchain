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


def function1(input):
    """This is function 1"""
    
    print(f"Function 1 executed {input}")

def function2(input):
    """This is function 2"""
    print(f"Function 2 executed {input}")

function1('amit')
function2('kumar')

# workflow_graph = Graph()

# workflow_graph.add_node()
# workflow_graph.add_node(Node(id="function2", func=function2, description="This is function 2"))
# workflow_graph.add_edge(Edge(from_node="function1", to_node="function2"))
# workflow_graph.set_entry_point("function1")
# workflow_graph.set_exit_point("function2")
# app= workflow_graph.compile()

# display(Image(app.get_graph().draw_mermaid_png()))


class GraphState(TypedDict):
    input: str


firstgraph = StateGraph(GraphState)


def greeting_node(state: GraphState) -> GraphState:
    print(state["input"])
    state["input"] = f"Hello, {state['input']} welcome to the LangGraph!"
    return state


firstgraph.add_node("greeting node", greeting_node)

def farewell_node(state: GraphState) -> GraphState:
    print(state["input"])
    state["input"] = f"Thanks for exploring..Goodbye! {state['input']} Have a great day!"
    print(state["input"])
    return state

firstgraph.add_node("farewell node", farewell_node)

firstgraph.add_edge("greeting node", "farewell node")

firstgraph.set_entry_point("greeting node")
firstgraph.set_finish_point("farewell node")

app= firstgraph.compile()

png_bytes = app.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved workflow_graph.png in current folder")

app.invoke({"input": "Amit" }) 