from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import YouTubeSearchTool
import os
import getpass
from dotenv import load_dotenv


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