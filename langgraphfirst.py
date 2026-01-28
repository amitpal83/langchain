from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import os
import getpass
from dotenv import load_dotenv


load_dotenv()
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

print(os.getenv("TAVILY_API_KEY"))

#initialize Tavily search tool
tool = TavilySearch(
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

# Tool with a query
# res= tool.invoke({"query": "What won  the last wimbledon"})


# Tool with Agent and LLM
model = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
agent = create_agent(model, [tool])

user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()