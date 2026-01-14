import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq 
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
     

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")    


# Initialize LLMs
llm= ChatOpenAI(model_name="gpt-4o", temperature=0)

groq_llm= ChatGroq(model_name="qwen/qwen3-32b", temperature=0)
      
# Using Prompt Templates 
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template=".Explain the concept of {topic} in simple terms.",
)
message_format= prompt_template.format(topic="Microservices")
response4 = llm.invoke(message_format)
print(response4.content);

# Using Chat Prompt Templates
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful DevOps expert."),
    ("human", "Explain {tool} in simple terms.")
])
messages = chat_prompt.format_messages(tool="Helm")
response2 = llm.invoke(messages)
print(response2.content)


# Chaining with  Chat Prompt Templates and and String Output Parser 

prompt = ChatPromptTemplate.from_messages([

    ("system", "You are a helpful AI assistant that translates languages and provides information. Translate the input text from {input_language} to {output_language}. "),
    ("human", "Translate the following: {input}"),

])

parser= StrOutputParser()
chain = prompt | groq_llm | parser
response = chain.invoke({"input_language":"English", "output_language":"German", "input":"How are you doing today?"})
print(response)

#Chaining with Prompt Templates and JSON Output Parser
outpur_parser= JsonOutputParser()

prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Answer the user in the format {format_instructions}.Explain the concept of {topic} in simple terms.",
    partial_variables={"format_instructions": outpur_parser.get_format_instructions()} 
)

chain2 = prompt_template | llm | outpur_parser
response3 = chain2.invoke({"topic":"Kubernetes"})
print(response3)