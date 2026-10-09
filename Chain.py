
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Disable optional LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "false"

# Check whether the API key exists
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )

# 1. Prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

# 2. Model
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)

# 3. Output parser
output_parser = StrOutputParser()

print("All three pieces are ready!")

# Create the chain
chain = prompt | model | output_parser

print("Chain is ready!")

answer = chain.invoke({"topic": "Python"})
print(answer)

# Create another chain using the same model
tip_prompt = ChatPromptTemplate.from_template(
    "Give one short, friendly tip for a beginner learning {skill}."
)

tip_chain = tip_prompt | model | output_parser

print(tip_chain.invoke({"skill": "Python"}))
