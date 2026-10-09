import os

print("LangSmith is connected!")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Check whether the API key exists
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )



prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# This run is automatically recorded in LangSmith.
print(chain.invoke({"topic": "LangSmith"}))

from langsmith import traceable

@traceable
def make_greeting(name):
    return f"Hello {name}, welcome to GenAI!"

# This call is now recorded in LangSmith as its own step.
print(make_greeting("Jai"))

from openai import OpenAI
from langsmith.wrappers import wrap_openai

# wrap_openai is the ONLY change needed for tracking.
client = wrap_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)

print(response.choices[0].message.content)