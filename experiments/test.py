from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load your API key from .env file
load_dotenv()

# Create a connection to GPT
llm = ChatOpenAI(model="gpt-4o-mini")

# Send a simple message
response = llm.invoke("What is the Euro NCAP safety rating system?")

# Print the response
print(response.content)