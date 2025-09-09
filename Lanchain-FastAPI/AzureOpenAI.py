from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # load from .env
app = FastAPI()
# Use your Azure deployment name (not model name!)
llm = AzureChatOpenAI(
    deployment_name="gpt35-turbo",  # your deployment name
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    try:
        response = llm.invoke(request.prompt)
        return {"response": response.content}
    except Exception as e:
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="Rate limit exceeded on Azure OpenAI")
        raise HTTPException(status_code=500, detail=str(e))