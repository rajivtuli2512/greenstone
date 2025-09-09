from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load from .env

app = FastAPI()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 temperature=0,
                 openai_api_key=os.getenv("OPENAI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat",
          responses={
              200: {"description": "Chat response"},
              400: {"description": "Invalid input"},
              429: {"description": "You exceeded your current quota, please check your plan and billing details"},
          })
async def chat(request: PromptRequest):
    try:
        response = llm.invoke(request.prompt)
        return {"response": response.content}
    except Exception as e:
        # If OpenAI returns a 429 or similar
        if "429" in str(e):
            raise HTTPException(status_code=429, detail="You exceeded your current quota, please check your plan and billing details")
        raise HTTPException(status_code=500, detail=str(e))
 