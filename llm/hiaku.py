from langchain_aws import ChatBedrockConverse
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Any
import asyncio
import json

load_dotenv()



bedrock_claude = ChatBedrockConverse(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",  
    region_name="us-west-2",
    temperature=0.7,
    max_tokens=60000, 
)

async def claude_haiku(system_prompt: str, user_prompt: str, user_context:str, pydantic_model: BaseModel):
    user_message = f"""
    {user_prompt}
    
    context:
    {user_context}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        structured_llm = bedrock_claude.with_structured_output(pydantic_model)
        response = await structured_llm.ainvoke(messages)
        result = response.model_dump()
        return result
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None