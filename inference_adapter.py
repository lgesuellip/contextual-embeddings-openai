from typing import List, Dict
import dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import os
from langsmith.wrappers import wrap_openai
import logging

logger = logging.getLogger(__name__)

def setup_env(dotenv_path: str = ".env"):
    """Load environment variables from .env file."""

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f"File not found: {dotenv_path}")

    try:
        dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=True, override=False)
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        raise e
    
class OpenAIClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            setup_env("./.env")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            cls._instance = wrap_openai(AsyncOpenAI(api_key=api_key))
        return cls._instance

class InferenceAdapter:
    def __init__(self):
        self.client = OpenAIClientSingleton.get_instance()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def predict_with_parse_async(self, model_args: Dict, response_format, messages: List[Dict]):
        
        response = await self.client.beta.chat.completions.parse(
                        **model_args,
                        messages=messages,
                        response_format=response_format,
                    )

        return response.choices[0].message.parsed, response.usage
