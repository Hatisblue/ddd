"""xAI Grok integration."""

from typing import List, Dict, Optional
import httpx
from loguru import logger

from bot.config import settings
from bot.database.models import AIModel


class GrokClient:
    """xAI Grok API client."""

    def __init__(self):
        """Initialize Grok client."""
        self.api_key = settings.grok_api_key
        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: AIModel = AIModel.GROK,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> tuple[str, int]:
        """
        Get chat completion from Grok.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: AI model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            model_name = model.value

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()

            response_text = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]

            logger.info(
                f"Grok {model_name} completion: {tokens_used} tokens used"
            )

            return response_text, tokens_used

        except httpx.HTTPError as e:
            logger.error(f"Grok HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            raise

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: AIModel = AIModel.GROK,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Stream chat completion from Grok.

        Args:
            messages: List of message dictionaries
            model: AI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks from the streaming response
        """
        try:
            model_name = model.value

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix

                            if data == "[DONE]":
                                break

                            try:
                                import json
                                chunk = json.loads(data)
                                if chunk["choices"][0].get("delta", {}).get("content"):
                                    yield chunk["choices"][0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Grok streaming error: {e}")
            raise

    def _estimate_tokens(
        self,
        messages: List[Dict[str, str]],
        response: str
    ) -> int:
        """Estimate token count (rough approximation)."""
        total_text = " ".join([msg["content"] for msg in messages]) + response
        # Rough estimate: ~4 characters per token
        return len(total_text) // 4


# Global Grok client instance
grok_client = GrokClient()
