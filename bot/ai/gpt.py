"""OpenAI GPT integration."""

from typing import List, Dict, Optional, AsyncGenerator
from openai import AsyncOpenAI
from loguru import logger

from bot.config import settings
from bot.database.models import AIModel


class GPTClient:
    """OpenAI GPT API client."""

    def __init__(self):
        """Initialize GPT client."""
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            organization=settings.openai_org_id if settings.openai_org_id else None
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: AIModel = AIModel.GPT4O_MINI,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> tuple[str, int]:
        """
        Get chat completion from GPT.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: AI model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            model_name = model.value

            if stream:
                response_text = ""
                async for chunk in self._stream_completion(
                    messages, model_name, temperature, max_tokens
                ):
                    response_text += chunk

                # Estimate tokens (will be updated with actual usage later)
                tokens_used = self._estimate_tokens(messages, response_text)
                return response_text, tokens_used
            else:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

                logger.info(
                    f"GPT {model_name} completion: {tokens_used} tokens used"
                )

                return response_text, tokens_used

        except Exception as e:
            logger.error(f"GPT API error: {e}")
            raise

    async def _stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> AsyncGenerator[str, None]:
        """Stream GPT completion."""
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"GPT streaming error: {e}")
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

    async def create_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1
    ) -> List[str]:
        """
        Generate images using DALL-E.

        Args:
            prompt: Image description
            model: dall-e-2 or dall-e-3
            size: Image size
            quality: standard or hd (dall-e-3 only)
            n: Number of images

        Returns:
            List of image URLs
        """
        try:
            response = await self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality if model == "dall-e-3" else None,
                n=n
            )

            image_urls = [img.url for img in response.data]
            logger.info(f"Generated {len(image_urls)} images with {model}")

            return image_urls

        except Exception as e:
            logger.error(f"DALL-E API error: {e}")
            raise

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio using Whisper.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ru"
                )

            logger.info(f"Transcribed audio: {len(transcript.text)} characters")
            return transcript.text

        except Exception as e:
            logger.error(f"Whisper API error: {e}")
            raise

    async def check_moderation(self, text: str) -> tuple[bool, Dict]:
        """
        Check content moderation.

        Args:
            text: Text to check

        Returns:
            Tuple of (is_flagged, categories)
        """
        try:
            response = await self.client.moderations.create(input=text)
            result = response.results[0]

            return result.flagged, result.categories.model_dump()

        except Exception as e:
            logger.error(f"Moderation API error: {e}")
            return False, {}


# Global GPT client instance
gpt_client = GPTClient()
