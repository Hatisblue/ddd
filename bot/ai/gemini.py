"""Google Gemini integration."""

from typing import List, Dict, Optional
import google.generativeai as genai
from loguru import logger

from bot.config import settings
from bot.database.models import AIModel


class GeminiClient:
    """Google Gemini API client."""

    def __init__(self):
        """Initialize Gemini client."""
        genai.configure(api_key=settings.gemini_api_key)

    def _get_model(self, model: AIModel):
        """Get Gemini model instance."""
        model_name = model.value
        return genai.GenerativeModel(model_name)

    def _convert_messages_to_gemini_format(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[List[Dict], str]:
        """
        Convert OpenAI-style messages to Gemini format.

        Returns:
            Tuple of (history, user_message)
        """
        history = []
        system_prompt = ""

        for msg in messages[:-1]:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_prompt = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        # Last message is the current user message
        user_message = messages[-1]["content"] if messages else ""

        # Prepend system prompt to first user message if exists
        if system_prompt and history:
            if history[0]["role"] == "user":
                history[0]["parts"][0] = f"{system_prompt}\n\n{history[0]['parts'][0]}"
            else:
                history.insert(0, {"role": "user", "parts": [system_prompt]})
                history.insert(1, {"role": "model", "parts": ["Понял, буду следовать инструкциям."]})

        return history, user_message

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: AIModel = AIModel.GEMINI_15_FLASH,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> tuple[str, int]:
        """
        Get chat completion from Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: AI model to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            gemini_model = self._get_model(model)
            history, user_message = self._convert_messages_to_gemini_format(messages)

            # Configure generation
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Start chat with history
            chat = gemini_model.start_chat(history=history)

            # Send message
            response = await chat.send_message_async(
                user_message,
                generation_config=generation_config
            )

            response_text = response.text

            # Estimate tokens (Gemini doesn't provide exact count in the same way)
            tokens_used = self._estimate_tokens(messages, response_text)

            logger.info(
                f"Gemini {model.value} completion: ~{tokens_used} tokens used"
            )

            return response_text, tokens_used

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
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

    async def generate_content(
        self,
        prompt: str,
        model: AIModel = AIModel.GEMINI_15_FLASH,
        temperature: float = 0.7
    ) -> tuple[str, int]:
        """
        Generate content from a simple prompt.

        Args:
            prompt: Input prompt
            model: AI model to use
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            gemini_model = self._get_model(model)

            generation_config = {
                "temperature": temperature,
            }

            response = await gemini_model.generate_content_async(
                prompt,
                generation_config=generation_config
            )

            response_text = response.text
            tokens_used = len(prompt + response_text) // 4

            logger.info(f"Gemini content generation: ~{tokens_used} tokens used")

            return response_text, tokens_used

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        model: AIModel = AIModel.GEMINI_15_PRO
    ) -> tuple[str, int]:
        """
        Analyze image with Gemini Vision.

        Args:
            image_path: Path to image file
            prompt: Question/instruction about the image
            model: AI model to use

        Returns:
            Tuple of (response_text, tokens_used)
        """
        try:
            from PIL import Image

            gemini_model = self._get_model(model)
            image = Image.open(image_path)

            response = await gemini_model.generate_content_async([prompt, image])

            response_text = response.text
            tokens_used = len(prompt + response_text) // 4

            logger.info(f"Gemini image analysis: ~{tokens_used} tokens used")

            return response_text, tokens_used

        except Exception as e:
            logger.error(f"Gemini image analysis error: {e}")
            raise


# Global Gemini client instance
gemini_client = GeminiClient()
