"""Image generation module."""

from typing import Optional, List
from io import BytesIO
import httpx
from PIL import Image
from loguru import logger

from bot.config import settings
from bot.ai.gpt import gpt_client


class ImageGenerator:
    """Image generation client supporting multiple models."""

    def __init__(self):
        """Initialize image generator."""
        self.dalle_model = settings.dalle_model

    async def generate_dalle(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1
    ) -> List[str]:
        """
        Generate images using DALL-E.

        Args:
            prompt: Image description
            size: Image size (1024x1024, 1024x1792, 1792x1024)
            quality: standard or hd
            n: Number of images

        Returns:
            List of image URLs
        """
        try:
            image_urls = await gpt_client.create_image(
                prompt=prompt,
                model=self.dalle_model,
                size=size,
                quality=quality,
                n=n
            )

            logger.info(f"DALL-E generated {len(image_urls)} images")
            return image_urls

        except Exception as e:
            logger.error(f"DALL-E generation error: {e}")
            raise

    async def generate_stable_diffusion(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30
    ) -> List[str]:
        """
        Generate images using Stable Diffusion API.

        Args:
            prompt: Image description
            negative_prompt: What to avoid in image
            width: Image width
            height: Image height
            steps: Number of inference steps

        Returns:
            List of image URLs or base64 strings
        """
        try:
            # This is a placeholder for Stable Diffusion API integration
            # You would need to use a specific SD API service like:
            # - Stability AI API
            # - DreamStudio
            # - RunPod
            # - Your own hosted SD instance

            logger.warning("Stable Diffusion API not configured")
            raise NotImplementedError(
                "Stable Diffusion API requires configuration. "
                "Please set up SD API endpoint."
            )

        except Exception as e:
            logger.error(f"Stable Diffusion generation error: {e}")
            raise

    async def download_image(self, url: str) -> BytesIO:
        """
        Download image from URL.

        Args:
            url: Image URL

        Returns:
            BytesIO object with image data
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                image_data = BytesIO(response.content)
                logger.info(f"Downloaded image from {url}")

                return image_data

        except Exception as e:
            logger.error(f"Image download error: {e}")
            raise

    async def generate_and_download(
        self,
        prompt: str,
        model: str = "dalle",
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> BytesIO:
        """
        Generate image and download it.

        Args:
            prompt: Image description
            model: Model to use (dalle, sd)
            size: Image size
            quality: Image quality

        Returns:
            BytesIO object with image data
        """
        try:
            if model == "dalle":
                urls = await self.generate_dalle(
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1
                )
                return await self.download_image(urls[0])

            elif model == "sd" or model == "stable-diffusion":
                # Stable Diffusion would be implemented here
                raise NotImplementedError("Stable Diffusion not configured")

            else:
                raise ValueError(f"Unknown image generation model: {model}")

        except Exception as e:
            logger.error(f"Image generation and download error: {e}")
            raise

    def resize_image(
        self,
        image_data: BytesIO,
        max_size: tuple = (2048, 2048)
    ) -> BytesIO:
        """
        Resize image if it's too large.

        Args:
            image_data: Image data
            max_size: Maximum dimensions (width, height)

        Returns:
            Resized image data
        """
        try:
            image = Image.open(image_data)

            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {image.size}")

            output = BytesIO()
            image.save(output, format=image.format or 'PNG')
            output.seek(0)

            return output

        except Exception as e:
            logger.error(f"Image resize error: {e}")
            raise

    async def generate_variations(
        self,
        image_path: str,
        n: int = 1,
        size: str = "1024x1024"
    ) -> List[str]:
        """
        Generate variations of an existing image using DALL-E.

        Args:
            image_path: Path to source image
            n: Number of variations
            size: Output size

        Returns:
            List of image URLs
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)

            with open(image_path, "rb") as image_file:
                response = await client.images.create_variation(
                    image=image_file,
                    n=n,
                    size=size
                )

            image_urls = [img.url for img in response.data]
            logger.info(f"Generated {len(image_urls)} image variations")

            return image_urls

        except Exception as e:
            logger.error(f"Image variation error: {e}")
            raise

    def calculate_image_cost(
        self,
        model: str,
        size: str,
        quality: str = "standard"
    ) -> int:
        """
        Calculate token cost for image generation.

        Args:
            model: Image model
            size: Image size
            quality: Image quality

        Returns:
            Token cost
        """
        # Approximate token costs for image generation
        costs = {
            "dall-e-3": {
                "1024x1024": {"standard": 4000, "hd": 8000},
                "1024x1792": {"standard": 8000, "hd": 12000},
                "1792x1024": {"standard": 8000, "hd": 12000},
            },
            "dall-e-2": {
                "1024x1024": {"standard": 2000},
                "512x512": {"standard": 1500},
                "256x256": {"standard": 1000},
            }
        }

        try:
            return costs.get(model, {}).get(size, {}).get(quality, 4000)
        except:
            return 4000  # Default cost


# Global image generator instance
image_generator = ImageGenerator()
