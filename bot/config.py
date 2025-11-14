"""Configuration module for the AI Telegram Bot."""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Bot configuration settings."""

    # Telegram
    bot_token: str = Field(..., env='BOT_TOKEN')
    admin_ids: List[int] = Field(default_factory=list, env='ADMIN_IDS')

    # Database
    database_url: str = Field(
        default='sqlite+aiosqlite:///./data/bot.db',
        env='DATABASE_URL'
    )

    # Redis
    redis_url: str = Field(default='redis://localhost:6379/0', env='REDIS_URL')

    # AI APIs
    openai_api_key: str = Field(default='', env='OPENAI_API_KEY')
    openai_org_id: str = Field(default='', env='OPENAI_ORG_ID')
    gemini_api_key: str = Field(default='', env='GEMINI_API_KEY')
    grok_api_key: str = Field(default='', env='GROK_API_KEY')

    # Payment
    yookassa_shop_id: str = Field(default='', env='YOOKASSA_SHOP_ID')
    yookassa_secret_key: str = Field(default='', env='YOOKASSA_SECRET_KEY')

    # Bot Settings
    default_token_limit: int = Field(default=100000, env='DEFAULT_TOKEN_LIMIT')
    free_tokens_on_register: int = Field(default=10000, env='FREE_TOKENS_ON_REGISTER')
    enable_voice_messages: bool = Field(default=True, env='ENABLE_VOICE_MESSAGES')
    enable_referral_system: bool = Field(default=True, env='ENABLE_REFERRAL_SYSTEM')
    referral_bonus: int = Field(default=5000, env='REFERRAL_BONUS')

    # Image Generation
    dalle_model: str = Field(default='dall-e-3', env='DALLE_MODEL')
    stable_diffusion_api_key: str = Field(default='', env='STABLE_DIFFUSION_API_KEY')

    # Logging
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    sentry_dsn: str = Field(default='', env='SENTRY_DSN')

    # Limits
    max_context_length: int = Field(default=10, env='MAX_CONTEXT_LENGTH')
    rate_limit_messages: int = Field(default=20, env='RATE_LIMIT_MESSAGES')
    rate_limit_period: int = Field(default=60, env='RATE_LIMIT_PERIOD')

    # Features
    enable_gpt4: bool = Field(default=True, env='ENABLE_GPT4')
    enable_gpt4o: bool = Field(default=True, env='ENABLE_GPT4O')
    enable_gemini_pro: bool = Field(default=True, env='ENABLE_GEMINI_PRO')
    enable_grok: bool = Field(default=True, env='ENABLE_GROK')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse admin IDs if provided as string
        if isinstance(self.admin_ids, str):
            self.admin_ids = [int(x.strip()) for x in self.admin_ids.split(',') if x.strip()]


# Global settings instance
settings = Settings()
