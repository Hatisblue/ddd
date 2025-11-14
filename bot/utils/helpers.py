"""Helper functions for the bot."""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import csv
from io import StringIO, BytesIO
from loguru import logger

from bot.database.models import User, Message, UserStatistics


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator."""
    return f"{tokens:,}".replace(',', ' ')


def format_datetime(dt: datetime) -> str:
    """Format datetime to readable string."""
    return dt.strftime("%d.%m.%Y %H:%M")


def calculate_cost(tokens: int) -> float:
    """
    Calculate approximate cost in rubles for token usage.

    Args:
        tokens: Number of tokens

    Returns:
        Cost in rubles
    """
    # Approximate pricing (can be adjusted)
    return tokens * 0.002  # 0.002₽ per token


def format_user_info(user: User) -> str:
    """Format user information for display."""
    info = [
        f"👤 ID: {user.id}",
        f"👤 Имя: {user.first_name or 'N/A'}",
    ]

    if user.username:
        info.append(f"📱 @{user.username}")

    info.extend([
        f"",
        f"💎 Роль: {user.role.value}",
        f"🪙 Токены: {format_tokens(user.tokens_available)} / {format_tokens(user.tokens_limit)}",
        f"📊 Использовано: {format_tokens(user.tokens_used)}",
        f"",
        f"💬 Сообщений: {user.messages_count}",
        f"🎨 Изображений: {user.images_generated}",
    ])

    if user.is_premium:
        premium_until = format_datetime(user.premium_until) if user.premium_until else "∞"
        info.append(f"⭐️ Премиум до: {premium_until}")

    info.extend([
        f"",
        f"📅 Регистрация: {format_datetime(user.created_at)}",
        f"🕐 Последняя активность: {format_datetime(user.last_active)}",
    ])

    return "\n".join(info)


def format_statistics(stats: List[UserStatistics]) -> str:
    """Format user statistics for display."""
    if not stats:
        return "Нет данных за выбранный период"

    total_messages = sum(s.messages_sent for s in stats)
    total_tokens = sum(s.tokens_used for s in stats)
    total_images = sum(s.images_generated for s in stats)

    total_gpt = sum(s.gpt_requests for s in stats)
    total_gemini = sum(s.gemini_requests for s in stats)
    total_grok = sum(s.grok_requests for s in stats)

    info = [
        f"📊 Статистика за период",
        f"",
        f"💬 Сообщений: {total_messages}",
        f"🪙 Токенов: {format_tokens(total_tokens)}",
        f"🎨 Изображений: {total_images}",
        f"",
        f"🤖 Модели:",
        f"  • GPT: {total_gpt}",
        f"  • Gemini: {total_gemini}",
        f"  • Grok: {total_grok}",
    ]

    return "\n".join(info)


def export_chat_history(messages: List[Message]) -> BytesIO:
    """
    Export chat history to JSON file.

    Args:
        messages: List of messages

    Returns:
        BytesIO with JSON data
    """
    chat_data = []

    for msg in messages:
        chat_data.append({
            "timestamp": msg.created_at.isoformat(),
            "role": msg.role,
            "content": msg.content,
            "model": msg.model.value,
            "tokens": msg.tokens_used,
        })

    json_data = json.dumps(chat_data, ensure_ascii=False, indent=2)
    buffer = BytesIO(json_data.encode('utf-8'))
    buffer.name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    return buffer


def export_chat_to_csv(messages: List[Message]) -> BytesIO:
    """
    Export chat history to CSV file.

    Args:
        messages: List of messages

    Returns:
        BytesIO with CSV data
    """
    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['Timestamp', 'Role', 'Content', 'Model', 'Tokens'])

    # Data
    for msg in messages:
        writer.writerow([
            msg.created_at.isoformat(),
            msg.role,
            msg.content,
            msg.model.value,
            msg.tokens_used,
        ])

    # Convert to BytesIO
    buffer = BytesIO(output.getvalue().encode('utf-8'))
    buffer.name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return buffer


def generate_referral_link(bot_username: str, referral_code: str) -> str:
    """
    Generate referral link for user.

    Args:
        bot_username: Bot's username
        referral_code: User's referral code

    Returns:
        Referral link
    """
    return f"https://t.me/{bot_username}?start=ref_{referral_code}"


def parse_referral_code(start_param: Optional[str]) -> Optional[str]:
    """
    Parse referral code from start parameter.

    Args:
        start_param: Start parameter from deep link

    Returns:
        Referral code or None
    """
    if not start_param:
        return None

    if start_param.startswith("ref_"):
        return start_param[4:]

    return None


def get_model_display_name(model_value: str) -> str:
    """Get user-friendly model name."""
    model_names = {
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "gpt-4": "GPT-4",
        "gpt-4-turbo": "GPT-4 Turbo",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o mini",
        "o1-preview": "o1-preview",
        "o1-mini": "o1-mini",
        "gemini-pro": "Gemini Pro",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
        "gemini-1.5-flash": "Gemini 1.5 Flash",
        "grok-beta": "Grok",
        "grok-vision-beta": "Grok Vision",
    }
    return model_names.get(model_value, model_value)


def get_model_emoji(model_value: str) -> str:
    """Get emoji for model."""
    if "gpt" in model_value or "o1" in model_value:
        return "🤖"
    elif "gemini" in model_value:
        return "✨"
    elif "grok" in model_value:
        return "🚀"
    else:
        return "💬"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_time_delta(td: timedelta) -> str:
    """Format timedelta to human-readable string."""
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}д")
    if hours > 0:
        parts.append(f"{hours}ч")
    if minutes > 0:
        parts.append(f"{minutes}м")
    if seconds > 0 and not parts:
        parts.append(f"{seconds}с")

    return " ".join(parts) if parts else "0с"


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time in minutes.

    Args:
        text: Text to read
        words_per_minute: Average reading speed

    Returns:
        Estimated time in minutes
    """
    words = len(text.split())
    return max(1, words // words_per_minute)


def prepare_messages_for_ai(
    history: List[Message],
    new_message: str,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Prepare messages in format for AI APIs.

    Args:
        history: Previous messages
        new_message: New user message
        system_prompt: Optional system prompt

    Returns:
        List of message dictionaries
    """
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add history
    for msg in history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add new message
    messages.append({"role": "user", "content": new_message})

    return messages


def get_system_prompt(mode: str = "normal") -> str:
    """Get system prompt based on chat mode."""
    prompts = {
        "normal": "Ты полезный AI-ассистент. Отвечай кратко и по существу. Используй русский язык.",
        "creative": "Ты креативный AI-ассистент. Будь оригинальным и творческим в ответах. "
                   "Используй метафоры и яркие образы.",
        "professional": "Ты деловой AI-ассистент. Давай профессиональные, структурированные ответы. "
                        "Будь формальным и точным.",
        "coding": "Ты AI-ассистент программиста. Помогай с кодом, давай примеры и объяснения. "
                  "Форматируй код правильно с подсветкой синтаксиса.",
    }
    return prompts.get(mode, prompts["normal"])


def sanitize_filename(filename: str) -> str:
    """Remove potentially dangerous characters from filename."""
    import re
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:max_length - len(ext) - 1] + '.' + ext if ext else name[:max_length]
    return filename


def parse_package_data(callback_data: str) -> tuple[int, int]:
    """
    Parse token package data from callback.

    Args:
        callback_data: Callback data (format: pkg_tokens_price)

    Returns:
        Tuple of (tokens, price)
    """
    try:
        parts = callback_data.split('_')
        tokens = int(parts[1])
        price = int(parts[2])
        return tokens, price
    except (IndexError, ValueError):
        logger.error(f"Failed to parse package data: {callback_data}")
        return 0, 0
