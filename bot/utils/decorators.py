"""Decorators for bot handlers."""

from functools import wraps
from typing import Callable
from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.config import settings
from bot.database import db, UserRole


def admin_only(func: Callable) -> Callable:
    """Decorator to restrict access to admin users only."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id

        if user_id not in settings.admin_ids:
            await update.message.reply_text(
                "⛔️ У вас нет доступа к этой команде."
            )
            logger.warning(f"Unauthorized admin access attempt by user {user_id}")
            return

        return await func(update, context, *args, **kwargs)

    return wrapper


def check_tokens(min_tokens: int = 100):
    """Decorator to check if user has enough tokens."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user_id = update.effective_user.id
            user = await db.get_user(user_id)

            if not user:
                await update.message.reply_text(
                    "❌ Ошибка: пользователь не найден. Отправьте /start"
                )
                return

            if user.tokens_available < min_tokens:
                await update.message.reply_text(
                    f"⚠️ Недостаточно токенов!\n\n"
                    f"Доступно: {user.tokens_available}\n"
                    f"Требуется: {min_tokens}\n\n"
                    f"Пополните баланс: /balance"
                )
                logger.info(f"User {user_id} has insufficient tokens: {user.tokens_available}")
                return

            return await func(update, context, *args, **kwargs)

        return wrapper

    return decorator


def check_banned(func: Callable) -> Callable:
    """Decorator to check if user is banned."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        user = await db.get_user(user_id)

        if user and user.role == UserRole.BANNED:
            await update.message.reply_text(
                "🚫 Вы заблокированы и не можете использовать бота.\n"
                "Для разблокировки обратитесь к администратору."
            )
            logger.warning(f"Banned user {user_id} tried to access bot")
            return

        return await func(update, context, *args, **kwargs)

    return wrapper


def log_command(func: Callable) -> Callable:
    """Decorator to log command execution."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        command = update.message.text if update.message else "callback"

        logger.info(
            f"User {user.id} ({user.username or user.first_name}) "
            f"executed: {command}"
        )

        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing {command}: {e}")
            raise

    return wrapper


def rate_limit(calls: int = 5, period: int = 60):
    """
    Decorator to rate limit user requests.

    Args:
        calls: Number of allowed calls
        period: Time period in seconds
    """
    user_calls = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            import time

            user_id = update.effective_user.id
            current_time = time.time()

            # Initialize user call history
            if user_id not in user_calls:
                user_calls[user_id] = []

            # Remove old calls outside the time window
            user_calls[user_id] = [
                call_time for call_time in user_calls[user_id]
                if current_time - call_time < period
            ]

            # Check rate limit
            if len(user_calls[user_id]) >= calls:
                time_left = int(period - (current_time - user_calls[user_id][0]))
                await update.message.reply_text(
                    f"⏱ Превышен лимит запросов!\n"
                    f"Попробуйте снова через {time_left} секунд."
                )
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return

            # Add current call
            user_calls[user_id].append(current_time)

            return await func(update, context, *args, **kwargs)

        return wrapper

    return decorator


def typing_action(func: Callable) -> Callable:
    """Decorator to show typing action while processing."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        return await func(update, context, *args, **kwargs)

    return wrapper


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle errors gracefully."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

            error_message = (
                "❌ Произошла ошибка при обработке запроса.\n"
                "Пожалуйста, попробуйте позже или обратитесь к администратору."
            )

            if update.message:
                await update.message.reply_text(error_message)
            elif update.callback_query:
                await update.callback_query.answer(
                    "Произошла ошибка",
                    show_alert=True
                )

    return wrapper


def premium_only(func: Callable) -> Callable:
    """Decorator to restrict access to premium users only."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        user = await db.get_user(user_id)

        if not user or not user.is_premium:
            await update.message.reply_text(
                "⭐️ Эта функция доступна только для премиум-пользователей.\n\n"
                "Оформите подписку: /balance"
            )
            logger.info(f"Non-premium user {user_id} tried to access premium feature")
            return

        return await func(update, context, *args, **kwargs)

    return wrapper


def track_usage(func: Callable) -> Callable:
    """Decorator to track feature usage statistics."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id

        # Track usage in database
        await db.update_statistics(
            user_id=user_id,
            messages_sent=1
        )

        return await func(update, context, *args, **kwargs)

    return wrapper
