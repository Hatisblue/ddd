"""Chat with AI handlers."""

from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db, AIModel
from bot.ai.gpt import gpt_client
from bot.ai.gemini import gemini_client
from bot.ai.grok import grok_client
from bot.utils.decorators import (
    handle_errors,
    check_banned,
    check_tokens,
    typing_action,
    log_command
)
from bot.utils.helpers import prepare_messages_for_ai, get_system_prompt
from bot.config import settings


@handle_errors
@check_banned
@check_tokens(min_tokens=100)
@typing_action
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages for AI chat."""
    user_id = update.effective_user.id
    message_text = update.message.text

    # Check if awaiting promo code
    if context.user_data.get('awaiting_promo'):
        await handle_promo_code(update, context, message_text)
        return

    # Get user and their settings
    user = await db.get_user(user_id)
    if not user:
        await update.message.reply_text("❌ Ошибка: отправьте /start")
        return

    # Get chat history
    history = await db.get_user_history(user_id, limit=settings.max_context_length)

    # Prepare messages for AI
    system_prompt = get_system_prompt(context.user_data.get('chat_mode', 'normal'))
    messages = prepare_messages_for_ai(history, message_text, system_prompt)

    try:
        # Select AI client based on user's model
        model = user.current_model
        response_text = ""
        tokens_used = 0

        if "gpt" in model.value or "o1" in model.value:
            response_text, tokens_used = await gpt_client.chat_completion(
                messages=messages,
                model=model
            )
        elif "gemini" in model.value:
            response_text, tokens_used = await gemini_client.chat_completion(
                messages=messages,
                model=model
            )
        elif "grok" in model.value:
            response_text, tokens_used = await grok_client.chat_completion(
                messages=messages,
                model=model
            )
        else:
            await update.message.reply_text("❌ Неизвестная модель")
            return

        # Save messages to database
        await db.add_message(
            user_id=user_id,
            role="user",
            content=message_text,
            model=model,
            tokens_used=0
        )

        await db.add_message(
            user_id=user_id,
            role="assistant",
            content=response_text,
            model=model,
            tokens_used=tokens_used
        )

        # Update user tokens
        await db.update_user_tokens(user_id, tokens_used)

        # Update statistics
        await db.update_statistics(
            user_id=user_id,
            messages_sent=1,
            tokens_used=tokens_used,
            model=model
        )

        # Send response
        await update.message.reply_text(response_text)

        logger.info(
            f"User {user_id} chat: {len(message_text)} chars in, "
            f"{len(response_text)} chars out, {tokens_used} tokens"
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при обработке запроса.\n"
            "Попробуйте еще раз или выберите другую модель: /models"
        )


async def handle_promo_code(update: Update, context: ContextTypes.DEFAULT_TYPE, code: str):
    """Handle promo code activation."""
    user_id = update.effective_user.id

    success, message = await db.use_promo_code(user_id, code)

    if success:
        await update.message.reply_text(f"✅ {message}")
    else:
        await update.message.reply_text(f"❌ {message}")

    # Clear awaiting state
    context.user_data['awaiting_promo'] = False


@handle_errors
@log_command
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages."""
    if not settings.enable_voice_messages:
        await update.message.reply_text("🎤 Голосовые сообщения отключены")
        return

    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user or user.tokens_available < 500:
        await update.message.reply_text(
            "⚠️ Недостаточно токенов для обработки голосового сообщения"
        )
        return

    try:
        # Download voice file
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        file_path = f"/tmp/voice_{user_id}_{voice.file_id}.ogg"
        await file.download_to_drive(file_path)

        # Transcribe using Whisper
        transcribed_text = await gpt_client.transcribe_audio(file_path)

        # Process as regular message
        context.user_data['transcribed_text'] = transcribed_text
        update.message.text = transcribed_text

        await update.message.reply_text(f"🎤 Распознано: {transcribed_text}\n\n⏳ Обрабатываю...")
        await message_handler(update, context)

    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        await update.message.reply_text("❌ Ошибка обработки голосового сообщения")
