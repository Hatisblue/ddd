"""Enhanced chat handler with security and optimization."""

import time
from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db, AIModel
from bot.ai.gpt import gpt_client
from bot.ai.gemini import gemini_client
from bot.ai.grok import grok_client
from bot.utils.decorators import handle_errors, check_banned, typing_action
from bot.utils.security import check_user_security, security_monitor
from bot.utils.cache import smart_cache
from bot.utils.prompt_engineering import prompt_engineer, response_optimizer
from bot.utils.monitoring import performance_monitor, usage_analytics, cost_tracker
from bot.config import settings


@handle_errors
@check_banned
@typing_action
async def enhanced_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Enhanced message handler with security, caching, and optimization.

    Features:
    - Input validation and sanitization
    - Rate limiting with progressive penalties
    - Smart caching for similar prompts
    - Advanced prompt engineering
    - Performance monitoring
    - Cost tracking
    """
    start_time = time.time()
    user_id = update.effective_user.id
    message_text = update.message.text

    # Check if awaiting promo code
    if context.user_data.get('awaiting_promo'):
        from bot.handlers.chat import handle_promo_code
        await handle_promo_code(update, context, message_text)
        return

    try:
        # Security check
        allowed, error_message = check_user_security(
            user_id,
            message_text,
            'messages'
        )

        if not allowed:
            await update.message.reply_text(f"⚠️ {error_message}")

            # Record security violation
            performance_monitor.record_request(
                'chat',
                time.time() - start_time,
                success=False
            )
            return

        # Get user and check tokens
        user = await db.get_user(user_id)
        if not user:
            await update.message.reply_text("❌ Ошибка: отправьте /start")
            return

        if user.tokens_available < 100:
            await update.message.reply_text(
                f"⚠️ Недостаточно токенов!\n\n"
                f"Доступно: {user.tokens_available}\n"
                f"Пополните баланс: /balance"
            )
            return

        model = user.current_model
        chat_mode = context.user_data.get('chat_mode', 'normal')

        # Check cache for similar prompts (save API calls)
        cached_response = smart_cache.get_similar_responses(
            message_text,
            model.value
        )

        if cached_response and isinstance(cached_response, dict):
            # Cache hit - return cached response
            response_text = cached_response['response']
            tokens_used = cached_response['tokens']

            await update.message.reply_text(
                response_text + "\n\n💡 Кэшированный ответ"
            )

            logger.info(f"Cache hit for user {user_id}, saved {tokens_used} tokens")

            # Record metrics
            performance_monitor.record_request(
                'chat_cached',
                time.time() - start_time,
                success=True,
                tokens=0
            )

            return

        # Get chat history
        history = await db.get_user_history(user_id, limit=settings.max_context_length)

        # Build optimized system prompt
        user_context = {
            'name': user.first_name,
            'language': user.language,
        }

        model_type = 'gpt' if 'gpt' in model.value or 'o1' in model.value else \
                    'gemini' if 'gemini' in model.value else 'grok'

        system_prompt = prompt_engineer.get_system_prompt(
            model_type,
            chat_mode,
            user_context
        )

        # Detect intent and enhance prompt
        intent = prompt_engineer.detect_intent(message_text)
        enhanced_prompt = prompt_engineer.enhance_prompt(
            message_text,
            task_type=intent
        )

        # Build conversation context
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        for msg in history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        messages.append({"role": "user", "content": enhanced_prompt})

        # Optimize context if too large
        messages = prompt_engineer.build_conversation_context(
            messages,
            max_context_messages=settings.max_context_length
        )

        # Call AI API based on model
        response_text = ""
        tokens_used = 0

        if 'gpt' in model.value or 'o1' in model.value:
            response_text, tokens_used = await gpt_client.chat_completion(
                messages=messages,
                model=model,
                temperature=0.7 if chat_mode != 'professional' else 0.5
            )
        elif 'gemini' in model.value:
            response_text, tokens_used = await gemini_client.chat_completion(
                messages=messages,
                model=model,
                temperature=0.7 if chat_mode != 'professional' else 0.5
            )
        elif 'grok' in model.value:
            response_text, tokens_used = await grok_client.chat_completion(
                messages=messages,
                model=model,
                temperature=0.7 if chat_mode != 'professional' else 0.5
            )
        else:
            await update.message.reply_text("❌ Неизвестная модель")
            return

        # Format response
        formatted_response = response_optimizer.format_response(
            response_text,
            'code' if intent == 'code' else 'text'
        )

        # Add context hints if helpful
        formatted_response = response_optimizer.add_context_hints(
            formatted_response,
            tokens_used,
            model.value
        )

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

        # Cache the response
        smart_cache.cache_response(
            message_text,
            model.value,
            response_text,
            tokens_used
        )

        # Record monitoring metrics
        duration = time.time() - start_time
        performance_monitor.record_request(
            'chat',
            duration,
            success=True,
            tokens=tokens_used
        )

        # Record analytics
        usage_analytics.record_usage(
            user_id,
            model.value,
            tokens_used,
            success=True
        )

        # Track costs
        cost_tracker.record_cost(
            model.value,
            tokens=tokens_used
        )

        # Send response
        await update.message.reply_text(formatted_response)

        logger.info(
            f"User {user_id} chat success: "
            f"{len(message_text)} chars in, "
            f"{len(response_text)} chars out, "
            f"{tokens_used} tokens, "
            f"{duration:.2f}s"
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)

        # Record failure
        performance_monitor.record_request(
            'chat',
            time.time() - start_time,
            success=False
        )

        usage_analytics.record_usage(
            user_id,
            model.value if 'model' in locals() else 'unknown',
            0,
            success=False
        )

        await update.message.reply_text(
            "❌ Произошла ошибка при обработке запроса.\n"
            "Попробуйте еще раз или выберите другую модель: /models"
        )
