"""Image generation handlers."""

from telegram import Update, InputMediaPhoto
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db
from bot.ai.image_gen import image_generator
from bot.utils.keyboards import kb
from bot.utils.decorators import (
    handle_errors,
    check_banned,
    check_tokens,
    log_command
)


@handle_errors
@log_command
@check_banned
async def image_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command."""
    await update.message.reply_text(
        "🎨 <b>Генерация изображений</b>\n\n"
        "Выберите модель для генерации:",
        parse_mode='HTML',
        reply_markup=kb.image_generation_menu()
    )


@handle_errors
@check_banned
@check_tokens(min_tokens=2000)
async def generate_image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image generation request."""
    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Ошибка: отправьте /start")
        return

    # Check if we're awaiting image prompt
    if not context.user_data.get('awaiting_image_prompt'):
        await update.message.reply_text(
            "Используйте команду /image для генерации изображений"
        )
        return

    prompt = update.message.text
    image_model = context.user_data.get('image_model', 'dalle')
    size = context.user_data.get('image_size', '1024x1024')
    quality = context.user_data.get('image_quality', 'standard')

    # Calculate cost
    cost = image_generator.calculate_image_cost(
        model="dall-e-3" if image_model == "dalle" else "dall-e-2",
        size=size,
        quality=quality
    )

    if user.tokens_available < cost:
        await update.message.reply_text(
            f"⚠️ Недостаточно токенов!\n\n"
            f"Требуется: {cost:,}\n"
            f"Доступно: {user.tokens_available:,}\n\n"
            f"Пополните баланс: /balance"
        )
        return

    try:
        status_msg = await update.message.reply_text(
            "🎨 Генерирую изображение...\n"
            "Это может занять до 30 секунд."
        )

        # Generate image
        image_data = await image_generator.generate_and_download(
            prompt=prompt,
            model=image_model,
            size=size,
            quality=quality
        )

        # Send image
        await update.message.reply_photo(
            photo=image_data,
            caption=f"🎨 <b>Сгенерировано!</b>\n\n"
                    f"Промпт: {prompt}\n"
                    f"Модель: {image_model}\n"
                    f"Размер: {size}\n"
                    f"Стоимость: {cost:,} токенов",
            parse_mode='HTML'
        )

        await status_msg.delete()

        # Update database
        await db.update_user_tokens(user_id, cost)
        await db.update_statistics(
            user_id=user_id,
            images_generated=1,
            tokens_used=cost
        )

        # Update user stats
        user_result = await db.get_user(user_id)
        if user_result:
            await db.async_session().execute(
                "UPDATE users SET images_generated = images_generated + 1 WHERE id = :user_id",
                {"user_id": user_id}
            )

        logger.info(f"User {user_id} generated image: {cost} tokens")

        # Clear state
        context.user_data['awaiting_image_prompt'] = False

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        await update.message.reply_text(
            "❌ Ошибка при генерации изображения.\n"
            "Попробуйте изменить промпт или попробуйте позже."
        )


@handle_errors
async def image_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle image generation callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data

    if data == "image_dalle3":
        context.user_data['image_model'] = "dalle"
        context.user_data['image_size'] = "1024x1024"
        context.user_data['image_quality'] = "standard"
        context.user_data['awaiting_image_prompt'] = True

        await query.edit_message_text(
            "🎨 <b>DALL-E 3</b>\n\n"
            "Отправьте описание изображения, которое хотите сгенерировать.\n\n"
            "Примеры:\n"
            "• Кот в костюме астронавта на Луне\n"
            "• Футуристический город на закате\n"
            "• Абстрактная картина в стиле Пикассо",
            parse_mode='HTML'
        )

    elif data == "image_dalle2":
        context.user_data['image_model'] = "dalle2"
        context.user_data['image_size'] = "1024x1024"
        context.user_data['awaiting_image_prompt'] = True

        await query.edit_message_text(
            "🖼 <b>DALL-E 2</b>\n\n"
            "Отправьте описание изображения:",
            parse_mode='HTML'
        )

    elif data == "image_settings":
        await query.edit_message_text(
            "⚙️ <b>Настройки генерации</b>\n\n"
            "Выберите параметры:",
            parse_mode='HTML',
            reply_markup=kb.image_size_selection()
        )

    elif data.startswith("size_"):
        size = data.replace("size_", "")
        context.user_data['image_size'] = size

        await query.edit_message_text(
            f"✅ Размер установлен: {size}\n\n"
            "Выберите качество:",
            reply_markup=kb.image_quality_selection()
        )

    elif data.startswith("quality_"):
        quality = data.replace("quality_", "")
        context.user_data['image_quality'] = quality

        await query.edit_message_text(
            f"✅ Настройки сохранены!\n\n"
            f"Размер: {context.user_data.get('image_size', '1024x1024')}\n"
            f"Качество: {quality}\n\n"
            "Вернитесь в меню для генерации:",
            reply_markup=kb.image_generation_menu()
        )

    elif data == "image_menu":
        await query.edit_message_text(
            "🎨 <b>Генерация изображений</b>\n\n"
            "Выберите модель:",
            parse_mode='HTML',
            reply_markup=kb.image_generation_menu()
        )
