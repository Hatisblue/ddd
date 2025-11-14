"""Callback query handlers."""

from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db, AIModel
from bot.utils.helpers import format_statistics, get_model_display_name, export_chat_history


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main callback query handler."""
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = update.effective_user.id

    # Model selection
    if data.startswith("model_"):
        model_value = data.replace("model_", "")
        try:
            model = AIModel(model_value)
            await db.set_user_model(user_id, model)

            await query.edit_message_text(
                f"✅ Модель изменена на: <b>{get_model_display_name(model_value)}</b>\n\n"
                f"Теперь можете начать общение!",
                parse_mode='HTML'
            )
            logger.info(f"User {user_id} switched to model {model_value}")
        except ValueError:
            await query.edit_message_text("❌ Неизвестная модель")

    # Chat modes
    elif data.startswith("mode_"):
        mode = data.replace("mode_", "")
        context.user_data['chat_mode'] = mode

        mode_names = {
            "normal": "Обычный",
            "creative": "Креативный",
            "professional": "Деловой",
            "coding": "Кодирование"
        }

        await query.edit_message_text(
            f"✅ Режим изменен на: <b>{mode_names.get(mode, mode)}</b>",
            parse_mode='HTML'
        )

    # Statistics
    elif data.startswith("stats_"):
        days = data.replace("stats_", "")

        if days == "all":
            days = 365
        else:
            days = int(days)

        stats = await db.get_user_statistics(user_id, days=days)
        stats_text = format_statistics(stats)

        await query.edit_message_text(
            stats_text,
            parse_mode='HTML'
        )

    # Settings
    elif data == "setting_clear_history":
        await db.clear_user_history(user_id)
        await query.edit_message_text(
            "✅ История диалога очищена!"
        )

    elif data == "setting_export_chat":
        history = await db.get_user_history(user_id, limit=1000)

        if not history:
            await query.answer("История пуста", show_alert=True)
            return

        try:
            file_data = export_chat_history(history)
            await context.bot.send_document(
                chat_id=user_id,
                document=file_data,
                filename=file_data.name,
                caption="📤 Экспорт истории чата"
            )
            await query.answer("Файл отправлен!")
        except Exception as e:
            logger.error(f"Export error: {e}")
            await query.answer("Ошибка экспорта", show_alert=True)

    # Referral
    elif data == "ref_stats":
        user = await db.get_user(user_id)
        if user:
            await query.edit_message_text(
                f"👥 <b>Статистика рефералов</b>\n\n"
                f"Приглашено: {user.referral_count}\n"
                f"Заработано: {user.referral_count * 5000:,} токенов",
                parse_mode='HTML'
            )

    # Close button
    elif data == "close":
        await query.message.delete()

    # Cancel
    elif data == "cancel":
        await query.edit_message_text("❌ Отменено")
        context.user_data.clear()

    # Back to main menu
    elif data == "main_menu":
        await query.edit_message_text(
            "Главное меню",
            reply_markup=None
        )
