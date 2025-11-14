"""Admin panel handlers."""

from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db, UserRole
from bot.utils.keyboards import kb
from bot.utils.decorators import admin_only, handle_errors, log_command
from bot.utils.helpers import format_user_info, format_statistics
from datetime import datetime, timedelta


@handle_errors
@log_command
@admin_only
async def admin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /admin command."""
    total_users = await db.get_total_users()

    admin_text = (
        f"👑 <b>Админ-панель</b>\n\n"
        f"👥 Всего пользователей: {total_users}\n\n"
        f"Выберите действие:"
    )

    await update.message.reply_text(
        admin_text,
        parse_mode='HTML',
        reply_markup=kb.admin_menu()
    )


@handle_errors
@admin_only
async def admin_users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List users for admin."""
    users = await db.get_all_users(limit=10)

    if not users:
        await update.message.reply_text("Нет пользователей")
        return

    text = "👥 <b>Последние пользователи:</b>\n\n"
    for user in users:
        text += (
            f"ID: {user.id}\n"
            f"👤 {user.first_name or 'N/A'}\n"
            f"🪙 Токены: {user.tokens_available:,}\n"
            f"💎 Роль: {user.role.value}\n"
            f"━━━━━━━━━━\n"
        )

    await update.message.reply_text(text, parse_mode='HTML')


@handle_errors
@admin_only
async def admin_stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show global statistics."""
    total_users = await db.get_total_users()

    # Get statistics for last 7 days
    stats_text = (
        f"📊 <b>Глобальная статистика</b>\n\n"
        f"👥 Всего пользователей: {total_users}\n"
    )

    await update.message.reply_text(stats_text, parse_mode='HTML')


@handle_errors
@admin_only
async def admin_broadcast_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle broadcast to all users."""
    if not context.args:
        await update.message.reply_text(
            "Использование: /broadcast <сообщение>"
        )
        return

    message = " ".join(context.args)
    users = await db.get_all_users(limit=10000)

    sent = 0
    failed = 0

    for user in users:
        try:
            await context.bot.send_message(
                chat_id=user.id,
                text=message,
                parse_mode='HTML'
            )
            sent += 1
        except Exception as e:
            failed += 1
            logger.error(f"Failed to send broadcast to {user.id}: {e}")

    await update.message.reply_text(
        f"📢 Рассылка завершена!\n\n"
        f"✅ Отправлено: {sent}\n"
        f"❌ Ошибок: {failed}"
    )


@handle_errors
async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin callbacks."""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data.startswith("admin_add_tokens_"):
        user_id = int(data.replace("admin_add_tokens_", ""))
        context.user_data['admin_target_user'] = user_id
        context.user_data['admin_action'] = 'add_tokens'

        await query.edit_message_text(
            f"Введите количество токенов для добавления пользователю {user_id}:"
        )

    elif data.startswith("admin_ban_"):
        user_id = int(data.replace("admin_ban_", ""))
        await db.update_user_role(user_id, UserRole.BANNED)

        await query.edit_message_text(
            f"🚫 Пользователь {user_id} заблокирован"
        )

    elif data.startswith("admin_premium_"):
        user_id = int(data.replace("admin_premium_", ""))
        user = await db.get_user(user_id)

        if user:
            user.is_premium = True
            user.premium_until = datetime.utcnow() + timedelta(days=30)
            await db.async_session().commit()

            await query.edit_message_text(
                f"⭐️ Пользователь {user_id} получил премиум на 30 дней"
            )

    elif data.startswith("admin_limit_"):
        user_id = int(data.replace("admin_limit_", ""))
        context.user_data['admin_target_user'] = user_id
        context.user_data['admin_action'] = 'set_limit'

        await query.edit_message_text(
            f"Введите новый лимит токенов для пользователя {user_id}:"
        )
