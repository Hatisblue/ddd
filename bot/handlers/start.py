"""Start and basic command handlers."""

from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from bot.database import db
from bot.utils.keyboards import kb
from bot.utils.decorators import log_command, handle_errors, check_banned
from bot.utils.helpers import (
    format_user_info,
    parse_referral_code,
    generate_referral_link
)
from bot.config import settings


@handle_errors
@log_command
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    args = context.args

    # Parse referral code
    referrer_id = None
    if args and len(args) > 0:
        referral_code = parse_referral_code(args[0])
        if referral_code:
            # Find referrer by code
            referrer = await db.get_user_by_referral_code(referral_code)
            if referrer and referrer.id != user.id:
                referrer_id = referrer.id

    # Get or create user
    db_user = await db.get_or_create_user(
        user_id=user.id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        referred_by=referrer_id
    )

    welcome_text = (
        f"👋 Привет, {user.first_name}!\n\n"
        f"🤖 Я продвинутый AI-бот с доступом к последним моделям:\n"
        f"• GPT-4o, GPT-4 Turbo, o1\n"
        f"• Gemini 1.5 Pro & Flash\n"
        f"• Grok & Grok Vision\n\n"
        f"🎨 Генерация изображений через DALL-E 3\n"
        f"🎤 Поддержка голосовых сообщений\n"
        f"💾 История диалогов\n"
        f"📊 Детальная статистика\n\n"
        f"💰 Ваш баланс: {db_user.tokens_available:,} токенов\n\n"
        f"Выберите действие в меню ⬇️"
    )

    if referrer_id:
        welcome_text += (
            f"\n\n🎁 Вы зарегистрировались по реферальной ссылке!\n"
            f"Бонус: +{settings.referral_bonus} токенов"
        )

    await update.message.reply_text(
        welcome_text,
        reply_markup=kb.main_menu()
    )


@handle_errors
@log_command
@check_banned
async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "📖 <b>Справка по боту</b>\n\n"
        "<b>Основные команды:</b>\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать эту справку\n"
        "/balance - Проверить баланс токенов\n"
        "/models - Выбрать AI модель\n"
        "/stats - Показать статистику\n"
        "/clear - Очистить историю диалога\n"
        "/settings - Настройки\n\n"
        "<b>Как использовать:</b>\n"
        "• Просто отправьте текстовое сообщение для чата с AI\n"
        "• Используйте команду /image для генерации изображений\n"
        "• Отправьте голосовое сообщение для транскрипции\n\n"
        "<b>Модели AI:</b>\n"
        "• <b>GPT-4o</b> - Самая мощная модель OpenAI\n"
        "• <b>GPT-4o mini</b> - Быстрая и экономичная\n"
        "• <b>Gemini 1.5 Pro</b> - Продвинутая модель Google\n"
        "• <b>Grok</b> - Модель от xAI\n\n"
        "<b>Генерация изображений:</b>\n"
        "Используйте команду /image или кнопку в меню\n\n"
        "<b>Стоимость:</b>\n"
        "Токены расходуются в зависимости от модели и длины ответа.\n"
        "Пополнить баланс: /balance\n\n"
        "По вопросам: @support"
    )

    await update.message.reply_text(
        help_text,
        parse_mode='HTML'
    )


@handle_errors
@log_command
@check_banned
async def balance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /balance command."""
    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Ошибка: пользователь не найден")
        return

    balance_text = (
        f"💰 <b>Ваш баланс</b>\n\n"
        f"🪙 Доступно токенов: <b>{user.tokens_available:,}</b>\n"
        f"📊 Использовано: {user.tokens_used:,}\n"
        f"📈 Лимит: {user.tokens_limit:,}\n\n"
    )

    if user.is_premium:
        balance_text += "⭐️ У вас премиум-подписка\n\n"

    balance_text += "Выберите способ пополнения:"

    await update.message.reply_text(
        balance_text,
        parse_mode='HTML',
        reply_markup=kb.payment_menu()
    )


@handle_errors
@log_command
@check_banned
async def models_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /models command."""
    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Ошибка: пользователь не найден")
        return

    models_text = (
        f"🤖 <b>Выбор AI модели</b>\n\n"
        f"Текущая модель: <b>{user.current_model.value}</b>\n\n"
        f"Выберите модель из списка ниже:"
    )

    await update.message.reply_text(
        models_text,
        parse_mode='HTML',
        reply_markup=kb.ai_model_selection()
    )


@handle_errors
@log_command
@check_banned
async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command."""
    user_id = update.effective_user.id

    await update.message.reply_text(
        "📊 Выберите период для статистики:",
        reply_markup=kb.statistics_period()
    )


@handle_errors
@log_command
@check_banned
async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command - clear chat history."""
    user_id = update.effective_user.id

    await db.clear_user_history(user_id)

    await update.message.reply_text(
        "🗑 История диалога очищена!\n\n"
        "Можете начать новый разговор."
    )

    logger.info(f"User {user_id} cleared chat history")


@handle_errors
@log_command
@check_banned
async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /settings command."""
    await update.message.reply_text(
        "⚙️ <b>Настройки</b>\n\nВыберите параметр:",
        parse_mode='HTML',
        reply_markup=kb.settings_menu()
    )


@handle_errors
@log_command
@check_banned
async def referral_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /referral command."""
    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Ошибка: пользователь не найден")
        return

    bot_username = (await context.bot.get_me()).username
    referral_link = generate_referral_link(bot_username, user.referral_code)

    referral_text = (
        f"👥 <b>Реферальная программа</b>\n\n"
        f"Приглашайте друзей и получайте бонусы!\n\n"
        f"🎁 За каждого приглашенного: <b>{settings.referral_bonus:,} токенов</b>\n\n"
        f"👤 Ваших рефералов: <b>{user.referral_count}</b>\n\n"
        f"🔗 Ваша реферальная ссылка:\n"
        f"<code>{referral_link}</code>\n\n"
        f"Поделитесь ссылкой с друзьями!"
    )

    await update.message.reply_text(
        referral_text,
        parse_mode='HTML'
    )


@handle_errors
@log_command
@check_banned
async def promo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /promo command."""
    await update.message.reply_text(
        "🎁 <b>Активация промокода</b>\n\n"
        "Отправьте промокод следующим сообщением:",
        parse_mode='HTML'
    )

    # Set state to wait for promo code
    context.user_data['awaiting_promo'] = True


@handle_errors
@log_command
async def profile_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /profile command."""
    user_id = update.effective_user.id
    user = await db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Ошибка: пользователь не найден")
        return

    profile_text = format_user_info(user)

    await update.message.reply_text(
        f"<pre>{profile_text}</pre>",
        parse_mode='HTML'
    )
