"""Payment handlers for Telegram Stars and YooKassa."""

from telegram import Update, LabeledPrice
from telegram.ext import ContextTypes
from loguru import logger
import uuid

from bot.database import db
from bot.utils.keyboards import kb
from bot.utils.decorators import handle_errors, check_banned, log_command
from bot.utils.helpers import parse_package_data
from bot.config import settings


# Token packages (tokens, price in rubles, price in stars)
TOKEN_PACKAGES = {
    "pkg_50000_99": (50000, 99, 50),
    "pkg_100000_179": (100000, 179, 100),
    "pkg_250000_399": (250000, 399, 200),
    "pkg_500000_699": (500000, 699, 350),
    "pkg_1000000_1199": (1000000, 1199, 600),
}


@handle_errors
@log_command
@check_banned
async def payment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle payment selection."""
    await update.message.reply_text(
        "💰 <b>Выберите пакет токенов:</b>\n\n"
        "Токены используются для:\n"
        "• Общения с AI моделями\n"
        "• Генерации изображений\n"
        "• Обработки голосовых сообщений",
        parse_mode='HTML',
        reply_markup=kb.token_packages()
    )


@handle_errors
async def payment_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle payment callbacks."""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data.startswith("pkg_"):
        # Store selected package
        context.user_data['selected_package'] = data

        if data in TOKEN_PACKAGES:
            tokens, price_rub, price_stars = TOKEN_PACKAGES[data]

            await query.edit_message_text(
                f"💰 <b>Выбранный пакет:</b>\n\n"
                f"🪙 Токенов: {tokens:,}\n"
                f"💵 Цена: {price_rub}₽ / {price_stars} ⭐️\n\n"
                f"Выберите способ оплаты:",
                parse_mode='HTML',
                reply_markup=kb.payment_menu()
            )

    elif data == "pay_stars":
        # Telegram Stars payment
        package_id = context.user_data.get('selected_package')

        if not package_id or package_id not in TOKEN_PACKAGES:
            await query.answer("Выберите пакет", show_alert=True)
            return

        tokens, price_rub, price_stars = TOKEN_PACKAGES[package_id]

        try:
            # Create invoice for Telegram Stars
            await context.bot.send_invoice(
                chat_id=query.message.chat_id,
                title=f"Пакет токенов: {tokens:,}",
                description=f"Покупка {tokens:,} токенов для AI бота",
                payload=package_id,
                provider_token="",  # Empty for Stars
                currency="XTR",  # Telegram Stars
                prices=[LabeledPrice(label=f"{tokens:,} токенов", amount=price_stars)]
            )

            await query.message.reply_text(
                "⭐️ Оплата через Telegram Stars\n\n"
                "Нажмите на кнопку для оплаты"
            )

        except Exception as e:
            logger.error(f"Stars payment error: {e}")
            await query.message.reply_text(
                "❌ Ошибка создания платежа. Попробуйте позже."
            )

    elif data == "pay_yookassa":
        # YooKassa payment
        package_id = context.user_data.get('selected_package')

        if not package_id or package_id not in TOKEN_PACKAGES:
            await query.answer("Выберите пакет", show_alert=True)
            return

        tokens, price_rub, price_stars = TOKEN_PACKAGES[package_id]

        try:
            from yookassa import Configuration, Payment

            Configuration.account_id = settings.yookassa_shop_id
            Configuration.secret_key = settings.yookassa_secret_key

            payment_id = str(uuid.uuid4())

            payment = Payment.create({
                "amount": {
                    "value": f"{price_rub}.00",
                    "currency": "RUB"
                },
                "confirmation": {
                    "type": "redirect",
                    "return_url": "https://t.me/your_bot"
                },
                "capture": True,
                "description": f"Покупка {tokens:,} токенов",
                "metadata": {
                    "user_id": query.from_user.id,
                    "package_id": package_id,
                    "tokens": tokens
                }
            }, payment_id)

            # Save payment to database
            await db.create_payment(
                user_id=query.from_user.id,
                amount=price_rub,
                currency="RUB",
                tokens_purchased=tokens,
                payment_method="yookassa",
                payment_id=payment.id
            )

            confirmation_url = payment.confirmation.confirmation_url

            await query.message.reply_text(
                f"💳 <b>Оплата через ЮКассу</b>\n\n"
                f"Сумма: {price_rub}₽\n"
                f"Токенов: {tokens:,}\n\n"
                f"<a href='{confirmation_url}'>Перейти к оплате</a>",
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"YooKassa payment error: {e}")
            await query.message.reply_text(
                "❌ Ошибка создания платежа. Проверьте настройки ЮКассы."
            )

    elif data == "promo_code":
        await query.message.reply_text(
            "🎁 Введите промокод:"
        )
        context.user_data['awaiting_promo'] = True


@handle_errors
async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle pre-checkout query for Telegram Stars."""
    query = update.pre_checkout_query

    # Always approve (you can add validation here)
    await query.answer(ok=True)


@handle_errors
async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle successful payment."""
    payment = update.message.successful_payment
    user_id = update.effective_user.id

    package_id = payment.invoice_payload

    if package_id in TOKEN_PACKAGES:
        tokens, _, _ = TOKEN_PACKAGES[package_id]

        # Add tokens to user
        await db.add_user_tokens(user_id, tokens)

        # Create payment record
        await db.create_payment(
            user_id=user_id,
            amount=payment.total_amount,
            currency=payment.currency,
            tokens_purchased=tokens,
            payment_method="telegram_stars",
            payment_id=payment.telegram_payment_charge_id
        )

        # Mark as completed
        await db.complete_payment(payment.telegram_payment_charge_id)

        await update.message.reply_text(
            f"✅ <b>Оплата успешна!</b>\n\n"
            f"💰 Начислено: {tokens:,} токенов\n\n"
            f"Проверить баланс: /balance",
            parse_mode='HTML'
        )

        logger.info(f"User {user_id} purchased {tokens} tokens via Stars")
    else:
        logger.error(f"Unknown package ID in payment: {package_id}")


@handle_errors
async def yookassa_webhook_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle YooKassa webhook notifications.
    This should be set up as a separate webhook endpoint.
    """
    # This is a placeholder for YooKassa webhook handling
    # In production, you'd need to:
    # 1. Set up a web server to receive webhooks
    # 2. Verify webhook signatures
    # 3. Process payment notifications
    pass
