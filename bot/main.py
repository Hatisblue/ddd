"""Main bot file - entry point."""

import asyncio
import sys
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    filters
)
from loguru import logger

from bot.config import settings
from bot.database import db

# Import handlers
from bot.handlers.start import (
    start_handler,
    help_handler,
    balance_handler,
    models_handler,
    stats_handler,
    clear_handler,
    settings_handler,
    referral_handler,
    promo_handler,
    profile_handler
)
from bot.handlers.chat import message_handler, voice_handler
from bot.handlers.image import (
    image_command_handler,
    generate_image_handler,
    image_callback_handler
)
from bot.handlers.admin import (
    admin_handler,
    admin_users_handler,
    admin_stats_handler,
    admin_broadcast_handler,
    admin_callback_handler
)
from bot.handlers.payment import (
    payment_handler,
    payment_callback_handler,
    precheckout_callback,
    successful_payment_callback
)
from bot.handlers.callbacks import callback_handler


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="INFO"
)


async def error_handler(update: object, context) -> None:
    """Handle errors."""
    logger.error(f"Exception while handling an update: {context.error}")

    # Notify admins about error
    if update and isinstance(update, Update):
        if update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ Произошла ошибка. Администраторы уже уведомлены."
                )
            except:
                pass


def main():
    """Main function to run the bot."""
    logger.info("Starting AI Telegram Bot...")

    # Initialize database
    asyncio.run(db.init_db())
    logger.info("Database initialized")

    # Create application
    application = Application.builder().token(settings.bot_token).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("balance", balance_handler))
    application.add_handler(CommandHandler("models", models_handler))
    application.add_handler(CommandHandler("stats", stats_handler))
    application.add_handler(CommandHandler("clear", clear_handler))
    application.add_handler(CommandHandler("settings", settings_handler))
    application.add_handler(CommandHandler("referral", referral_handler))
    application.add_handler(CommandHandler("promo", promo_handler))
    application.add_handler(CommandHandler("profile", profile_handler))
    application.add_handler(CommandHandler("image", image_command_handler))

    # Admin commands
    application.add_handler(CommandHandler("admin", admin_handler))
    application.add_handler(CommandHandler("users", admin_users_handler))
    application.add_handler(CommandHandler("adminstats", admin_stats_handler))
    application.add_handler(CommandHandler("broadcast", admin_broadcast_handler))

    # Payment handlers
    application.add_handler(CommandHandler("pay", payment_handler))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(
        MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback)
    )

    # Callback query handlers
    application.add_handler(CallbackQueryHandler(
        image_callback_handler,
        pattern="^(image_|size_|quality_)"
    ))
    application.add_handler(CallbackQueryHandler(
        payment_callback_handler,
        pattern="^(pkg_|pay_|promo_code)"
    ))
    application.add_handler(CallbackQueryHandler(
        admin_callback_handler,
        pattern="^admin_"
    ))
    application.add_handler(CallbackQueryHandler(callback_handler))

    # Message handlers
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        message_handler
    ))
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))

    # Special filter for image prompts
    async def is_awaiting_image(update: Update) -> bool:
        if update.message and update.message.from_user:
            context = application.context_types.context
            return context.user_data.get('awaiting_image_prompt', False)
        return False

    application.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.PRIVATE,
        generate_image_handler
    ))

    # Error handler
    application.add_error_handler(error_handler)

    # Start bot
    logger.info("Bot is running...")
    logger.info(f"Admin IDs: {settings.admin_ids}")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
