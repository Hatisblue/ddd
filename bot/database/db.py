"""Database connection and session management."""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import select, func
from loguru import logger

from bot.config import settings
from bot.database.models import (
    Base, User, Message, Payment, UserStatistics,
    PromoCode, UserRole, AIModel
)


class Database:
    """Database manager class."""

    def __init__(self, database_url: str):
        """Initialize database connection."""
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self):
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")

    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.async_session() as session:
            yield session

    # User operations
    async def get_or_create_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        referred_by: Optional[int] = None
    ) -> User:
        """Get existing user or create new one."""
        async with self.async_session() as session:
            # Check if user exists
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if user:
                # Update last active
                user.last_active = datetime.utcnow()
                await session.commit()
                await session.refresh(user)
                return user

            # Create new user
            referral_code = secrets.token_urlsafe(8)
            user = User(
                id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                tokens_available=settings.free_tokens_on_register,
                referral_code=referral_code,
                referred_by=referred_by
            )
            session.add(user)

            # Add bonus to referrer
            if referred_by and settings.enable_referral_system:
                referrer_result = await session.execute(
                    select(User).where(User.id == referred_by)
                )
                referrer = referrer_result.scalar_one_or_none()
                if referrer:
                    referrer.tokens_available += settings.referral_bonus
                    referrer.referral_count += 1
                    logger.info(f"Referral bonus {settings.referral_bonus} tokens to user {referred_by}")

            await session.commit()
            await session.refresh(user)
            logger.info(f"New user created: {user_id}")
            return user

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_user_by_referral_code(self, referral_code: str) -> Optional[User]:
        """Get user by referral code."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.referral_code == referral_code)
            )
            return result.scalar_one_or_none()

    async def update_user_tokens(self, user_id: int, tokens_used: int):
        """Update user token balance."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.tokens_available -= tokens_used
                user.tokens_used += tokens_used
                await session.commit()

    async def set_user_model(self, user_id: int, model: AIModel):
        """Set user's preferred AI model."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.current_model = model
                await session.commit()

    # Message operations
    async def add_message(
        self,
        user_id: int,
        role: str,
        content: str,
        model: AIModel,
        tokens_used: int = 0
    ) -> Message:
        """Add message to history."""
        async with self.async_session() as session:
            message = Message(
                user_id=user_id,
                role=role,
                content=content,
                model=model,
                tokens_used=tokens_used
            )
            session.add(message)

            # Update user stats
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.messages_count += 1

            await session.commit()
            await session.refresh(message)
            return message

    async def get_user_history(
        self,
        user_id: int,
        limit: int = 10
    ) -> list[Message]:
        """Get user's message history."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            messages = result.scalars().all()
            return list(reversed(messages))

    async def clear_user_history(self, user_id: int):
        """Clear user's message history."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Message).where(Message.user_id == user_id)
            )
            messages = result.scalars().all()
            for message in messages:
                await session.delete(message)
            await session.commit()

    # Payment operations
    async def create_payment(
        self,
        user_id: int,
        amount: float,
        currency: str,
        tokens_purchased: int,
        payment_method: str,
        payment_id: str
    ) -> Payment:
        """Create payment record."""
        async with self.async_session() as session:
            payment = Payment(
                user_id=user_id,
                amount=amount,
                currency=currency,
                tokens_purchased=tokens_purchased,
                payment_method=payment_method,
                payment_id=payment_id,
                status="pending"
            )
            session.add(payment)
            await session.commit()
            await session.refresh(payment)
            return payment

    async def complete_payment(self, payment_id: str):
        """Mark payment as completed and add tokens to user."""
        async with self.async_session() as session:
            result = await session.execute(
                select(Payment).where(Payment.payment_id == payment_id)
            )
            payment = result.scalar_one_or_none()

            if payment and payment.status == "pending":
                payment.status = "completed"
                payment.completed_at = datetime.utcnow()

                # Add tokens to user
                user_result = await session.execute(
                    select(User).where(User.id == payment.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user:
                    user.tokens_available += payment.tokens_purchased

                await session.commit()
                logger.info(f"Payment {payment_id} completed for user {payment.user_id}")

    # Statistics operations
    async def update_statistics(
        self,
        user_id: int,
        messages_sent: int = 0,
        tokens_used: int = 0,
        images_generated: int = 0,
        model: Optional[AIModel] = None
    ):
        """Update user statistics for today."""
        async with self.async_session() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            result = await session.execute(
                select(UserStatistics).where(
                    UserStatistics.user_id == user_id,
                    UserStatistics.date == today
                )
            )
            stats = result.scalar_one_or_none()

            if not stats:
                stats = UserStatistics(
                    user_id=user_id,
                    date=today
                )
                session.add(stats)

            stats.messages_sent += messages_sent
            stats.tokens_used += tokens_used
            stats.images_generated += images_generated

            if model:
                if "gpt" in model.value:
                    stats.gpt_requests += 1
                elif "gemini" in model.value:
                    stats.gemini_requests += 1
                elif "grok" in model.value:
                    stats.grok_requests += 1

            await session.commit()

    async def get_user_statistics(
        self,
        user_id: int,
        days: int = 7
    ) -> list[UserStatistics]:
        """Get user statistics for last N days."""
        async with self.async_session() as session:
            start_date = datetime.utcnow() - timedelta(days=days)
            result = await session.execute(
                select(UserStatistics)
                .where(
                    UserStatistics.user_id == user_id,
                    UserStatistics.date >= start_date
                )
                .order_by(UserStatistics.date.desc())
            )
            return list(result.scalars().all())

    # Admin operations
    async def get_all_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """Get all users (for admin panel)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User)
                .order_by(User.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_total_users(self) -> int:
        """Get total number of users."""
        async with self.async_session() as session:
            result = await session.execute(select(func.count(User.id)))
            return result.scalar() or 0

    async def update_user_role(self, user_id: int, role: UserRole):
        """Update user role."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.role = role
                await session.commit()

    async def set_user_token_limit(self, user_id: int, limit: int):
        """Set user token limit."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.tokens_limit = limit
                await session.commit()

    async def add_user_tokens(self, user_id: int, tokens: int):
        """Add tokens to user balance."""
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if user:
                user.tokens_available += tokens
                await session.commit()

    # Promo code operations
    async def create_promo_code(
        self,
        code: str,
        tokens_bonus: int,
        premium_days: int = 0,
        max_uses: int = 0,
        valid_until: Optional[datetime] = None
    ) -> PromoCode:
        """Create promo code."""
        async with self.async_session() as session:
            promo = PromoCode(
                code=code,
                tokens_bonus=tokens_bonus,
                premium_days=premium_days,
                max_uses=max_uses,
                valid_until=valid_until
            )
            session.add(promo)
            await session.commit()
            await session.refresh(promo)
            return promo

    async def use_promo_code(self, user_id: int, code: str) -> tuple[bool, str]:
        """Use promo code."""
        async with self.async_session() as session:
            result = await session.execute(
                select(PromoCode).where(PromoCode.code == code)
            )
            promo = result.scalar_one_or_none()

            if not promo:
                return False, "Промокод не найден"

            if not promo.is_active:
                return False, "Промокод неактивен"

            if promo.valid_until and datetime.utcnow() > promo.valid_until:
                return False, "Промокод истек"

            if promo.max_uses > 0 and promo.current_uses >= promo.max_uses:
                return False, "Промокод использован максимальное количество раз"

            # Apply promo code
            user_result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()

            if user:
                user.tokens_available += promo.tokens_bonus
                if promo.premium_days > 0:
                    if user.premium_until:
                        user.premium_until += timedelta(days=promo.premium_days)
                    else:
                        user.premium_until = datetime.utcnow() + timedelta(days=promo.premium_days)
                    user.is_premium = True

                promo.current_uses += 1
                await session.commit()

                return True, f"Промокод активирован! Получено {promo.tokens_bonus} токенов"

            return False, "Ошибка активации промокода"


# Global database instance
db = Database(settings.database_url)
