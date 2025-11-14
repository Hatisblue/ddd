"""Database models for the AI Telegram Bot."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, Enum as SQLEnum
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class UserRole(enum.Enum):
    """User roles enumeration."""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    BANNED = "banned"


class AIModel(enum.Enum):
    """Available AI models."""
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    GEMINI_PRO = "gemini-pro"
    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GROK = "grok-beta"
    GROK_VISION = "grok-vision-beta"


class MessageType(enum.Enum):
    """Message types."""
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    DOCUMENT = "document"


class User(Base):
    """User model."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String(255))
    first_name: Mapped[Optional[str]] = mapped_column(String(255))
    last_name: Mapped[Optional[str]] = mapped_column(String(255))

    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole),
        default=UserRole.USER
    )

    # Token management
    tokens_available: Mapped[int] = mapped_column(Integer, default=10000)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    tokens_limit: Mapped[int] = mapped_column(Integer, default=100000)

    # Preferences
    current_model: Mapped[AIModel] = mapped_column(
        SQLEnum(AIModel),
        default=AIModel.GPT4O_MINI
    )
    language: Mapped[str] = mapped_column(String(10), default="ru")

    # Referral system
    referral_code: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    referred_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey('users.id'))
    referral_count: Mapped[int] = mapped_column(Integer, default=0)

    # Stats
    messages_count: Mapped[int] = mapped_column(Integer, default=0)
    images_generated: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Subscription
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    premium_until: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    payments: Mapped[list["Payment"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    statistics: Mapped[list["UserStatistics"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Message(Base):
    """Message history model."""
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.id'))

    # Message content
    role: Mapped[str] = mapped_column(String(50))  # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    message_type: Mapped[MessageType] = mapped_column(
        SQLEnum(MessageType),
        default=MessageType.TEXT
    )

    # AI model used
    model: Mapped[AIModel] = mapped_column(SQLEnum(AIModel))

    # Token usage
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="messages")


class Payment(Base):
    """Payment transactions model."""
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.id'))

    # Payment details
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(10))
    tokens_purchased: Mapped[int] = mapped_column(Integer)

    # Payment method
    payment_method: Mapped[str] = mapped_column(String(50))  # telegram_stars, yookassa
    payment_id: Mapped[str] = mapped_column(String(255))

    # Status
    status: Mapped[str] = mapped_column(String(50))  # pending, completed, failed

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="payments")


class UserStatistics(Base):
    """Daily user statistics model."""
    __tablename__ = "user_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey('users.id'))
    date: Mapped[datetime] = mapped_column(DateTime)

    # Usage stats
    messages_sent: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    images_generated: Mapped[int] = mapped_column(Integer, default=0)

    # Model usage
    gpt_requests: Mapped[int] = mapped_column(Integer, default=0)
    gemini_requests: Mapped[int] = mapped_column(Integer, default=0)
    grok_requests: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="statistics")


class PromoCode(Base):
    """Promo codes model."""
    __tablename__ = "promo_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True)

    # Bonus
    tokens_bonus: Mapped[int] = mapped_column(Integer)
    premium_days: Mapped[int] = mapped_column(Integer, default=0)

    # Limits
    max_uses: Mapped[int] = mapped_column(Integer, default=0)  # 0 = unlimited
    current_uses: Mapped[int] = mapped_column(Integer, default=0)

    # Validity
    valid_from: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    valid_until: Mapped[Optional[datetime]] = mapped_column(DateTime)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
