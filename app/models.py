from __future__ import annotations

from datetime import datetime
from typing import Optional

from flask_login import UserMixin
from sqlalchemy import Index, Column, Integer, String, ForeignKey, DateTime, Text, JSON, Table, Float
from sqlalchemy.orm import relationship, Mapped, mapped_column

from . import db, login_manager


# Association table for many-to-many between Communication and Tag
communication_tags = Table(
    'communication_tags',
    db.metadata,
    Column('communication_id', Integer, ForeignKey('communications.id', ondelete='CASCADE'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True),
)


class Organization(db.Model):
    __tablename__ = 'organizations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    users = relationship('User', back_populates='organization', cascade='all, delete')
    communications = relationship('Communication', back_populates='organization', cascade='all, delete')
    tags = relationship('Tag', back_populates='organization', cascade='all, delete')


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    organization = relationship('Organization', back_populates='users')


@login_manager.user_loader
def load_user(user_id: str) -> Optional['User']:
    return db.session.get(User, int(user_id))


class Communication(db.Model):
    __tablename__ = 'communications'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)

    # Optional conversation threading
    conversation_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('conversations.id', ondelete='SET NULL'), nullable=True)

    sender: Mapped[str] = mapped_column(String(255), nullable=False)
    recipients: Mapped[str] = mapped_column(Text, nullable=False)  # comma-separated for v1
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)

    content_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_analysis: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Extracted metrics (duplicated for convenient querying)
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    char_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    readability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auto_detected_entities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pos_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    organization = relationship('Organization', back_populates='communications')
    tags = relationship('Tag', secondary=communication_tags, back_populates='communications')
    conversation = relationship('Conversation', back_populates='communications', lazy='joined')

    __table_args__ = (
        Index('ix_communications_org_hash', 'organization_id', 'sha256_hash'),
    )


class Tag(db.Model):
    __tablename__ = 'tags'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)

    organization = relationship('Organization', back_populates='tags')
    communications = relationship('Communication', secondary=communication_tags, back_populates='tags')

    __table_args__ = (
        Index('uq_tag_name_per_org', 'organization_id', 'name', unique=True),
    )


class Conversation(db.Model):
    __tablename__ = 'conversations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False, index=True)
    subject: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    communications = relationship('Communication', back_populates='conversation')
