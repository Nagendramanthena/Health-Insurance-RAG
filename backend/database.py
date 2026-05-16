from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# SQLite Database setup
DB_PATH = "storage/chat_history.db"
os.makedirs("storage", exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True, index=True) # session_id
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String) # 'user' or 'ai'
    content = Column(Text)
    intent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSession", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

def save_message(session_id: str, role: str, content: str, intent: str = None):
    db = SessionLocal()
    try:
        # Ensure session exists
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(id=session_id)
            db.add(session)
        
        # Add message
        msg = ChatMessage(session_id=session_id, role=role, content=content, intent=intent)
        db.add(msg)
        db.commit()
    finally:
        db.close()

def get_session_history(session_id: str):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
        return [{"role": m.role, "content": m.content, "intent": m.intent} for m in messages]
    finally:
        db.close()

def get_all_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        return [{"id": s.id, "updated_at": s.updated_at.isoformat()} for s in sessions]
    finally:
        db.close()
