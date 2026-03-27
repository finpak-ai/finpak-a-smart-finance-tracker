# src/models.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Date
from sqlalchemy.orm import relationship
from .database import Base
from datetime import date

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    # New Profile Fields (Must match what we just added in migration)
    full_name = Column(String, default="User")
    country = Column(String, default="Pakistan")
    currency = Column(String, default="PKR")
    profile_pic = Column(String, default="/static/default_profile.png")

    records = relationship("FinancialRecord", back_populates="owner")

class FinancialRecord(Base):
    __tablename__ = "financial_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date, default=date.today)
    
    # Financial Data
    income = Column(Float); age = Column(Integer); dependents = Column(Integer)
    occupation = Column(String); city_tier = Column(String); desired_savings = Column(Float)
    rent = Column(Float); groceries = Column(Float); transport = Column(Float)
    eating_out = Column(Float); entertainment = Column(Float); utilities = Column(Float)
    healthcare = Column(Float); education = Column(Float); miscellaneous = Column(Float)
    insurance = Column(Float); loan_repayment = Column(Float)

    owner = relationship("User", back_populates="records")