from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime, Table, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base
import datetime

user_selected_categories = Table('user_selected_categories', Base.metadata,
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('category_id', Integer, ForeignKey('categories.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True) # Firebase UID
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # --- UPDATE: Add relationship to categories ---
    selected_categories = relationship(
        "Category",
        secondary=user_selected_categories,
        back_populates="selected_by_users"
    )
    
    wardrobe_items = relationship("WardrobeItem", back_populates="owner")

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)

    # Relationship back to User
    selected_by_users = relationship(
        "User",
        secondary=user_selected_categories,
        back_populates="selected_categories"
    )

class Product(Base):
    __tablename__ = "products"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    price = Column(Float)
    image_urls = Column(JSON) 
    category = Column(String, nullable=True)     
    sub_category = Column(String, nullable=True) 
    color = Column(String, nullable=True)        
    pattern = Column(String, nullable=True)      
    style = Column(String, nullable=True)  
    gender = Column(String, nullable=True)       
    tags = Column(JSON)
    embedding = Column(Vector(1536))

class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    owner = relationship("User", back_populates="wardrobe_items")
    
    image_url = Column(String)
    
    category = Column(String, nullable=True)
    sub_category = Column(String, nullable=True)
    color = Column(String, nullable=True)
    pattern = Column(String, nullable=True)
    style = Column(String, nullable=True)
    gender = Column(String, nullable=True) 
    tags = Column(JSON)
    embedding = Column(Vector(1536))