from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True) # This will be the Firebase UID
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship: One user has many wardrobe items
    # wardrobe_items = relationship("WardrobeItem", back_populates="owner")

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
    user_id = Column(String, index=True)
    # ForeignKey links to the User table
    # user_id = Column(String, ForeignKey("users.id"), index=True)
    
    image_url = Column(String)
    
    # AI Data
    category = Column(String, nullable=True)
    sub_category = Column(String, nullable=True)
    color = Column(String, nullable=True)
    pattern = Column(String, nullable=True)
    style = Column(String, nullable=True)
    gender = Column(String, nullable=True) 
    tags = Column(JSON)
    
    # Vector for matching
    embedding = Column(Vector(1536))

    # Relationship link back to User
    # owner = relationship("User", back_populates="wardrobe_items")