from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
from pgvector.sqlalchemy import Vector
from database import Base

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
    tags = Column(JSON)
    embedding = Column(Vector(1536))

class WardrobeItem(Base):
    __tablename__ = "wardrobe_items"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True) # "user_123"
    image_url = Column(String)
    
    # AI Data
    category = Column(String, nullable=True)
    sub_category = Column(String, nullable=True)
    color = Column(String, nullable=True)
    pattern = Column(String, nullable=True)
    style = Column(String, nullable=True)
    tags = Column(JSON)
    
    # Vector for matching
    embedding = Column(Vector(1536))