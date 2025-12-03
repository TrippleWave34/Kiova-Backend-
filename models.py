from sqlalchemy import Column, Integer, String, Float, ARRAY
from pgvector.sqlalchemy import Vector
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    price = Column(Float)
    image_urls = Column(ARRAY(String)) 
    category = Column(String)     
    sub_category = Column(String) 
    color = Column(String)        
    pattern = Column(String)      
    style = Column(String)        
    tags = Column(ARRAY(String))
    embedding = Column(Vector(1536))