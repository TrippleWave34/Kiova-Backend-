from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

if not DB_PASS:
    if os.getenv("AZURE_CONNECTION_STRING"): 
        pass 
    else:
        raise ValueError("No DB_PASSWORD found in .env file")

# Encode password to handle special characters (e.g. @, #, /)
encoded_password = urllib.parse.quote_plus(DB_PASS) if DB_PASS else ""

# CONNECTION STRING
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}"

# CREATE ENGINE (With SSL & Connection Keep-Alive)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"sslmode": "require"}, 
    pool_pre_ping=True,  
    pool_recycle=1800    
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()