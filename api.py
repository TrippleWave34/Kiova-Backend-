from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles # New import
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import shutil
import os
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

# Import our new modules
import models
from database import engine, get_db
from pydantic import BaseModel

load_dotenv()

# --- AZURE CONFIG ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Validate (Crash if missing keys)
if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage keys missing in .env")

# Initialize Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# --- DATABASE RESET ---
# Since we changed the model (String -> Array), we need to recreate tables.
# In production, use Alembic. For now, we drop and create.
# models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Kiova Real Backend v1")

# --- 1. SETUP STATIC FILE SERVING (Local Storage) ---
# Create a folder named 'static' if it doesn't exist
os.makedirs("static", exist_ok=True)

# This makes http://localhost:8000/static/image.jpg accessible
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- PYDANTIC SCHEMAS ---
class ProductCreate(BaseModel):
    name: str
    price: float
    # CHANGED: Now accepts a list of strings
    image_urls: List[str] 
    category: str
    sub_category: str
    color: str
    pattern: str
    style: str
    tags: List[str]

class ProductResponse(ProductCreate):
    id: str
    class Config:
        orm_mode = True 

# --- 2. THE NEW UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads file to Azure Blob Storage and returns the public URL.
    """
    try:
        # 1. Create a unique filename
        file_extension = file.filename.split(".")[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # 2. Get the blob client
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, 
            blob=unique_filename
        )

        # 3. Upload the data
        # Note: file.file is a SpooledTemporaryFile, we read it
        blob_client.upload_blob(file.file.read())

        # 4. Construct the Public URL
        # Format: https://<account>.blob.core.windows.net/<container>/<filename>
        blob_url = blob_client.url
        
        return {"url": blob_url}

    except Exception as e:
        # Good for debugging what went wrong
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. CREATE PRODUCT ENDPOINT ---
@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    new_id = str(uuid.uuid4())
    
    db_item = models.Product(
        id=new_id,
        name=product.name,
        price=product.price,
        # Save the list of URLs directly
        image_urls=product.image_urls,
        category=product.category,
        sub_category=product.sub_category,
        color=product.color,
        pattern=product.pattern,
        style=product.style,
        tags=product.tags 
    )
    
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/products", response_model=List[ProductResponse])
def get_products(category: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Product)
    if category:
        query = query.filter(models.Product.category == category)
    return query.all()


@app.delete("/products/{product_id}")
def delete_product(product_id: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(product)
    db.commit()
    return {"status": "deleted", "id": product_id}