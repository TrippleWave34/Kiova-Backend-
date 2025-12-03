from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles 
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from azure.storage.blob import BlobServiceClient
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Import our new modules
import models
from database import engine, get_db
from pydantic import BaseModel

load_dotenv()

# --- AZURE CONFIG ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AI_KEY = os.getenv("AZURE_AI_KEY")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT", "o4-mini")
DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")

# Validate (Crash if missing keys)
if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage keys missing in .env")

# Initialize Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
ai_client = AzureOpenAI(
    azure_endpoint=AI_ENDPOINT,
    api_key=AI_KEY,
    api_version="2024-12-01-preview"
)

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

class StyleMeRequest(BaseModel):
    # Only one needed, but keep flexible
    wardrobe_item_id: Optional[str] = None
    product_id: Optional[str] = None

# --- HELPER: GENERATE EMBEDDING ---
def get_vector(text_input: str):
    """Turns text into a list of 1536 numbers using Azure OpenAI."""
    try:
        response = ai_client.embeddings.create(
            input=text_input,
            model=DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

# --- 1. THE NEW UPLOAD ENDPOINT ---
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


# --- 2. AI AUTO-TAGGING (Vision) ---
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # 1. Upload temp image to get URL for Vision model
    upload_res = await upload_file(file)
    image_url = upload_res["url"]

    # 2. Call GPT-4o-Mini Vision
    try:
        response = ai_client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fashion expert. Analyze the clothing image. Return a VALID JSON object with keys: category, sub_category, color, pattern, style, and tags (list of 5 string keywords)."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this image:"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        ai_data = json.loads(response.choices[0].message.content)
        ai_data["processed_image_url"] = image_url # Return the URL to frontend
        ai_data["suggested_price"] = 0.0 # Placeholder
        
        return {"status": "success", "ai_results": ai_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. CREATE PRODUCT ENDPOINT ---
@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    new_id = str(uuid.uuid4())

    description_for_ai = f"{product.style} {product.color} {product.sub_category} {product.name} {' '.join(product.tags)}"
    vector = get_vector(description_for_ai)
    
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
        tags=product.tags,
        embedding=vector
    )
    
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/products", response_model=List[ProductResponse])
def get_products(category: Optional[str] = None, search: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Product)
    
    if category:
        query = query.filter(models.Product.category == category)
        
    # Basic Text Search (Can be upgraded to Vector Search later)
    if search:
        search_term = f"%{search}%"
        query = query.filter(models.Product.name.ilike(search_term))
        
    return query.all()


@app.delete("/products/{product_id}")
def delete_product(product_id: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(product)
    db.commit()
    return {"status": "deleted", "id": product_id}


# --- 5. STYLE ME (The Recommendation Engine) ---
@app.post("/style-me")
def style_me(data: StyleMeRequest, db: Session = Depends(get_db)):
    # 1. Fetch Source Item
    if data.product_id:
        source_item = db.query(models.Product).filter(models.Product.id == data.product_id).first()
    # Add logic for Wardrobe item fetching here when you build the Wardrobe Table
    else:
        raise HTTPException(404, "No item specified")
        
    if not source_item:
        raise HTTPException(404, "Item not found")

    # 2. Define Complementary Categories
    target_categories = []
    if source_item.category == "Top": target_categories = ["Bottom", "Shoes", "Outerwear"]
    elif source_item.category in ["Bottom", "Pants"]: target_categories = ["Top", "Shoes", "Outerwear"]
    elif source_item.category == "Shoes": target_categories = ["Top", "Bottom", "Outerwear"]
    
    # 3. Query Database
    # Logic: Get items in target categories + Matching Style
    matches = db.query(models.Product).filter(
        models.Product.category.in_(target_categories),
        models.Product.style == source_item.style,
        models.Product.id != source_item.id
    ).limit(5).all()

    return {
        "user_item": source_item,
        "styled_matches": matches,
        "style_tip": f"Matching {source_item.style} vibes."
    }