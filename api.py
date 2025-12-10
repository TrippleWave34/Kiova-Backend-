from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Header
from fastapi.staticfiles import StaticFiles 
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from azure.storage.blob import BlobServiceClient
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import models
from database import engine, get_db
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import io

load_dotenv()

# --- CONFIGURATION ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AI_KEY = os.getenv("AZURE_AI_KEY")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT", "o4-mini")
DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")

if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage keys missing in .env")

# Initialize Clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
ai_client = AzureOpenAI(
    azure_endpoint=AI_ENDPOINT,
    api_key=AI_KEY,
    api_version="2024-12-01-preview"
)

# --- DB INIT ---
# (Note: Use Alembic for migrations, this is just for initial setup)
# models.Base.metadata.drop_all(bind=engine) 
# models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Kiova Real Backend v2")

# --- PYDANTIC SCHEMAS ---
class ProductCreate(BaseModel):
    name: str
    price: float
    image_urls: List[str] 
    category: str
    sub_category: str
    gender: str
    color: str
    pattern: str
    style: str
    tags: List[str]

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    image_urls: Optional[List[str]] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    gender: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    style: Optional[str] = None
    tags: Optional[List[str]] = None

class ProductResponse(ProductCreate):
    id: str
    class Config:
        from_attributes = True

class WardrobeItemResponse(BaseModel):
    id: str
    image_url: str
    category: str
    gender: str
    style: str
    tags: List[str]
    class Config:
        from_attributes = True

class StyleMeRequest(BaseModel):
    wardrobe_item_id: Optional[str] = None
    product_id: Optional[str] = None

class StyleSourceResponse(BaseModel):
    id: str
    category: Optional[str] = None
    style: Optional[str] = None
    gender: Optional[str] = None
    image_url: Optional[str] = None       
    image_urls: Optional[List[str]] = None 
    class Config:
        from_attributes = True

class StyleMeResponse(BaseModel):
    user_item: StyleSourceResponse
    styled_matches: List[ProductResponse]
    style_tip: str

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def process_and_upload(file_bytes, filename):
    try:
        input_image = Image.open(io.BytesIO(file_bytes))
        output_image = remove(input_image)
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        unique_name = f"{uuid.uuid4()}.png"
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=unique_name)
        blob_client.upload_blob(output_buffer)
        return blob_client.url
    except Exception as e:
        print(f"Background Removal Error: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed")

def get_vector(text_input: str):
    try:
        response = ai_client.embeddings.create(
            input=text_input,
            model=DEPLOYMENT_EMBEDDING
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def get_ai_metadata(image_url: str):
    """
    Calls GPT-4o-Mini Vision to analyze the image.
    """
    try:
        response = ai_client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fashion expert. Analyze the clothing image. Return a VALID JSON object with keys: category (Top/Bottom/Shoes/Outerwear), sub_category, gender (Men/Women/Unisex), color, pattern, style, and tags (list of 5 string keywords)."
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
        
        data = json.loads(response.choices[0].message.content)

        # Check if color is returned as a list/array
        if "color" in data and isinstance(data["color"], list):
            data["color"] = " and ".join(data["color"])
            
        return data
    except Exception as e:
        print(f"Vision Error: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis Failed")

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image_url = process_and_upload(file.file.read(), file.filename)
    ai_data = get_ai_metadata(image_url)
    ai_data["processed_image_url"] = image_url
    return {"status": "success", "ai_results": ai_data}

@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    new_id = str(uuid.uuid4())

    # Generate Vector from the verified tags
    description = f"{product.gender} {product.style} {product.color} {product.sub_category} {product.name} {' '.join(product.tags)}"
    vector = get_vector(description)
    
    db_item = models.Product(
        id=new_id,
        name=product.name,
        price=product.price,
        image_urls=product.image_urls,
        category=product.category,
        sub_category=product.sub_category,
        gender=product.gender,
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

@app.post("/wardrobe", response_model=WardrobeItemResponse)
async def add_to_wardrobe(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_url = process_and_upload(file.file.read(), file.filename)
    meta = get_ai_metadata(image_url)
    
    description = f"{meta.get('gender')} {meta.get('style')} {meta.get('color')} {meta.get('sub_category')} {' '.join(meta.get('tags', []))}"
    vector = get_vector(description)
    
    new_item = models.WardrobeItem(
        id=str(uuid.uuid4()),
        user_id="user_12345",
        image_url=image_url,
        category=meta.get("category", "Unknown"),
        sub_category=meta.get("sub_category", ""),
        gender=meta.get("gender", "Unisex"),
        color=meta.get("color", ""),
        pattern=meta.get("pattern", ""),
        style=meta.get("style", "Casual"),
        tags=meta.get("tags", []),
        embedding=vector
    )
    
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@app.get("/wardrobe", response_model=List[WardrobeItemResponse])
def get_wardrobe(db: Session = Depends(get_db)):
    return db.query(models.WardrobeItem).all()

# --- 4. STYLE ME (Hybrid Search) ---
@app.post("/style-me", response_model=StyleMeResponse)
def style_me(data: StyleMeRequest, db: Session = Depends(get_db)):
    source_item = None
    if data.wardrobe_item_id:
        source_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == data.wardrobe_item_id).first()
    elif data.product_id:
        source_item = db.query(models.Product).filter(models.Product.id == data.product_id).first()
        
    if not source_item:
        raise HTTPException(404, "Item not found")

    # 1. Category Logic
    target_categories = []
    if source_item.category == "Top": target_categories = ["Bottom", "Shoes", "Outerwear"]
    elif source_item.category in ["Bottom", "Pants"]: target_categories = ["Top", "Shoes", "Outerwear"]
    elif source_item.category == "Shoes": target_categories = ["Top", "Bottom", "Outerwear"]
    
    # 2. Gender Logic 
    # If source is Men -> Match Men OR Unisex
    # If source is Women -> Match Women OR Unisex
    # If source is Unisex -> Match All
    target_genders = ["Unisex"]
    if source_item.gender == "Men":
        target_genders.append("Men")
    elif source_item.gender == "Women":
        target_genders.append("Women")
    else:
        # Source is Unisex, match everything
        target_genders.extend(["Men", "Women"])

    # 3. Query
    matches = db.query(models.Product).filter(
        models.Product.category.in_(target_categories),
        models.Product.gender.in_(target_genders), # Apply Gender Filter
        models.Product.id != getattr(source_item, 'id', '')
    ).order_by(
        models.Product.embedding.cosine_distance(source_item.embedding)
    ).limit(5).all()

    return {
        "user_item": source_item,
        "styled_matches": matches,
        "style_tip": f"Matching {source_item.style} vibes for {source_item.gender}."
    }

@app.get("/products", response_model=List[ProductResponse])
def get_products(category: Optional[str] = None, gender: Optional[str] = None, search: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Product)
    
    if category:
        query = query.filter(models.Product.category == category)
    
    # Filter by Gender in feed
    if gender:
        query = query.filter(models.Product.gender.in_([gender, "Unisex"]))

    if search:
        search_term = f"%{search}%"
        query = query.filter(models.Product.name.ilike(search_term))
        
    return query.all()

@app.put("/products/{product_id}", response_model=ProductResponse)
def update_product(product_id: str, updates: ProductUpdate, db: Session = Depends(get_db)):
    item = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not item: raise HTTPException(404, detail="Product not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    # Re-vectorize if descriptors change
    search_fields = ['style', 'color', 'sub_category', 'name', 'tags', 'gender']
    if any(k in update_data for k in search_fields):
        description = f"{item.gender} {item.style} {item.color} {item.sub_category} {item.name} {' '.join(item.tags)}"
        item.embedding = get_vector(description)

    db.commit()
    db.refresh(item)
    return item

@app.delete("/products/{product_id}")
def delete_product(product_id: str, db: Session = Depends(get_db)):
    item = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not item: raise HTTPException(404, detail="Product not found")
    db.delete(item)
    db.commit()
    return {"status": "deleted", "id": product_id}

@app.delete("/wardrobe/{item_id}")
def delete_wardrobe_item(item_id: str, db: Session = Depends(get_db)):
    item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id).first()
    if not item: raise HTTPException(404, detail="Wardrobe item not found")
    db.delete(item)
    db.commit()
    return {"status": "deleted", "id": item_id}