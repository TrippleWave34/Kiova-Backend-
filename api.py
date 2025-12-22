from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import uuid
from azure.storage.blob import BlobServiceClient, ContentSettings
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
import base64
import firebase_admin
from firebase_admin import auth, credentials
from contextlib import asynccontextmanager

load_dotenv()

# --- CONFIGURATION ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AI_KEY = os.getenv("AZURE_AI_KEY")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT", "o4-mini")
DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")
FIREBASE_CREDS_B64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")

if not firebase_admin._apps:
    try:
        print("Loading Firebase credentials from Environment Variable...")
        cred_json = json.loads(base64.b64decode(FIREBASE_CREDS_B64))
        cred = credentials.Certificate(cred_json)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")

if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    raise ValueError("Azure Storage keys missing in .env")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
ai_client = AzureOpenAI(
    azure_endpoint=AI_ENDPOINT,
    api_key=AI_KEY,
    api_version="2024-12-01-preview"
)

# models.Base.metadata.drop_all(bind=engine) 
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Executing startup database check...")
    
    # We need to manually get a DB session here because Dependency Injection 
    # doesn't work inside lifespan events the same way as endpoints
    db = next(get_db()) 
    
    try:
        # Create tables (if not using Alembic)
        models.Base.metadata.create_all(bind=engine)

        # Check and Seed Categories
        existing_categories = db.query(models.Category).all()
        existing_names = {cat.name for cat in existing_categories}
        
        for category_name in INITIAL_CATEGORIES:
            if category_name not in existing_names:
                new_cat = models.Category(name=category_name)
                db.add(new_cat)
        
        db.commit()
        print("Initial categories seeded successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
    finally:
        db.close()

    # Yield control back to FastAPI to run the application
    yield 

    # --- SHUTDOWN LOGIC (Optional) ---
    print("Shutting down application...")


app = FastAPI(title="Kiova Real Backend v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down to your real domain
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

INITIAL_CATEGORIES = [
    "Casual", "Streetwear", "Smart Casual", "Business/Formal", "Athletic",
    "Minimalist", "Trendy", "Vintage", "Y2K", "Techwear", "Peppy",
    "Elegant", "Boho", "Luxury", "Poppy"
]

security = HTTPBearer()

# --- PYDANTIC SCHEMAS ---

class UserSchema(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    has_completed_onboarding: Optional[bool] = False     
    class Config:
        from_attributes = True

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
    # user_id: str 
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

class CategoryBase(BaseModel):
    name: str

class CategoryCreate(CategoryBase):
    pass

class CategoryResponse(CategoryBase):
    id: int
    class Config:
        from_attributes = True

class UserCategorySelection(BaseModel):
    category_ids: List[int]

# ==========================================
# AUTHENTICATION DEPENDENCY
# ==========================================

# --- NEW: Logic to verify token and get current user ---
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        # 1. Verify Token with Firebase
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        
        # 2. Check if user exists in Postgres
        user = db.query(models.User).filter(models.User.id == uid).first()
        
        if not user:
            # Auto-create user if they passed Firebase check but aren't in Postgres yet
            # (Alternatively, you can throw an error and force them to hit /auth/sync)
            user = models.User(id=uid, email=email)
            db.add(user)
            db.commit()
            db.refresh(user)
            
        return user
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ==========================================
# HELPER FUNCTIONS (No changes)
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
        blob_client.upload_blob(
            output_buffer, 
            content_settings=ContentSettings(content_type='image/png') 
        )
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
        if "color" in data and isinstance(data["color"], list):
            data["color"] = " and ".join(data["color"])
        return data
    except Exception as e:
        print(f"Vision Error: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis Failed")

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/users/sync", response_model=UserSchema)
def sync_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Flutter calls this immediately after Firebase Login.
    It ensures the user exists in Postgres.
    """
    token = credentials.credentials
    decoded_token = auth.verify_id_token(token)
    uid = decoded_token['uid']
    email = decoded_token.get('email')
    
    user = db.query(models.User).filter(models.User.id == uid).first()
    if not user:
        user = models.User(id=uid, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@app.post("/users/me/onboarding", response_model=UserSchema)
def complete_onboarding(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Sets the onboarding flag to True. Called when user finishes the intro flow.
    """
    current_user.has_completed_onboarding = True
    db.commit()
    db.refresh(current_user)
    return current_user

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image_url = process_and_upload(file.file.read(), file.filename)
    ai_data = get_ai_metadata(image_url)
    ai_data["processed_image_url"] = image_url
    return {"status": "success", "ai_results": ai_data}

@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    new_id = str(uuid.uuid4())
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

@app.get("/products/top-picks", response_model=List[ProductResponse])
def get_top_picks(db: Session = Depends(get_db)):
    return db.query(models.Product).order_by(func.random()).limit(5).all()

@app.get("/products/featured", response_model=List[ProductResponse])
def get_featured_products(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    user_styles = [cat.name for cat in current_user.selected_categories]
    
    if not user_styles:
        return db.query(models.Product).order_by(func.random()).limit(20).all()

    return db.query(models.Product).filter(
        models.Product.style.in_(user_styles)
    ).limit(20).all()

@app.post("/wardrobe", response_model=WardrobeItemResponse)
async def add_to_wardrobe(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) # <--- LOCKED
):
    image_url = process_and_upload(file.file.read(), file.filename)
    meta = get_ai_metadata(image_url)
    
    description = f"{meta.get('gender')} {meta.get('style')} {meta.get('color')} {meta.get('sub_category')} {' '.join(meta.get('tags', []))}"
    vector = get_vector(description)
    
    new_item = models.WardrobeItem(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
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
def get_wardrobe(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.WardrobeItem).filter(models.WardrobeItem.user_id == current_user.id).all()

@app.post("/style-me", response_model=StyleMeResponse)
def style_me(
    data: StyleMeRequest, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    source_item = None
    if data.wardrobe_item_id:
        # Ensure the user owns this item
        source_item = db.query(models.WardrobeItem).filter(
            models.WardrobeItem.id == data.wardrobe_item_id,
            models.WardrobeItem.user_id == current_user.id
        ).first()
    elif data.product_id:
        source_item = db.query(models.Product).filter(models.Product.id == data.product_id).first()
        
    if not source_item:
        raise HTTPException(404, "Item not found or does not belong to you")

    target_categories = []
    if source_item.category == "Top": target_categories = ["Bottom", "Shoes", "Outerwear"]
    elif source_item.category in ["Bottom", "Pants"]: target_categories = ["Top", "Shoes", "Outerwear"]
    elif source_item.category == "Shoes": target_categories = ["Top", "Bottom", "Outerwear"]
    
    target_genders = ["Unisex"]
    if source_item.gender == "Men":
        target_genders.append("Men")
    elif source_item.gender == "Women":
        target_genders.append("Women")
    else:
        target_genders.extend(["Men", "Women"])

    matches = db.query(models.Product).filter(
        models.Product.category.in_(target_categories),
        models.Product.gender.in_(target_genders),
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
    if category: query = query.filter(models.Product.category == category)
    if gender: query = query.filter(models.Product.gender.in_([gender, "Unisex"]))
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
def delete_wardrobe_item(
    item_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    item = db.query(models.WardrobeItem).filter(
        models.WardrobeItem.id == item_id,
        models.WardrobeItem.user_id == current_user.id 
    ).first()
    
    if not item: raise HTTPException(404, detail="Wardrobe item not found")
    
    # Delete from Azure Blob Storage here to save space
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=item.image_url.split('/')[-1])
    blob_client.delete_blob()

    db.delete(item)
    db.commit()
    return {"status": "deleted", "id": item_id}


# ==========================================
# CATEGORY ENDPOINTS
# ==========================================

@app.get("/categories", response_model=List[CategoryResponse])
def get_all_categories(db: Session = Depends(get_db)):
    """
    Public endpoint to fetch all style categories for the selection screen.
    """
    return db.query(models.Category).order_by(models.Category.name).all()

@app.post("/users/me/categories", status_code=status.HTTP_204_NO_CONTENT)
def select_user_categories(
    selection: UserCategorySelection,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Saves the list of categories a user has selected. Replaces any old selection.
    """
    # Clear previous selections
    current_user.selected_categories.clear()

    # Find the category objects from the provided IDs
    categories_to_add = db.query(models.Category).filter(models.Category.id.in_(selection.category_ids)).all()
    
    if len(categories_to_add) != len(selection.category_ids):
        raise HTTPException(status_code=404, detail="One or more category IDs not found")
        
    current_user.selected_categories.extend(categories_to_add)
    db.commit()
    return

@app.get("/users/me/categories", response_model=List[CategoryResponse])
def get_user_selected_categories(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Returns the categories the current logged-in user has selected.
    """
    return current_user.selected_categories

@app.post("/admin/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
def create_category(
    category: CategoryCreate, 
    db: Session = Depends(get_db)
    # TODO: Add an admin verification dependency here
):
    db_category = models.Category(name=category.name)
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

@app.put("/admin/categories/{category_id}", response_model=CategoryResponse)
def update_category(
    category_id: int, 
    category_update: CategoryCreate, 
    db: Session = Depends(get_db)
    # TODO: Add admin check
):
    db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
    if not db_category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    db_category.name = category_update.name
    db.commit()
    db.refresh(db_category)
    return db_category

@app.delete("/admin/categories/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_category(
    category_id: int, 
    db: Session = Depends(get_db)
    # TODO: Add admin check
):
    db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
    if not db_category:
        raise HTTPException(status_code=404, detail="Category not found")
        
    db.delete(db_category)
    db.commit()
    return