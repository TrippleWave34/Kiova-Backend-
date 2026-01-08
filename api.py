from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Header, status, Form
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
import stripe
from fastapi import Request 

load_dotenv()

# --- CONFIGURATION ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AI_KEY = os.getenv("AZURE_AI_KEY")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT", "o4-mini")
DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING", "text-embedding-3-small")
FIREBASE_CREDS_B64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

stripe.api_key = STRIPE_SECRET_KEY

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

# --- NO MORE SEGFORMER ---
# We have removed the heavy transformer models to prevent 'eating' clothes.

# models.Base.metadata.drop_all(bind=engine) 
models.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Executing startup database check...")
    
    db = next(get_db()) 
    
    try:
        models.Base.metadata.create_all(bind=engine)
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

    yield 
    print("Shutting down application...")


app = FastAPI(title="Kiova Real Backend v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

INITIAL_CATEGORIES = [
    "Casual", "Streetwear", "Smart Casual", "Business/Formal", "Athletic",
    "Minimalist", "Trendy", "Vintage", "Y2K", "Techwear", "Peppy",
    "Elegant", "Boho", "Luxury", "Poppy"
]

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False) 

# --- PYDANTIC SCHEMAS ---
class UserSchema(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    has_completed_onboarding: Optional[bool] = False
    payout_info: Optional[str] = None
    is_admin: Optional[bool] = False
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
    affiliate_url: Optional[str] = None

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
    affiliate_url: Optional[str] = None

class ProductResponse(ProductCreate):
    id: str
    status: str = "AVAILABLE"
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

class StyledItemResponse(BaseModel):
    id: str
    # Fields from Product
    name: Optional[str] = None
    price: Optional[float] = None
    image_urls: Optional[List[str]] = None
    # Fields from WardrobeItem
    image_url: Optional[str] = None
    # Common Fields
    category: str
    sub_category: Optional[str] = ""
    gender: str
    color: Optional[str] = ""
    pattern: Optional[str] = ""
    style: str
    tags: List[str] = []
    
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
    styled_matches: List[StyledItemResponse]
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

class PayoutInfoUpdate(BaseModel):
    info: str

class ShipOrderRequest(BaseModel):
    tracking_number: str
    carrier: str
    seller_note: Optional[str] = None

class OrderStatusUpdate(BaseModel):
    status: str

# ==========================================
# AUTHENTICATION DEPENDENCY
# ==========================================
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
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
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional), 
    db: Session = Depends(get_db)
):
    if not credentials:
        return None
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        return db.query(models.User).filter(models.User.id == uid).first()
    except:
        return None

# ==========================================
# HELPER FUNCTIONS 
# ==========================================

def process_and_upload(file_bytes, filename):
    """
    Standard REMBG processing. Simple, reliable, free.
    """
    try:
        input_image = Image.open(io.BytesIO(file_bytes))
        
        # Free, local background removal
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
                    "content": "You are a fashion expert. Analyze the clothing image. Return a VALID JSON object with keys: category (one of: Top, Bottom, Shoes, Outerwear, Hat, Accessory for items like sunglasses/jewelry), sub_category, gender (Men/Women/Unisex), color, pattern, style, and tags (list of 5 string keywords)."
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

def transfer_to_wardrobe(db: Session, order: models.Order):
    product = order.product
    if not product: 
        return
    main_image = product.image_urls[0] if product.image_urls else ""
    exists = db.query(models.WardrobeItem).filter(
        models.WardrobeItem.user_id == order.buyer_id,
        models.WardrobeItem.image_url == main_image
    ).first()
    
    if exists:
        return

    new_item = models.WardrobeItem(
        id=str(uuid.uuid4()),
        user_id=order.buyer_id,
        image_url=main_image,
        category=product.category,
        sub_category=product.sub_category,
        gender=product.gender,
        color=product.color,
        pattern=product.pattern,
        style=product.style,
        tags=product.tags,
        embedding=product.embedding 
    )
    db.add(new_item)

# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/users/sync", response_model=UserSchema)
def sync_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
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
    current_user.has_completed_onboarding = True
    db.commit()
    db.refresh(current_user)
    return current_user

@app.put("/users/me/payout-info")
def update_payout_info(
    data: PayoutInfoUpdate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    current_user.payout_info = data.info
    db.commit()
    return {"status": "updated"}

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    skip_ai: bool = Form(False) # <--- New Parameter
):
    """
    Standard Single Item Upload.
    If skip_ai is True, it ONLY removes background and hosts the image (faster/cheaper).
    """
    image_url = process_and_upload(file.file.read(), file.filename)
    
    if skip_ai:
        # Just return the URL, empty metadata
        return {
            "status": "success", 
            "ai_results": {
                "processed_image_url": image_url,
                "category": "Unknown", # Defaults
                "tags": []
            }
        }

    # Full Analysis for the main image
    ai_data = get_ai_metadata(image_url)
    ai_data["processed_image_url"] = image_url
    return {"status": "success", "ai_results": ai_data}

@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, 
                   db: Session = Depends(get_db),
                   current_user: models.User = Depends(get_current_user)): # Use required user now
    
    affiliate_link = None
    if current_user.is_admin and product.affiliate_url:
        affiliate_link = product.affiliate_url
    elif not current_user.is_admin and product.price == 0:
        raise HTTPException(status_code=400, detail="Price must be greater than zero.")

    if not current_user.payout_info and not affiliate_link:
        raise HTTPException(status_code=400, detail="You must set up Payout Details before listing an item.")
    
    new_id = str(uuid.uuid4())
    description = f"{product.gender} {product.style} {product.color} {product.sub_category} {product.name} {' '.join(product.tags)}"
    vector = get_vector(description)
    
    db_item = models.Product(
        id=new_id,
        user_id=current_user.id,
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
        embedding=vector,
        affiliate_url=affiliate_link
    )
    
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.get("/products/top-picks", response_model=List[ProductResponse])
def get_top_picks(
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    sold_orders = db.query(models.Order.product_id).filter(
        models.Order.status.in_(["PAID", "SHIPPED", "RECEIVED", "COMPLETED"]),
        models.Order.product_id.isnot(None)
    ).all()
    sold_ids = [row[0] for row in sold_orders]

    query = db.query(models.Product)
    if sold_ids:
        query = query.filter(models.Product.id.notin_(sold_ids))
    
    if current_user:
        query = query.filter(models.Product.user_id != current_user.id)
        
    return query.order_by(func.random()).limit(5).all()

def get_featured_products(
    db: Session = Depends(get_db), 
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    base_query = db.query(models.Product)
    sold_ids = [row[0] for row in db.query(models.Order.product_id).filter(models.Order.status.in_(["PAID", "SHIPPED", "RECEIVED", "COMPLETED"])).all()]
    if sold_ids:
        base_query = base_query.filter(models.Product.id.notin_(sold_ids))
    if current_user:
        base_query = base_query.filter(models.Product.user_id != current_user.id)

    user_styles = []
    if current_user and current_user.selected_categories:
        user_styles = [cat.name for cat in current_user.selected_categories]

    if user_styles:
        return base_query.filter(models.Product.style.in_(user_styles)).order_by(func.random()).limit(10).all()
    else:
        return []
    
@app.get("/products/discover", response_model=List[ProductResponse])
def get_discover_products(
    page: int = 1,
    page_size: int = 10,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user_optional)
):
    # This endpoint provides all other items with pagination.
    
    # 1. Base query (same as above)
    query = db.query(models.Product)
    sold_ids = [row[0] for row in db.query(models.Order.product_id).filter(models.Order.status.in_(["PAID", "SHIPPED", "RECEIVED", "COMPLETED"])).all()]
    if sold_ids:
        query = query.filter(models.Product.id.notin_(sold_ids))
    if current_user:
        query = query.filter(models.Product.user_id != current_user.id)

    # 2. Apply pagination
    offset = (page - 1) * page_size
    return query.order_by(func.random()).offset(offset).limit(page_size).all()

@app.post("/wardrobe", response_model=WardrobeItemResponse)
async def add_to_wardrobe(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) 
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
    source_is_product = False

    if data.product_id:
        source_item = db.query(models.Product).filter(models.Product.id == data.product_id).first()
        source_is_product = True
    elif data.wardrobe_item_id:
        source_item = db.query(models.WardrobeItem).filter(
            models.WardrobeItem.id == data.wardrobe_item_id,
            models.WardrobeItem.user_id == current_user.id
        ).first()
    
    if not source_item:
        raise HTTPException(404, "Source item not found")

    target_categories = []
    if source_item.category == "Dress":
        target_categories = ["Shoes", "Outerwear", "Accessory", "Bag", "Hat"]
    elif source_item.category == "Top": 
        target_categories = ["Bottom", "Shoes", "Outerwear", "Accessory", "Hat"]
    elif source_item.category in ["Bottom", "Pants", "Skirt"]: 
        target_categories = ["Top", "Shoes", "Outerwear", "Accessory", "Hat"]
    elif source_item.category == "Shoes": 
        target_categories = ["Top", "Bottom", "Outerwear", "Dress", "Hat"] 
    elif source_item.category in ["Hat", "Accessory", "Bag"]:
        target_categories = ["Top", "Bottom", "Shoes", "Outerwear", "Dress"]
    else:
        target_categories = ["Top", "Bottom", "Shoes", "Dress"]

    target_genders = ["Unisex"]
    if source_item.gender == "Men": target_genders.append("Men")
    elif source_item.gender == "Women": target_genders.append("Women")
    else: target_genders.extend(["Men", "Women"])

    matches_from_db = []
    if source_is_product:
        matches_from_db = db.query(models.WardrobeItem).filter(
            models.WardrobeItem.user_id == current_user.id,
            models.WardrobeItem.category.in_(target_categories),
            models.WardrobeItem.gender.in_(target_genders)
        ).order_by(
            models.WardrobeItem.embedding.cosine_distance(source_item.embedding)
        ).limit(10).all()
    else:
        sold_product_ids = db.query(models.Order.product_id).filter(
            models.Order.status.in_(["PAID", "SHIPPED", "COMPLETED"])
        ).subquery()
        matches_from_db = db.query(models.Product).filter(
            models.Product.category.in_(target_categories),
            models.Product.gender.in_(target_genders),
            models.Product.id.notin_(sold_product_ids),     
            models.Product.user_id != current_user.id        
        ).order_by(
            models.Product.embedding.cosine_distance(source_item.embedding)
        ).limit(10).all()

    final_matches = []
    categories_present = {source_item.category}
    
    if source_item.category == "Dress":
        categories_present.update(["Top", "Bottom"])
    elif source_item.category in ["Top", "Bottom", "Pants", "Skirt"]:
        categories_present.add("Dress")

    for match in matches_from_db:
        cat = match.category
        if cat == "Dress" and ("Top" in categories_present or "Bottom" in categories_present):
            continue
        if cat in ["Top", "Bottom"] and "Dress" in categories_present:
            continue
        if cat in ["Hat", "Accessory"] and ("Hat" in categories_present or "Accessory" in categories_present):
             continue 

        if cat not in categories_present:
            categories_present.add(cat)
            final_matches.append(match)

    response_matches = []
    for match in final_matches[:5]:
        if isinstance(match, models.Product):
            response_matches.append(StyledItemResponse.model_validate(match))
        elif isinstance(match, models.WardrobeItem):
            response_matches.append(
                StyledItemResponse(
                    id=match.id,
                    name=f"My {match.style} {match.sub_category or match.category}",
                    price=None,
                    image_url=match.image_url,
                    image_urls=[match.image_url],
                    category=match.category,
                    sub_category=match.sub_category,
                    gender=match.gender,
                    color=match.color,
                    pattern=match.pattern,
                    style=match.style,
                    tags=match.tags
                )
            )

    return {
        "user_item": source_item,
        "styled_matches": response_matches,
        "style_tip": f"Matching {source_item.style} vibes for {source_item.gender}."
    }

@app.get("/products", response_model=List[ProductResponse])
def get_products(
    category: Optional[str] = None, 
    gender: Optional[str] = None, 
    search: Optional[str] = None, 
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user_optional) 
):
    query = db.query(models.Product)

    sold_orders = db.query(models.Order.product_id).filter(
        models.Order.status.in_(["PAID", "SHIPPED", "RECEIVED", "COMPLETED"]), 
        models.Order.product_id.isnot(None)
    ).all()
    
    sold_ids = [row[0] for row in sold_orders] 
    
    if sold_ids:
        query = query.filter(models.Product.id.notin_(sold_ids))

    if current_user:
        query = query.filter(models.Product.user_id != current_user.id)

    if category: query = query.filter(models.Product.category == category)
    if gender: query = query.filter(models.Product.gender.in_([gender, "Unisex"]))
    if search:
        search_term = f"%{search}%"
        query = query.filter(models.Product.name.ilike(search_term))
        
    return query.all()

@app.get("/products/me", response_model=List[ProductResponse])
def get_my_products(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    products = db.query(models.Product).filter(models.Product.user_id == current_user.id).all()
    sales = db.query(models.Order).filter(models.Order.seller_id == current_user.id).all()
    product_status_map = {order.product_id: order.status for order in sales}

    results = []
    for p in products:
        p_data = ProductResponse.model_validate(p)
        if p.id in product_status_map:
             p_data.status = product_status_map[p.id] 
        else:
             p_data.status = "AVAILABLE"
             
        results.append(p_data)
        
    return results

@app.put("/products/{product_id}", response_model=ProductResponse)
def update_product(
    product_id: str, 
    updates: ProductUpdate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) # Require user for security
):
    item = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not item: raise HTTPException(404, detail="Product not found")

    if item.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to edit this product")

    update_data = updates.model_dump(exclude_unset=True)
    
    if 'affiliate_url' in update_data:
        if current_user.is_admin:
            setattr(item, 'affiliate_url', update_data['affiliate_url'])
        del update_data['affiliate_url'] 

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
    return db.query(models.Category).order_by(models.Category.name).all()

@app.post("/users/me/categories", status_code=status.HTTP_204_NO_CONTENT)
def select_user_categories(
    selection: UserCategorySelection,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    current_user.selected_categories.clear()
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
    return current_user.selected_categories

@app.post("/admin/categories", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
def create_category(
    category: CategoryCreate, 
    db: Session = Depends(get_db)
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
):
    db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
    if not db_category:
        raise HTTPException(status_code=404, detail="Category not found")
        
    db.delete(db_category)
    db.commit()
    return

# --- PAYMENT ENDPOINTS ---

@app.post("/create-checkout-session")
def create_checkout_session(
    product_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product: raise HTTPException(404, "Product not found")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'gbp',
                    'product_data': {'name': product.name, 'images': product.image_urls[:1] if product.image_urls else []},
                    'unit_amount': int(product.price * 100), 
                },
                'quantity': 1,
            }],
            mode='payment',
            shipping_address_collection={
                'allowed_countries': ['US', 'CA', 'GB', 'DE', 'FR'], 
            },
            success_url= os.getenv("FRONTEND_URL") + '/#/success', 
            cancel_url= os.getenv("FRONTEND_URL") + '/#/cancel',
            metadata={
                'product_id': product.id,
                'buyer_id': current_user.id,
                'seller_id': product.user_id
            }
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET") 

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(400, "Invalid signature")

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        meta = session.get('metadata', {})
        
        shipping = session.get('shipping_details')
        if not shipping:
            shipping = session.get('customer_details')

        print(f"📦 Saving Address: {shipping}") 

        new_order = models.Order(
            id=session['id'],
            product_id=meta.get('product_id'),
            buyer_id=meta.get('buyer_id'),
            seller_id=meta.get('seller_id'),
            amount=session['amount_total'] / 100,
            status="PAID",
            shipping_details=shipping 
        )
        db.add(new_order)
        db.commit()
        print("✅ Order saved successfully!")

    return {"status": "success"}

# --- SELLER ENDPOINTS ---

@app.get("/orders/selling")
def get_seller_orders(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    orders = db.query(models.Order).filter(models.Order.seller_id == current_user.id).all()
    
    result = []
    for o in orders:
        p_name = "Unknown Product"
        if o.product:
            p_name = o.product.name
        elif o.product_id:
            p_name = f"Product (ID: {o.product_id})"

        result.append({
            "id": o.id, 
            "product": p_name, 
            "amount": o.amount, 
            "status": o.status,
            "shipping_details": o.shipping_details,
            "seller_note": o.seller_note 
        })
    
    return result

@app.post("/orders/{order_id}/ship")
def mark_order_shipped(
    order_id: str, 
    data: ShipOrderRequest, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    order = db.query(models.Order).filter(models.Order.id == order_id, models.Order.seller_id == current_user.id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = "SHIPPED"
    order.tracking_number = data.tracking_number
    order.carrier = data.carrier
    order.seller_note = data.seller_note
    db.commit()
    return {"status": "SHIPPED"}

@app.post("/orders/{order_id}/receive")
def mark_order_received(
    order_id: str, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    order = db.query(models.Order).filter(models.Order.id == order_id, models.Order.buyer_id == current_user.id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = "RECEIVED"
    transfer_to_wardrobe(db, order)
    
    db.commit()
    return {"status": "RECEIVED"}

@app.get("/orders/buying")
def get_buyer_orders(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    orders = db.query(models.Order).filter(models.Order.buyer_id == current_user.id).all()
    return [{
        "id": o.id, 
        "product": o.product.name if o.product else "Unknown", 
        "amount": o.amount, 
        "status": o.status,
        "tracking_number": o.tracking_number,
        "carrier": o.carrier,
        "seller_note": o.seller_note
    } for o in orders]

# --- ADMIN: View All Orders ---
@app.get("/admin/orders")
def get_all_orders(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(403, "Not an Admin")
    
    orders = db.query(models.Order).all()
    
    result = []
    for o in orders:
        p_name = o.product.name if o.product else f"ID: {o.product_id}"
        
        seller_email = "Unknown"
        seller_payout = "No Info"
        
        if o.seller_id:
            seller = db.query(models.User).filter(models.User.id == o.seller_id).first()
            if seller:
                seller_email = seller.email
                if seller.payout_info:
                    seller_payout = seller.payout_info

        result.append({
            "id": o.id,
            "product": p_name,
            "amount": o.amount,
            "status": o.status,
            "shipping_details": o.shipping_details, 
            "seller_id": o.seller_id,
            "seller_email": seller_email,
            "seller_payout_info": seller_payout,
            "tracking_number": o.tracking_number,
            "carrier": o.carrier
        })
        
    return result

# --- ADMIN: Mark Payout Complete ---
@app.post("/admin/orders/{order_id}/payout")
def mark_order_paid_out(order_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(403, "Not an Admin")
    
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = "COMPLETED"
    transfer_to_wardrobe(db, order)
    
    db.commit()
    return {"status": "COMPLETED"}

@app.put("/admin/orders/{order_id}/status")
def admin_update_order_status(
    order_id: str, 
    data: OrderStatusUpdate,
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    if not current_user.is_admin:
        raise HTTPException(403, "Not an Admin")
    
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = data.status

    if data.status in ["RECEIVED", "COMPLETED"]:
        transfer_to_wardrobe(db, order)

    db.commit()
    return {"status": order.status}

# --- BUYER/SELLER: Get Notifications ---
@app.get("/notifications")
def get_notifications(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    notifications = []

    # 1. Buyer Notifications
    shipped_orders = db.query(models.Order).filter(
        models.Order.buyer_id == current_user.id,
        models.Order.status == "SHIPPED"
    ).all()
    
    for o in shipped_orders:
        p_name = o.product.name if o.product else "Item"
        notifications.append({
            "id": o.id,
            "message": f"📦 Shipped: '{p_name}' is on the way!",
            "seller_note": o.seller_note if o.seller_note else "Check 'My Purchases' for tracking.",
            "type": "buyer"
        })

    # 2. Seller Notifications
    new_sales = db.query(models.Order).filter(
        models.Order.seller_id == current_user.id,
        models.Order.status == "PAID"
    ).all()

    for o in new_sales:
        p_name = o.product.name if o.product else "Item"
        notifications.append({
            "id": o.id,
            "message": f"💰 New Sale: Someone bought '{p_name}'!",
            "seller_note": "Please go to 'My Sales' to view address and ship.",
            "type": "seller"
        })
        
    # 3. Seller Notifications: Payout
    completed_sales = db.query(models.Order).filter(
        models.Order.seller_id == current_user.id,
        models.Order.status == "COMPLETED"
    ).limit(5).all()
    
    for o in completed_sales:
        p_name = o.product.name if o.product else "Item"
        notifications.append({
            "id": o.id,
            "message": f"✅ Payout Sent: '{p_name}' completed.",
            "seller_note": "Funds have been sent to your payout method.",
            "type": "seller"
        })

    return notifications