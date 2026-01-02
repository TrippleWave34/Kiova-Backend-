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
import torch
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
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

MODEL_NAME = "mattmdjaga/segformer_b2_clothes" 

print(f"Loading {MODEL_NAME}...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
print("Model Loaded.")

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

class WardrobeBatchCreate(BaseModel):
    processed_image_url: str
    category: str = "Unknown"
    sub_category: str = ""
    gender: str = "Unisex"
    color: str = ""
    pattern: str = ""
    style: str = ""
    tags: List[str] = []

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

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), 
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

def segment_and_upload(file_bytes, filename):
    try:
        # 1. Load Original Image
        input_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # 2. Get High-Quality Cutout using REMBG (The Outline)
        rembg_output = remove(input_image)
        rembg_arr = np.array(rembg_output)
        
        # 3. Get Semantic Segmentation (The Body Parts)
        inputs = processor(images=input_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Upscale mask to match image size
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=input_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0] # Keep as Tensor for dilation

        # 4. Define Body Labels to REMOVE
        # 0:Background, 1:Hat, 2:Hair, 3:Sunglasses, 4:Upper-clothes, 5:Skirt, 
        # 6:Pants, 7:Dress, 8:Belt, 9:Left-shoe, 10:Right-shoe, 11:Face, 
        # 12:Left-leg, 13:Right-leg, 14:Left-arm, 15:Right-arm, 16:Bag, 17:Scarf
        
        # We remove: Hair(2), Sunglasses(3), Face(11), Legs(12,13), Arms(14,15)
        body_labels = [2, 3, 11, 12, 13, 14, 15] 

        # Create binary mask (1 where body is, 0 elsewhere)
        # Using torch operations for speed
        body_mask = torch.zeros_like(pred_seg, dtype=torch.float32)
        for label in body_labels:
            body_mask = torch.where(pred_seg == label, 1.0, body_mask)

        # 5. DILATION (The Secret Sauce)
        # Expands the body mask by 5-10 pixels to eat up skin edges and halos
        # We use MaxPool2d as a hacky, fast dilation without needing OpenCV
        if body_mask.sum() > 0:
            body_mask = body_mask.unsqueeze(0).unsqueeze(0) # Add batch/channel dims
            # Kernel size 13 = Dilate by ~6 pixels. Increase if you still see skin edges.
            dilated_mask = torch.nn.functional.max_pool2d(body_mask, kernel_size=13, stride=1, padding=6)
            body_mask_np = dilated_mask.squeeze().numpy()
        else:
            body_mask_np = body_mask.numpy()

        # 6. Apply to REMBG Output
        alpha_channel = rembg_arr[:, :, 3]
        
        # If the pixel is body (1), make it transparent (0). Otherwise keep rembg alpha.
        new_alpha = np.where(body_mask_np > 0.5, 0, alpha_channel).astype(np.uint8)
        
        rembg_arr[:, :, 3] = new_alpha
        final_image = Image.fromarray(rembg_arr)

        # 7. Crop to Content
        bbox = final_image.getbbox()
        if bbox:
            final_image = final_image.crop(bbox)

        # 8. Upload
        output_buffer = io.BytesIO()
        final_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        unique_name = f"{uuid.uuid4()}.png"
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=unique_name)
        blob_client.upload_blob(
            output_buffer, 
            content_settings=ContentSettings(content_type='image/png') 
        )
        return blob_client.url

    except Exception as e:
        print(f"Segmentation Error: {e}")
        # Fallback to simple rembg if AI fails
        return process_and_upload(file_bytes, filename)
    

def segment_and_split(file_bytes):
    """
    Returns a list of dicts: [{'image_url': '...', 'category': 'Top', 'label_id': 4}, ...]
    """
    try:
        input_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        
        # 1. REMBG for the whole person outline
        rembg_output = remove(input_image)
        rembg_arr = np.array(rembg_output)
        
        # 2. Segformer for parts
        inputs = processor(images=input_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=input_image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

        # 3. Define Categories to Extract
        # Map Segformer IDs to readable names
        # 4:Upper-clothes, 5:Skirt, 6:Pants, 7:Dress, 9:Left-shoe, 10:Right-shoe, 16:Bag, 17:Scarf
        clothing_map = {
            4: "Top", 5: "Bottom", 6: "Bottom", 7: "Dress", 
            9: "Shoes", 10: "Shoes", 16: "Bag", 17: "Accessory"
        }
        
        detected_items = []
        found_labels = np.unique(pred_seg)

        for label_id in found_labels:
            if label_id not in clothing_map:
                continue
                
            category_name = clothing_map[label_id]
            
            # --- Special Logic for Shoes ---
            # Merge Left(9) and Right(10) into one "Shoes" item if both exist
            if label_id == 10 and 9 in found_labels:
                continue # Skip right shoe, we handle it with left shoe
            
            mask_ids = [label_id]
            if label_id == 9: mask_ids.append(10) # Grab both shoes
            
            # Create Mask for this specific item
            item_mask = np.isin(pred_seg, mask_ids).astype(np.uint8) * 255
            
            # Apply to REMBG alpha
            alpha_channel = rembg_arr[:, :, 3]
            new_alpha = np.where(item_mask > 0, alpha_channel, 0).astype(np.uint8)
            
            item_arr = rembg_arr.copy()
            item_arr[:, :, 3] = new_alpha
            final_item_img = Image.fromarray(item_arr)
            
            # Crop to content
            bbox = final_item_img.getbbox()
            if bbox:
                final_item_img = final_item_img.crop(bbox)
                
                # Upload
                output_buffer = io.BytesIO()
                final_item_img.save(output_buffer, format="PNG")
                output_buffer.seek(0)
                
                unique_name = f"{uuid.uuid4()}.png"
                blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=unique_name)
                blob_client.upload_blob(output_buffer, content_settings=ContentSettings(content_type='image/png'))
                
                detected_items.append({
                    "image_url": blob_client.url,
                    "category": category_name, # Rough category from Segformer
                    "confidence": 0.95 # Mock confidence
                })
                
        return detected_items

    except Exception as e:
        print(f"Split Error: {e}")
        return []

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
    image_url = segment_and_upload(file.file.read(), file.filename)
    ai_data = get_ai_metadata(image_url)
    ai_data["processed_image_url"] = image_url
    return {"status": "success", "ai_results": ai_data}

@app.post("/analyze-image-multi")
async def analyze_image_multi(file: UploadFile = File(...)):
    # 1. Split image into parts
    file_bytes = file.file.read()
    split_items = segment_and_split(file_bytes)
    
    if not split_items:
        # Fallback: If nothing detected, return original as one item
        # Reset file pointer or re-read? Better to pass bytes.
        # For simplicity, we just return error or fallback logic here.
        # Let's assume we reuse the single-item logic as fallback.
        url = segment_and_upload(file_bytes, file.filename)
        return {"status": "fallback", "items": [{"image_url": url, "category": "Unknown"}]}

    # 2. Run AI Analysis on EACH part (to get detailed tags/colors)
    final_results = []
    for item in split_items:
        meta = get_ai_metadata(item['image_url']) # GPT-4o analyzes the cropped part
        meta['processed_image_url'] = item['image_url']
        final_results.append(meta)

    return {"status": "success", "items": final_results}

@app.post("/products", response_model=ProductResponse)
def create_product(product: ProductCreate, 
                   db: Session = Depends(get_db),
                   current_user: models.User = Depends(get_current_user)):
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
        embedding=vector
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
    sold_ids = db.query(models.Order.product_id).filter(models.Order.status.in_(["PAID", "SHIPPED", "COMPLETED"])).subquery()
    
    query = db.query(models.Product).filter(models.Product.id.notin_(sold_ids))
    
    if current_user:
        query = query.filter(models.Product.user_id != current_user.id)
        
    return query.order_by(func.random()).limit(5).all()

@app.get("/products/featured", response_model=List[ProductResponse])
def get_featured_products(
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    # Filter Sold
    sold_ids = db.query(models.Order.product_id).filter(models.Order.status.in_(["PAID", "SHIPPED", "COMPLETED"])).subquery()
    
    user_styles = [cat.name for cat in current_user.selected_categories]
    
    query = db.query(models.Product).filter(
        models.Product.id.notin_(sold_ids),
        models.Product.user_id != current_user.id # Filter Own
    )

    if not user_styles:
        return query.order_by(func.random()).limit(20).all()

    return query.filter(models.Product.style.in_(user_styles)).limit(20).all()

@app.post("/wardrobe", response_model=WardrobeItemResponse)
async def add_to_wardrobe(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user) # <--- LOCKED
):
    image_url = segment_and_upload(file.file.read(), file.filename)
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

@app.post("/wardrobe/batch")
def add_wardrobe_batch(
    item: WardrobeBatchCreate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    # Vectorize
    description = f"{item.gender} {item.style} {item.color} {item.sub_category} {' '.join(item.tags)}"
    vector = get_vector(description)

    new_item = models.WardrobeItem(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        image_url=item.processed_image_url, # Already hosted
        category=item.category,
        sub_category=item.sub_category,
        gender=item.gender,
        color=item.color,
        pattern=item.pattern,
        style=item.style,
        tags=item.tags,
        embedding=vector
    )
    db.add(new_item)
    db.commit()
    return {"status": "saved"}

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
        source_item = db.query(models.WardrobeItem).filter(
            models.WardrobeItem.id == data.wardrobe_item_id,
            models.WardrobeItem.user_id == current_user.id
        ).first()
    elif data.product_id:
        source_item = db.query(models.Product).filter(models.Product.id == data.product_id).first()
        
    if not source_item:
        raise HTTPException(404, "Item not found")

    # 1. Determine Logic based on Category
    target_categories = []
    
    if source_item.category == "Dress":
        # If source is dress, we only want shoes, accessories, outerwear
        target_categories = ["Shoes", "Outerwear", "Accessory", "Bag"]
    elif source_item.category == "Top": 
        target_categories = ["Bottom", "Shoes", "Outerwear", "Accessory"]
    elif source_item.category in ["Bottom", "Pants", "Skirt"]: 
        target_categories = ["Top", "Shoes", "Outerwear", "Accessory"]
    elif source_item.category == "Shoes": 
        # Decide between Dress OR Top+Bottom? For now, stick to Top+Bottom defaults
        target_categories = ["Top", "Bottom", "Outerwear", "Dress"] 
    else:
        # Fallback
        target_categories = ["Top", "Bottom", "Shoes", "Dress"]

    # 2. Gender Logic
    target_genders = ["Unisex"]
    if source_item.gender == "Men": target_genders.append("Men")
    elif source_item.gender == "Women": target_genders.append("Women")
    else: target_genders.extend(["Men", "Women"])

    # 3. Fetch Matches

    sold_product_ids = db.query(models.Order.product_id).filter(
        models.Order.status.in_(["PAID", "SHIPPED", "COMPLETED"])
    ).subquery()

    matches = db.query(models.Product).filter(
        models.Product.category.in_(target_categories),
        models.Product.gender.in_(target_genders),
        models.Product.id != getattr(source_item, 'id', ''),
        models.Product.id.notin_(sold_product_ids),     
        models.Product.user_id != current_user.id        
    ).order_by(
        models.Product.embedding.cosine_distance(source_item.embedding)
    ).limit(10).all()

    # 4. Filter for Logic Conflicts (e.g. No Dress + Pants)
    final_matches = []
    categories_present = set()
    
    # If source is dress, mark top/bottom as 'taken' so we don't add them
    if source_item.category == "Dress":
        categories_present.add("Top")
        categories_present.add("Bottom")
    elif source_item.category in ["Top", "Bottom"]:
        categories_present.add("Dress") # Don't add a dress if we have a shirt/pants source

    for match in matches:
        cat = match.category
        
        # Conflict Resolution
        if cat == "Dress" and ("Top" in categories_present or "Bottom" in categories_present):
            continue # Skip dress if we have separates
        if cat in ["Top", "Bottom"] and "Dress" in categories_present:
            continue # Skip separates if we have a dress

        # Ensure variety (one per category)
        if cat not in categories_present:
            categories_present.add(cat)
            final_matches.append(match)
            
    # Limit to 5 for UI
    return {
        "user_item": source_item,
        "styled_matches": final_matches[:5],
        "style_tip": f"Matching {source_item.style} vibes for {source_item.gender}."
    }

@app.get("/products", response_model=List[ProductResponse])
def get_products(
    category: Optional[str] = None, 
    gender: Optional[str] = None, 
    search: Optional[str] = None, 
    db: Session = Depends(get_db),
    # Use the new optional dependency
    current_user: Optional[models.User] = Depends(get_current_user_optional) 
):
    query = db.query(models.Product)

    # 1. Hide Sold Products
    # Find all product IDs that are in active orders
    sold_product_ids = db.query(models.Order.product_id).filter(
        models.Order.status.in_(["PAID", "SHIPPED", "COMPLETED"])
    ).subquery()
    
    query = query.filter(models.Product.id.notin_(sold_product_ids))

    # 2. Hide Own Products (If logged in)
    if current_user:
        query = query.filter(models.Product.user_id != current_user.id)

    # 3. Standard Filters
    if category: 
        query = query.filter(models.Product.category == category)
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
        # Create Stripe Session
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': product.name, 'images': product.image_urls[:1] if product.image_urls else []},
                    'unit_amount': int(product.price * 100), # Cents
                },
                'quantity': 1,
            }],
            mode='payment',
            
            # CRITICAL: Ask Stripe to collect the address for us
            shipping_address_collection={
                'allowed_countries': ['US', 'CA', 'GB', 'DE', 'FR'], # Add your supported countries
            },
            
            success_url= os.getenv("FRONTEND_URL") + '/success', 
            cancel_url= os.getenv("FRONTEND_URL") + '/cancel',
            
            # Store metadata so we know what this payment is for in the webhook
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
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET") # Get this from CLI

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(400, "Invalid signature")

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Extract data
        meta = session.get('metadata', {})
        shipping = session.get('shipping_details', {})
        
        # Create Order in DB
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
    # 1. Filter: Only get orders where the current user is the SELLER
    orders = db.query(models.Order).filter(models.Order.seller_id == current_user.id).all()
    
    # 2. Map results safely
    result = []
    for o in orders:
        # Safe Product Name check
        p_name = "Unknown Product"
        if o.product:
            p_name = o.product.name
        elif o.product_id:
            # Fallback if the product row was deleted but order exists
            p_name = f"Product (ID: {o.product_id})"

        result.append({
            "id": o.id, 
            "product": p_name, 
            "amount": o.amount, 
            "status": o.status,
            "shipping": o.shipping_details
        })
    
    return result

@app.get("/orders/selling")
def get_seller_orders(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    orders = db.query(models.Order).filter(models.Order.seller_id == current_user.id).all()
    return orders

@app.post("/orders/{order_id}/ship")
def mark_order_shipped(order_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = db.query(models.Order).filter(models.Order.id == order_id, models.Order.seller_id == current_user.id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = "SHIPPED"
    db.commit()
    return {"status": "SHIPPED"}

# --- ADMIN: View All Orders ---
@app.get("/admin/orders")
def get_all_orders(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(403, "Not an Admin")
    
    orders = db.query(models.Order).all()
    
    # Manual mapping to ensure 'product' field is populated correctly
    result = []
    for o in orders:
        # Try to get the name, fallback to ID if product is missing/deleted
        p_name = "Unknown"
        if o.product:
            p_name = o.product.name
        elif o.product_id:
            p_name = f"ID: {o.product_id}"
            
        result.append({
            "id": o.id,
            "product": p_name, # <--- This sends the string the UI expects
            "amount": o.amount,
            "status": o.status,
            "shipping": o.shipping_details, # Ensure this matches frontend key
            "seller_id": o.seller_id
        })
        
    return result

# --- ADMIN: Mark Payout Complete ---
@app.post("/admin/orders/{order_id}/payout")
def mark_order_paid_out(order_id: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(403, "Not an Admin")
    
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order: raise HTTPException(404, "Order not found")
    
    order.status = "COMPLETED" # Payout sent to seller
    db.commit()
    return {"status": "COMPLETED"}