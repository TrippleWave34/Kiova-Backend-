from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
import uuid
from datetime import datetime

app = FastAPI(
    title="Kiova Mock Backend API",
    description="Auth, Wardrobe, Style Match, Market, Cart, Orders, Saved Outfits (CRUD).",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MOCK DATA STORES
# ==========================================

MOCK_PRODUCTS = [
    {
        "id": "item_001",
        "name": "Vintage Denim Jacket",
        "category": "Outerwear",
        "sub_category": "Jacket",
        "color": "Blue",
        "pattern": "Solid",
        "price": 89.99,
        "image_urls": ["https://placehold.co/400x400/000080/FFFFFF/png?text=Denim+Jacket"],
        "tags": ["Vintage", "Casual", "Blue", "Denim"],
        "style": "Vintage"
    },
    {
        "id": "item_002",
        "name": "White Linen Trousers",
        "category": "Bottom",
        "sub_category": "Pants",
        "color": "White",
        "pattern": "Solid",
        "price": 45.50,
        "image_urls": [
            "https://placehold.co/400x400/F5F5DC/000000/png?text=Linen+Pants+Front",
            "https://placehold.co/400x400/F5F5DC/000000/png?text=Linen+Pants+Back"
        ],
        "tags": ["Summer", "Classy", "White", "Linen"],
        "style": "Classy"
    },
    {
        "id": "item_003",
        "name": "Canvas Sneakers",
        "category": "Shoes",
        "sub_category": "Sneakers",
        "color": "Grey",
        "pattern": "Solid",
        "price": 29.99,
        "image_urls": [
            "https://placehold.co/400x400/808080/FFFFFF/png?text=Sneakers",
            "https://placehold.co/400x400/808080/FFFFFF/png?text=Sneakers+Side"
        ],
        "tags": ["Sporty", "Streetwear", "Grey"],
        "style": "Streetwear"
    },
    {
        "id": "item_004",
        "name": "Crisp White Oxford Shirt",
        "category": "Top",
        "sub_category": "Shirt",
        "color": "White",
        "pattern": "Solid",
        "price": 45.00,
        "image_urls": ["https://placehold.co/400x400/FFFFFF/000000/png?text=White+Shirt"],
        "tags": ["Formal", "Work", "White"],
        "style": "Classy"
    },
    {
        "id": "item_005",
        "name": "Graphic Band Tee",
        "category": "Top",
        "sub_category": "T-Shirt",
        "color": "Black",
        "pattern": "Graphic",
        "price": 35.00,
        "image_url": [
            "https://placehold.co/400x400/000000/FFFFFF/png?text=Band+Tee",
            "https://placehold.co/400x400/000000/FFFFFF/png?text=Band+Tee+Back"
            ],
        "tags": ["Edgy", "Black", "Cotton"],
        "style": "Edgy"
    }
]

MOCK_USER_WARDROBE: List[Dict] = []
MOCK_SAVED_OUTFITS: List[Dict] = [] # For the "Saved Outfits" sidebar
MOCK_USER_PROFILE = {
    "user_id": "user_12345",
    "selected_styles": [] 
}
MOCK_CART: List[Dict] = []
MOCK_ORDERS: List[Dict] = []

# ==========================================
# PYDANTIC MODELS
# ==========================================

class LoginRequest(BaseModel):
    token: str

class OnboardingRequest(BaseModel):
    styles: List[str] 

class ProductBase(BaseModel):
    name: str
    price: float
    tags: List[str]
    image_urls: List[str]
    category: str       
    sub_category: str   
    color: str          
    pattern: str        
    style: str          

class ProductResponse(ProductBase):
    id: str

class WardrobeItemResponse(BaseModel):
    id: str
    tags: List[str]
    image_url: str 
    category: str
    sub_category: str
    color: str
    pattern: str
    style: str

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    style: Optional[str] = None

class CartItemRequest(BaseModel):
    product_id: str
    quantity: int = 1

class CheckoutRequest(BaseModel):
    payment_method_id: str 
    shipping_address: str

class StyleMeRequest(BaseModel):
    wardrobe_item_id: Optional[str] = None
    product_id: Optional[str] = None 

class SaveOutfitRequest(BaseModel):
    name: str # "My Summer Look"
    wardrobe_item_id: Optional[str] = None
    product_ids: List[str] # List of items from store

# ==========================================
# ROUTES
# ==========================================

@app.get("/")
def health_check():
    return {"status": "Kiova API Online", "version": "3.2.1"}

# ------------------------------------------
# 1. AUTH & ONBOARDING
# ------------------------------------------
@app.post("/login")
def login(data: LoginRequest):
    if data.token == "error":
        raise HTTPException(status_code=401, detail="Invalid Token")
    
    return {
        "user_id": "user_12345",
        "email": "shopper@kiova.com",
        "role": "shopper", 
        "session_token": "mock_jwt_xyz",
        "has_completed_onboarding": len(MOCK_USER_PROFILE["selected_styles"]) > 0
    }

@app.post("/onboarding/styles")
def set_styles(data: OnboardingRequest):
    MOCK_USER_PROFILE["selected_styles"] = data.styles
    return {"status": "success", "message": "Profile updated!"}

@app.get("/profile")
def get_user_profile():
    return {
        "user_id": MOCK_USER_PROFILE["user_id"],
        "email": "shopper@kiova.com",
        "selected_styles": MOCK_USER_PROFILE["selected_styles"]
    }

# ------------------------------------------
# 2. ADMIN: INVENTORY MANAGEMENT
# ------------------------------------------
@app.post("/analyze-image")
def analyze_image(file: UploadFile = File(...)):
    time.sleep(1.5)
    return {
        "status": "success",
        "ai_results": {
            "suggested_category": "Top",
            "suggested_sub_category": "T-Shirt",
            "suggested_color": "Blue",
            "suggested_pattern": "Solid",
            "suggested_style": "Casual",
            "suggested_tags": ["Cotton", "Casual", "Vintage"],
            "suggested_price": 25.00,
            "processed_image_url": "https://placehold.co/400x400/transparent/png?text=Clean+Image"
        }
    }


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """
    Simulates uploading a file to storage.
    Returns a fake URL.
    """
    # Simulate network delay
    time.sleep(1)
    
    # Return a fake URL
    return {"url": f"https://placehold.co/600x800/orange/white/png?text={file.filename}"}

@app.post("/products", status_code=201)
def add_product(product: ProductBase):
    new_item = product.dict()
    new_item["id"] = str(uuid.uuid4())
    MOCK_PRODUCTS.append(new_item)
    return new_item

@app.put("/products/{product_id}")
def update_product(product_id: str, updates: ItemUpdate):
    product = next((p for p in MOCK_PRODUCTS if p['id'] == product_id), None)
    if not product: raise HTTPException(404, "Product not found")
    update_data = updates.dict(exclude_unset=True)
    product.update(update_data)
    return product

@app.delete("/products/{product_id}")
def delete_product(product_id: str):
    global MOCK_PRODUCTS
    MOCK_PRODUCTS = [p for p in MOCK_PRODUCTS if p['id'] != product_id]
    return {"status": "deleted"}

# ------------------------------------------
# 3. MY WARDROBE
# ------------------------------------------
@app.post("/wardrobe/upload")
def upload_to_wardrobe(file: UploadFile = File(...)):
    time.sleep(2) 
    new_item = {
        "id": "wardrobe_" + str(uuid.uuid4())[:8],
        "category": "Top", 
        "sub_category": "T-Shirt", 
        "color": "Red",            
        "pattern": "Graphic",      
        "tags": ["Cotton", "Red", "Graphic", "Vintage"],
        "style": "Vintage", 
        "image_url": "https://placehold.co/400x400/transparent/png?text=My+Clean+Tee",
        "date_added": datetime.now().isoformat()
    }
    MOCK_USER_WARDROBE.append(new_item)
    return new_item

@app.get("/wardrobe", response_model=List[WardrobeItemResponse])
def get_my_wardrobe(category: Optional[str] = None):
    if category:
        return [i for i in MOCK_USER_WARDROBE if i['category'].lower() == category.lower()]
    return MOCK_USER_WARDROBE

@app.put("/wardrobe/{item_id}")
def update_wardrobe_item(item_id: str, updates: ItemUpdate):
    item = next((i for i in MOCK_USER_WARDROBE if i['id'] == item_id), None)
    if not item: raise HTTPException(404, "Item not found")
    item.update(updates.dict(exclude_unset=True))
    return item

@app.delete("/wardrobe/{item_id}")
def delete_from_wardrobe(item_id: str):
    global MOCK_USER_WARDROBE
    MOCK_USER_WARDROBE = [i for i in MOCK_USER_WARDROBE if i['id'] != item_id]
    return {"status": "deleted"}

# ------------------------------------------
# 4. STYLE ME & MARKETPLACE
# ------------------------------------------
@app.post("/style-me")
def style_me(data: StyleMeRequest):
    source_item = None
    if data.wardrobe_item_id:
        source_item = next((i for i in MOCK_USER_WARDROBE if i['id'] == data.wardrobe_item_id), None)
    elif data.product_id:
        source_item = next((i for i in MOCK_PRODUCTS if i['id'] == data.product_id), None)
    
    if not source_item:
        source_item = {"id": "temp", "category": "Top", "style": "Streetwear", "image_url": "..."}

    # Match Logic
    matches = []
    for market_item in MOCK_PRODUCTS:
        if market_item.get('id') == source_item.get('id'): continue
        
        is_compatible = False
        s_cat = source_item['category']
        m_cat = market_item['category']
        
        if s_cat == "Top" and m_cat in ["Bottom", "Shoes", "Outerwear"]: is_compatible = True
        elif s_cat in ["Bottom", "Pants"] and m_cat in ["Top", "Shoes", "Outerwear"]: is_compatible = True
        elif s_cat == "Shoes" and m_cat in ["Top", "Bottom", "Outerwear"]: is_compatible = True
            
        if is_compatible: matches.append(market_item)

    return {
        "user_item": source_item,
        "styled_matches": matches,
        "style_tip": f"Matching {source_item.get('style', 'cool')} vibes."
    }

# NEW: Save/Delete outfits
@app.post("/outfits")
def save_outfit(data: SaveOutfitRequest):
    new_outfit = {
        "id": "outfit_" + str(uuid.uuid4())[:8],
        "name": data.name,
        "items": data.product_ids, # IDs of items in the outfit
        "created_at": datetime.now().isoformat()
    }
    MOCK_SAVED_OUTFITS.append(new_outfit)
    return {"status": "success", "outfit_id": new_outfit["id"]}

@app.get("/outfits")
def get_saved_outfits():
    return MOCK_SAVED_OUTFITS

@app.delete("/outfits/{outfit_id}")
def delete_saved_outfit(outfit_id: str):
    """Remove a saved outfit."""
    global MOCK_SAVED_OUTFITS
    initial_len = len(MOCK_SAVED_OUTFITS)
    MOCK_SAVED_OUTFITS = [o for o in MOCK_SAVED_OUTFITS if o['id'] != outfit_id]
    
    if len(MOCK_SAVED_OUTFITS) == initial_len:
        raise HTTPException(status_code=404, detail="Outfit not found")
        
    return {"status": "deleted", "outfit_id": outfit_id}

# SEARCH AND FEED
@app.get("/products", response_model=List[ProductResponse])
def get_marketplace_feed(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    user_styles = MOCK_USER_PROFILE["selected_styles"]
    filtered_list = MOCK_PRODUCTS

    # 1. Category Filter
    if category:
        filtered_list = [p for p in filtered_list if p['category'].lower() == category.lower()]

    # 2. Search Filter (Name or Tags)
    if search:
        q = search.lower()
        filtered_list = [
            p for p in filtered_list 
            if q in p['name'].lower() or any(q in t.lower() for t in p['tags'])
        ]

    # 3. Sort by Style Match
    sorted_feed = sorted(
        filtered_list, 
        key=lambda x: x.get('style') in user_styles, 
        reverse=True
    )
    return sorted_feed

@app.get("/products/{product_id}", response_model=ProductResponse)
def get_product_details(product_id: str):
    product = next((p for p in MOCK_PRODUCTS if p['id'] == product_id), None)
    if not product: raise HTTPException(404, "Product not found")
    return product

@app.get("/ai-picks")
def get_ai_picks():
    user_styles = MOCK_USER_PROFILE["selected_styles"]
    picks = [p for p in MOCK_PRODUCTS if p.get('style') in user_styles]
    if not picks: picks = MOCK_PRODUCTS[:3]
    return picks[:5]

# ------------------------------------------
# 5. CART & ORDERS
# ------------------------------------------
@app.get("/cart")
def get_cart():
    total = sum(item['price'] * item['quantity'] for item in MOCK_CART)
    return {"items": MOCK_CART, "total_price": round(total, 2)}

@app.post("/cart")
def add_to_cart(item: CartItemRequest):
    product = next((p for p in MOCK_PRODUCTS if p['id'] == item.product_id), None)
    if not product: raise HTTPException(404, "Product not found")
    
    existing = next((i for i in MOCK_CART if i['id'] == item.product_id), None)
    if existing: existing['quantity'] += item.quantity
    else:
        MOCK_CART.append({
            "id": product['id'], "name": product['name'], "price": product['price'], 
            "image_url": product['image_url'], "quantity": item.quantity
        })
    return {"status": "added", "cart_size": len(MOCK_CART)}

@app.delete("/cart/{product_id}")
def remove_from_cart(product_id: str):
    global MOCK_CART
    MOCK_CART = [i for i in MOCK_CART if i['id'] != product_id]
    return {"status": "removed"}

@app.post("/checkout")
def checkout(data: CheckoutRequest):
    if not MOCK_CART: raise HTTPException(400, "Cart is empty")
    time.sleep(1.5)
    
    total = sum(item['price'] * item['quantity'] for item in MOCK_CART)
    new_order = {
        "order_id": "ORD-" + str(uuid.uuid4())[:8].upper(),
        "date": datetime.now().isoformat(),
        "status": "Paid",
        "total_amount": round(total, 2),
        "items": MOCK_CART.copy(),
        "shipping_address": data.shipping_address
    }
    MOCK_ORDERS.append(new_order)
    MOCK_CART.clear()
    return {"status": "success", "order_id": new_order["order_id"]}

@app.get("/orders")
def get_orders():
    return sorted(MOCK_ORDERS, key=lambda x: x['date'], reverse=True)

# ================= RUNNER =================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)