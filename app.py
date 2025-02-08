from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import uuid
import logging
from supabase import create_client, Client
from datetime import datetime

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Supabase Client Initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials not found. Please set them.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ OpenAI API Initialization
openai.api_key = OPENAI_API_KEY

# ✅ Pydantic Models for API Requests
class StartNegotiationRequest(BaseModel):
    user_id: str
    product_id: str

class NegotiateRequest(BaseModel):
    session_id: str
    user_id: str
    product_id: str
    offer_price: Optional[float] = 0.0
    customer_message: Optional[str] = None

# ✅ Step 1: Start a Negotiation Session
@app.post("/start_negotiation")
async def start_negotiation(request: StartNegotiationRequest):
    try:
        # ✅ Fetch product details from Supabase
        product = supabase.table("pricing").select("*").eq("product_id", request.product_id).execute()
        if not product.data:
            raise HTTPException(status_code=404, detail="Product not found")

        # ✅ Check if session already exists for this user-product combination
        existing_session = supabase.table("negotiation").select("session_id").eq("user_id", request.user_id).eq("product_id", request.product_id).execute()
        if existing_session.data:
            session_id = existing_session.data[0]["session_id"]
        else:
            # ✅ Generate a new session_id
            session_id = str(uuid.uuid4())
            supabase.table("negotiation").insert({
                "session_id": session_id,
                "user_id": request.user_id,
                "product_id": request.product_id,
                "created_at": datetime.utcnow().isoformat()
            }).execute()

        return {
            "session_id": session_id,
            "message": f"Welcome to the negotiation for product {request.product_id}. Please make your first offer."
        }

    except Exception as e:
        logging.error(f"Error in /start_negotiation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Step 2: Process Negotiation Offers
@app.post("/negotiate")
async def negotiate(request: NegotiateRequest):
    try:
        # ✅ Fetch product pricing data
        product = supabase.table("pricing").select("*").eq("product_id", request.product_id).execute()
        if not product.data:
            raise HTTPException(status_code=404, detail="Product not found")

        product_data = product.data[0]

        # ✅ Debugging: Print retrieved pricing data
        print("DEBUG: Retrieved Product Data:", product_data)

        max_price = product_data.get('max_price', 0.0)
        min_price = product_data.get('min_price', 0.0)
        acc_min_price = product_data.get('acc_min_price', 0.0)  # ✅ Using correct variable name

        # ✅ Debugging: Print each variable value
        print(f"DEBUG: max_price = {max_price}, min_price = {min_price}, acc_min_price = {acc_min_price}")

        # ✅ Ensure values are valid before using them
        if max_price == 0.0 or min_price == 0.0 or acc_min_price == 0.0:
            raise HTTPException(status_code=500, detail=f"Pricing data is incomplete. max_price={max_price}, min_price={min_price}, acc_min_price={acc_min_price}")

        # ✅ Fetch past negotiations
        negotiations = supabase.table("negotiation").select("*").eq("session_id", request.session_id).eq("user_id", request.user_id).execute()
        
        # ✅ Debugging: Print past negotiation data
        print(f"DEBUG: negotiations.data = {negotiations.data}")

        last_offer = 0.0  # Default in case no previous offer exists
        if negotiations.data:
            last_offer = negotiations.data[-1].get('offer_price', 0.0)  # ✅ Ensuring no `NoneType`

        # ✅ Debugging: Print last offer
        print(f"DEBUG: last_offer = {last_offer}")

        # ✅ Ensure `request.offer_price` is not `None`
        offer_price = request.offer_price or 0.0
        print(f"DEBUG: request.offer_price = {offer_price}")

        # ✅ Determine AI's counteroffer safely
        counter_offer = ((last_offer or 0.0) + offer_price) / 2
        if counter_offer < acc_min_price:
            counter_offer = acc_min_price

        # ✅ AI message generation using OpenAI
        conversation_history = "\n".join([
            f"{'AI' if n['AI_response'] else 'Customer'}: ${n['offer_price'] if not n['AI_response'] else n['AI_response']} - {n['customer_message'] if not n['AI_response'] else n['AI_response']}"
            for n in negotiations.data
        ])

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)  # ✅ Initialize OpenAI client

        # ✅ OpenAI API Call with Psychological Tactics & Business Rules
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
                    "You are a negotiation AI for an e-commerce platform."
                    "You must maximize profit while keeping the customer engaged."
                    "Follow these rules:"
                    "\n1. **Never offer a price below the minimum acceptable price.**"
                    "\n2. **Never give a counteroffer less than customer offer.**"
                    "\n3. **Never accept the first customer offer immedieatly, offer a slightly higher counter offer, but within reasonable limits** "
                    "\n4. If a customer lowballs more than 3 times, politely exit the negotiation."
                    "\n5. Use psychological tactics to persuade the customer:"
                    "\n   - **Scarcity**: 'Only a few left at this price!'"
                    "\n   - **Urgency**: 'This deal expires soon!'"
                    "\n   - **Anchoring**: 'The original price was $X, you are getting a great deal at $Y.'"
                    "\n   - **Reciprocity**: 'As a returning customer, I can offer you something special.'"
                    "\n5. Keep responses short and directly related to negotiation."
                    "\n6. Never explain AI logic. Always sound human."
                },
                {"role": "user", "content": 
                    f"The customer offered ${request.offer_price}. Your last counteroffer was ${last_offer}."
                    f"Your new counteroffer is ${counter_offer}. Generate a concise response using negotiation techniques."
                }
            ],
            temperature=0.5,
            max_tokens=50  # ✅ Keeps responses short and relevant
        )

        AI_response = response.choices[0].message.content.strip()

        # ✅ Log negotiation in Supabase
        supabase.table("negotiation").insert({
            "session_id": request.session_id,
            "user_id": request.user_id,
            "product_id": request.product_id,
            "offer_price": offer_price,  # ✅ Using correct column name
            "counter_offer": counter_offer,
            "customer_message": request.customer_message,
            "AI_response": AI_response,  # ✅ Using correct column name
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return {
            "counter_offer": counter_offer,
            "AI_response": AI_response
        }

    except Exception as e:
        logging.error(f"Error in /negotiate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Step 3: Run API (For local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
