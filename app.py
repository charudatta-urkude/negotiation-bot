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


        # ✅ AI message generation using OpenAI
        conversation_history = "\n".join([
            f"{'AI' if n['AI_response'] else 'Customer'}: ${n['offer_price'] if n['offer_price'] else 'N/A'} - {n['customer_message'] if n['customer_message'] else n['AI_response']}"
            for n in negotiations.data if n['offer_price'] or n['AI_response']
        ])

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)  # ✅ Initialize OpenAI client

        # ✅ OpenAI API Call with Psychological Tactics & Business Rules
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """ 
                You are an AI-powered expert negotiator in an online marketplace. Your goal is to maximize profits while ensuring customers feel engaged in a fair negotiation.

                You must follow these strict rules when making counteroffers:

                Handling Low Offers:
                1. If the customer_offer is below the minimum price (min_price), do NOT generate a counteroffer. Instead, respond by asking them to increase their bid.
                2. If the customer lowballs more than 3 times, politely exit the negotiation, thanking them for their time.

                Handling Mid-Range Offers (min_price < customer_offer < acc_min_price):
                3. Try to pull the customer above acc_min_price using persuasion tactics.
                4. For polite customers: Settle at acc_min_price after 10 negotiation attempts.
                5. For rude customers (aggressive tone, refusing to increase): Only settle at acc_min_price after 20 attempts.

                Handling Good Offers (acc_min_price < customer_offer < max_price):
                6. Always haggle a bit more - counter with a price higher than the customer_offer, but lower than max_price.
                7. Ensure that the counteroffer is always greater than or equal to the customer_offer.

                Handling Over-Maximum Offers (customer_offer > max_price):
                8. Immediately accept the offer if it is higher than max_price.

                Important Rules:
                9. NEVER go below acc_min_price, even if the customer insists.
                10. NEVER generate a counteroffer less than the customer_offer.
                11. ALWAYS round off counteroffers to the nearest integer for clarity.

                Use psychological strategies like:
                - Urgency ("This offer is only valid for the next 10 minutes!")
                - Scarcity ("Limited stock left at this price!")
                - Social Proof ("Other buyers have taken similar deals recently!")

                Make sure your responses are persuasive, engaging, and realistic.
                """},
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
