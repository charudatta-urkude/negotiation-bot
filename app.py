import os
import uuid
import random
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from supabase import create_client, Client
from datetime import datetime
import openai
import json
from cachetools import TTLCache

API_KEY = os.getenv("API_KEY", "your-secure-api-key")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = api_key_header):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OpenAI API key is missing. Please set it as an environment variable.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase credentials not found. Please set them.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

pricing_cache = TTLCache(maxsize=100, ttl=600)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174","http://localhost:5173", "http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_response(status: str, message: str, data: dict = None):
    return {"status": status, "message": message, "data": data or {}}

# --- Updated Model: product_id is now an integer ---
class StartSessionRequest(BaseModel):
    user_id: str
    product_id: str

class OfferRequest(BaseModel):
    session_id: str
    customer_message: str

async def fetch_session_data(session_id: str):
    return await asyncio.to_thread(
        supabase.table("history").select("*").eq("session_id", session_id).order("created_at", desc=True).limit(1).execute
    )

async def get_pricing_data(product_id: int):
    if product_id in pricing_cache:
        return pricing_cache[product_id]
    
    pricing_data = await asyncio.to_thread(
        supabase.table("pricing").select("*").eq("product_id", product_id).execute
    )
    
    if pricing_data.data:
        pricing_cache[product_id] = pricing_data.data[0]
        return pricing_data.data[0]
    
    raise HTTPException(status_code=404, detail="Product pricing not found")

# --- AI-Powered Extraction & Intent Detection ---
def extract_offer_intent(customer_message: str) -> dict:
    extraction_prompt = f"""
Extract the numerical offer and negotiation intent from the following message:
- Customer message: "{customer_message}"

Provide a structured JSON response:
{{
  "extracted_offer": <value_or_null>,
  "intent": <"final_offer", "discount_request", "hesitation", "negotiate", "other">
}}
    """
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=extraction_prompt,
        temperature=0.2,
        max_tokens=50
    )
    try:
        result = json.loads(response.choices[0].text.strip())
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in extracting intent/offer: {e}")
        result = {"extracted_offer": None, "intent": "other"}
    except Exception as e:
        logging.error(f"Error in extracting intent/offer: {e}")
        result = {"extracted_offer": None, "intent": "other"}
    return result

def generate_ai_response(customer_message: str, extracted_offer: float, counter_offer: float, round_number: int, intent: str, deal_status: str) -> str:
    if deal_status == "success":
        extra_instructions = "Generate a positive, congratulatory message confirming the deal at the agreed price."
    elif deal_status == "terminated":
        extra_instructions = "Generate a message expressing regret that the negotiation has failed."
    else:
        extra_instructions = "Your goal is to continue negotiating and provide a persuasive counteroffer."

    response_prompt = f"""
    Customer: "{customer_message}"
    Extracted Offer: {extracted_offer if extracted_offer else 'no offer'}
    Correct Counter Offer: {counter_offer}
    Intent: {intent}
    Current round is {round_number}

    {extra_instructions}

    STRICT RULES:
    - If the deal is SUCCESS, create a cheerful and professional congratulatory message.
    - If the deal is TERMINATED, create a polite but firm regret message.
    - If negotiation is ONGOING, follow regular negotiation tactics.
    - Never modify the counter-offer price. The correct response MUST include: "{counter_offer}".
    """

    response_generation = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": response_prompt}],
        temperature=0.6,
        max_tokens=50
    )

    return response_generation.choices[0].message.content.strip()

class RuleBasedNegotiation:
    def __init__(self, max_price, min_price, acc_min_price):
        self.max_price = max_price
        self.min_price = min_price
        self.acc_min_price = acc_min_price
        self.counter_offer = max_price
        self.negotiation_rounds = 0
        self.lowball_rounds = 0
        self.last_offer = None
        self.last_counter = max_price
        self.consecutive_small_increases = 0
        self.discount_ceiling = 0.3 * self.max_price
        self.total_discount_given = 0
        self.urgency_trigger_round = random.randint(4, 6)

    def generate_counteroffer(self, customer_offer):
        self.negotiation_rounds += 1

        if self.last_offer is None:
            self.last_offer = self.max_price
        if self.last_counter is None:
            self.last_counter = self.max_price

        if customer_offer >= self.max_price:
            logging.info(f"Customer offered {customer_offer}, meeting/exceeding max price. Accepting deal.")
            return customer_offer

        if customer_offer < self.min_price:
            self.lowball_rounds += 1
            logging.warning(f"Customer lowballed: {customer_offer}, Warning {self.lowball_rounds}/3.")
            if self.lowball_rounds > 3:
                logging.warning("Negotiation terminated due to too many rejected lowball offers.")
                return None
            return None

        del_offer = customer_offer - self.last_offer if self.last_offer is not None else 0

        # Use logging.warning instead of print for consistency
        if del_offer < 0:
            logging.warning("Offer cannot be lower than the previous bid. Keeping last offer unchanged.")
            return self.last_counter

        del_counter = 0  
        offer_increase_percentage = (del_offer / self.last_offer) * 100 if self.last_offer else 0

        if offer_increase_percentage < 2:
            self.consecutive_small_increases += 1
        else:
            self.consecutive_small_increases = 0

        if self.consecutive_small_increases >= 3:
            del_counter = 0.02 * del_offer
        else:
            if offer_increase_percentage >= 10:
                del_counter = 0.3 * del_offer
            elif 5 <= offer_increase_percentage < 10:
                del_counter = 0.2 * del_offer
            elif 2 <= offer_increase_percentage < 5:
                del_counter = 0.1 * del_offer
            else:
                del_counter = 0.05 * del_offer

        del_counter = min(del_counter, 0.05 * self.max_price)

        if self.total_discount_given + del_counter > self.discount_ceiling:
            del_counter = self.discount_ceiling - self.total_discount_given

        self.total_discount_given += del_counter

        new_counter_offer = max(self.last_counter - del_counter, self.acc_min_price, customer_offer)
        new_counter_offer = min(new_counter_offer, self.last_counter)
        new_counter_offer = round(new_counter_offer, 1)

        if abs(self.last_counter - customer_offer) < 0.01 * self.max_price:
            return customer_offer

        if self.negotiation_rounds >= self.urgency_trigger_round and self.consecutive_small_increases < 2:
            urgency_adjustment = (self.last_counter - self.acc_min_price) * 0.3
            new_counter_offer = max(self.acc_min_price, customer_offer, self.last_counter - urgency_adjustment)

        self.last_offer = customer_offer
        self.last_counter = new_counter_offer
        return new_counter_offer

@app.post("/start_negotiation")
async def start_negotiation(request: StartSessionRequest):
    try:
        product = supabase.table("pricing").select("max_price", "min_price", "acc_min_price")\
                    .eq("product_id", request.product_id).single().execute()
        
        if not product.data:
            raise HTTPException(status_code=404, detail="Product not found")

        existing_session = supabase.table("history").select("session_id")\
                    .eq("user_id", request.user_id)\
                    .eq("product_id", request.product_id).limit(1).execute()
        
        if existing_session.data:
            session_id = existing_session.data[0]["session_id"]
        else:
            session_id = str(uuid.uuid4())
            supabase.table("history").insert([{
                "session_id": session_id,
                "user_id": request.user_id,
                "product_id": request.product_id,
                "created_at": datetime.utcnow().isoformat()
            }]).execute()

        return {
            "session_id": session_id,
            "message": f"Welcome to the negotiation for product {request.product_id}."
        }
    except Exception as e:
        logging.error(f"Error in /start_negotiation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

@app.post("/negotiate")
async def negotiate(offer: OfferRequest):
    try:
        last_deal_status = supabase.table("history").select("deal_status", "lowball_rounds")\
                              .eq("session_id", offer.session_id).order("created_at", desc=True).limit(1).execute()

        if last_deal_status.data:
            deal_status = last_deal_status.data[0].get("deal_status")
            lowball_rounds = last_deal_status.data[0].get("lowball_rounds", 0) or 0

            if deal_status == "success" or lowball_rounds > 3:
                logging.info("Negotiation closed. No further offers allowed.")
                return {"status": "failed", "message": "Negotiation closed. No further offers allowed."}

        session_data = supabase.table("history").select("user_id", "product_id")\
                          .eq("session_id", offer.session_id).limit(1).execute()
        if not session_data.data:
            raise HTTPException(status_code=403, detail="Invalid session ID")

        user_id = session_data.data[0]["user_id"]  
        product_id = session_data.data[0]["product_id"]

        pricing_data = supabase.table("pricing").select("max_price", "min_price", "acc_min_price")\
                          .eq("product_id", product_id).single().execute()
        if not pricing_data.data:
            raise HTTPException(status_code=404, detail="Product pricing not found")

        max_price = pricing_data.data["max_price"]
        min_price = pricing_data.data["min_price"]
        acc_min_price = pricing_data.data["acc_min_price"]

        negotiator = RuleBasedNegotiation(max_price, min_price, acc_min_price)

        last_negotiation = supabase.table("history").select("customer_offer", "counter_offer", "round_number")\
                              .eq("session_id", offer.session_id).order("created_at", desc=True).limit(1).execute()

        if last_negotiation.data:
            last_offer = float(last_negotiation.data[0].get("customer_offer") or 0.0)
            last_counter = float(last_negotiation.data[0].get("counter_offer") or max_price)
            last_round_number = int(last_negotiation.data[0].get("round_number") or 0)
        else:
            last_offer = 0.0
            last_counter = max_price
            last_round_number = 0

        round_number = last_round_number + 1

        extracted_data = extract_offer_intent(offer.customer_message)
        extracted_offer = extracted_data.get("extracted_offer")
        intent = extracted_data.get("intent", "normal")

        if extracted_offer is None:
            return {"status": "failed", "message": "No valid offer detected. Please provide a clear price."}

        negotiator.last_offer = last_offer
        negotiator.last_counter = last_counter

        if last_deal_status.data:
            previous_lowball = last_deal_status.data[0].get("lowball_rounds", 0)
            negotiator.lowball_rounds = int(previous_lowball) if previous_lowball is not None else 0

        counter_offer = negotiator.generate_counteroffer(extracted_offer)

        if counter_offer is None:
            supabase.table("history").insert([{
                "session_id": offer.session_id,
                "user_id": user_id,
                "product_id": product_id,
                "round_number": round_number,
                "customer_offer": float(extracted_offer),
                "counter_offer": None,
                "lowball_rounds": negotiator.lowball_rounds,
                "deal_status": "terminated" if negotiator.lowball_rounds > 3 else "ongoing",
                "created_at": datetime.utcnow().isoformat()
            }]).execute()

            if negotiator.lowball_rounds > 3:
                return {"status": "failed", "message": "Negotiation terminated due to too many low offers."}
            return {"status": "failed", "message": "Your offer is too low. Please make a reasonable offer to continue."}

        # Determine deal status using a tolerance (0.1)
        deal_status_computed = "success" if abs(counter_offer - extracted_offer) < 0.1 else "ongoing"

        human_response = generate_ai_response(offer.customer_message, extracted_offer, counter_offer, round_number, intent, deal_status_computed)

        extracted_offer = float(extracted_offer)
        counter_offer = float(counter_offer)

        supabase.table("history").insert([{
            "session_id": offer.session_id,
            "user_id": user_id,
            "product_id": product_id,
            "round_number": round_number,
            "customer_offer": extracted_offer,
            "counter_offer": counter_offer,
            "lowball_rounds": negotiator.lowball_rounds,
            "deal_status": deal_status_computed,
            "created_at": datetime.utcnow().isoformat()
        }]).execute()

        return {
            "status": deal_status_computed,
            "human_response": human_response,
            "counter_offer": counter_offer,
            "round_number": round_number
        }

    except Exception as e:
        logging.error(f"Error in /negotiate: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")
