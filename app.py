import os
import uuid
import random
import logging
import asyncio
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client
from datetime import datetime
import openai
import json
from cachetools import TTLCache
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict

# --- Environment Setup & Global Configurations ---
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # Change to "gpt-4-turbo" if needed
MAX_TOKENS = 50

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

# Caches for pricing data and AI responses
pricing_cache = TTLCache(maxsize=100, ttl=600)
response_cache = TTLCache(maxsize=100, ttl=300)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://localhost:5173",
        "http://localhost:3000",
        "https://ai-negotiation-jqzi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class StartSessionRequest(BaseModel):
    user_id: str
    product_id: str

class OfferRequest(BaseModel):
    session_id: str
    customer_message: str
    decision: Optional[str] = None  # "deal" or "no_deal"

# --- Helper: Run Supabase Query ---
async def run_query(query_lambda):
    return await asyncio.to_thread(query_lambda)

def normalize_response_data(response):
    data = response.data
    if isinstance(data, list):
        return data[0] if len(data) == 1 else data
    return data

# --- Asynchronous Supabase Helpers ---
async def fetch_session_data(session_id: str):
    session_response = await run_query(lambda: supabase.table("history")
                                       .select("*")
                                       .eq("session_id", session_id)
                                       .order("created_at", desc=True)
                                       .limit(1)
                                       .execute())
    return normalize_response_data(session_response)

async def get_pricing_data(product_id: str):
    if product_id in pricing_cache:
        return pricing_cache[product_id]
    pricing_response = await run_query(lambda: supabase.table("pricing")
                                         .select("*")
                                         .eq("product_id", product_id)
                                         .execute())
    data = normalize_response_data(pricing_response)
    if data:
        pricing_cache[product_id] = data
        return data
    raise HTTPException(status_code=404, detail="Product pricing not found")

# --- Helper: Build Conversation History ---
async def build_conversation_history(session_id: str, limit: int = 5) -> List[Dict[str, str]]:
    response = await run_query(lambda: supabase.table("history")
                                 .select("customer_message", "ai_response")
                                 .eq("session_id", session_id)
                                 .order("created_at", desc=True)
                                 .limit(limit)
                                 .execute())
    messages = []
    if response.data:
        for entry in reversed(response.data):
            if entry.get("customer_message"):
                messages.append({"role": "user", "content": entry.get("customer_message")})
            if entry.get("ai_response"):
                messages.append({"role": "assistant", "content": entry.get("ai_response")})
    return messages

# --- Asynchronous OpenAI API Calls with Caching ---
async def extract_offer_intent_async(customer_message: str) -> dict:
    extraction_prompt = f"""
Extract the numerical offer and negotiation intent from the following message:
- Customer message: "{customer_message}"

If the customer's message consists solely of affirmative words (e.g., "yes", "okay", "sure", "I agree") and does not contain any numerical value, then return the intent as "affirmative" and the extracted_offer as null.

Use the following guidelines:
1. "final_offer": Use this if the customer states a definitive price and indicates that it is final (e.g., "My final offer is 850").
2. "discount_request": Use this if the customer explicitly asks for a discount.
3. "hesitation": Use this if the customer shows uncertainty or hesitation (e.g., "I'm not sure", "Let me think").
4. "affirmative": Use this if the customer agrees to the last counter-offer without specifying a new number. Examples:
    - "Yes"
    - "Okay"
    - "Sure"
    - "I agree"
    - "Let's go with this"
    - "I'll take it"
    - similar affirmative words
5. "negotiate": Use this if the customer is open to further discussion (e.g., "Can I get it at 800?").
6. "other": Use this if none of the above apply.

Return a valid JSON object exactly in the following format:
{{
    "extracted_offer": null,
    "intent": "other"
}}
Replace null with the numerical offer if found, and "other" with the detected intent.
"""

    try:
        response = await run_query(lambda: client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=extraction_prompt,
            temperature=0.2,
            max_tokens=MAX_TOKENS
        ))
        result = json.loads(response.choices[0].text.strip())
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in extracting intent/offer: {e}")
        result = {"extracted_offer": None, "intent": "other"}
    except Exception as e:
        logging.error(f"Error in extracting intent/offer: {e}")
        result = {"extracted_offer": None, "intent": "other"}
    
    logging.info(f"Extracted intent: {result}")  # Log for debugging
    return result

async def generate_ai_response_async(customer_message: str, extracted_offer: float, counter_offer: float, round_number: int, intent: str, deal_status: str, conversation_history: List[Dict[str, str]] = None) -> str:
    cache_key = f"{customer_message}-{extracted_offer}-{counter_offer}-{round_number}-{intent}-{deal_status}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    if not extracted_offer or extracted_offer == 0:
        extra_instructions = (
            "Generate a friendly, natural response that acknowledges the customer's sentiment. Encourage further discussion without insisting on a number."
        )
    else:
        if intent == "final_offer":
            extra_instructions = (
                "Acknowledge the customer's final offer in a friendly manner, and clearly include the counter-offer exactly as provided."
            )
        elif intent == "discount_request":
            extra_instructions = (
                "Explain your counteroffer with a light-hearted tone, include the counter-offer exactly as provided."
            )
        elif intent == "hesitation":
            extra_instructions = (
                "Address the customer's hesitation warmly and casually, and include the counter-offer exactly as provided."
            )
        else:
            extra_instructions = (
                "Generate a natural, varied response to continue the negotiation. Include the counter-offer exactly as provided."
            )
    
    # Build initial messages with context
    messages = [{"role": "system", "content": "You are a negotiation bot that maintains context across multiple turns."}]
    if conversation_history:
        messages.extend(conversation_history)
    
    # Create the response prompt with strict instructions
    response_prompt = f"""
            You are a negotiation bot. Based on the following context, generate a final, humanlike response in a casual, friendly tone.
            Avoid repeating fixed templates or phrases. Use diverse and witty language while ensuring the counter-offer value remains exactly as: "{counter_offer}".
            The response must be complete within 50 tokens.

            Customer Message: "{customer_message}"
            Extracted Offer: {"no offer" if not extracted_offer else extracted_offer}
            Counter Offer: {counter_offer}
            Intent: {intent}
            Round: {round_number}

            Instructions: {extra_instructions}

            ***STRICT RULES:
            - The response must be complete within 50 tokens.
            - Use friendly, conversational, witty language, but don't overdo it. speak like a human
            - The response MUST include the counter-offer: "{counter_offer}".
            - Your final response must explicitly mention the counter-offer value exactly as: "{counter_offer}". Do not change this number.
            - DO NOT GENERATE YOUR OWN OFFER. DO NOT GIVE ANY OFFER OTHER THAN THE COUNTER OFFER: "{counter_offer}" ***

            Final Response:
            """
    messages.append({"role": "user", "content": response_prompt})
    
    response_generation = await run_query(lambda: client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=MAX_TOKENS
    ))
    result = response_generation.choices[0].message.content.strip()
    response_cache[cache_key] = result
    return result

# --- Rule-Based Negotiation Logic ---
class RuleBasedNegotiation:
    def __init__(self, max_price, min_price, acc_min_price):
        self.max_price = max_price
        self.min_price = min_price
        self.acc_min_price = acc_min_price  # Acceptable minimum (profit threshold)
        self.counter_offer = max_price
        self.negotiation_rounds = 0
        self.lowball_rounds = 0
        self.last_offer = None
        self.last_counter = max_price
        self.consecutive_small_increases = 0
        self.discount_ceiling = 0.3 * self.max_price
        self.total_discount_given = 0
        #self.urgency_trigger_round = random.randint(4, 6)
        

    def generate_counteroffer(self, customer_offer, intent="normal"):
        self.negotiation_rounds += 1
        if self.last_offer is None:
            self.last_offer = self.max_price
        if self.last_counter is None:
            self.last_counter = self.max_price

        # Special handling for discount_request intent:
        if intent == "discount_request":
            if self.consecutive_small_increases < 2:
                self.consecutive_small_increases += 1
                discount_factor = random.uniform(0.0033, 0.005)  # 0.33% to 0.5%
                discount_amount = discount_factor * self.last_counter
                self.total_discount_given += discount_amount
                new_counter_offer = max(self.last_counter - discount_amount, self.acc_min_price, customer_offer)
                new_counter_offer = min(new_counter_offer, self.last_counter)
                new_counter_offer = round(new_counter_offer, 1)
                self.last_offer = customer_offer
                self.last_counter = new_counter_offer
                return new_counter_offer
            else:
                return self.last_counter

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

         # Compute the change from the previous valid offer
        del_offer = customer_offer - self.last_offer if self.last_offer is not None else 0
        logging.info(f"Round {self.negotiation_rounds} - del_offer: {del_offer}")
        if del_offer < 0:
            # Return the message while leaving previous values unchanged.
            logging.warning("Offer cannot be lower than the previous bid")
            return "invalid_offer"

        del_counter = 0
        offer_increase_percentage = (del_offer / self.last_offer) * 100 if self.last_offer else 0
        if offer_increase_percentage < 2:
            self.consecutive_small_increases += 1
        else:
            self.consecutive_small_increases = 0

        if self.consecutive_small_increases >= 2:
                disc_factor = random.uniform(0.03, 0.05)
                minimum_discount = disc_factor * self.last_counter
                del_counter = minimum_discount

        else:
            if offer_increase_percentage >= 5:
                del_counter = 0.9 * del_offer
            elif 2 <= offer_increase_percentage < 5:
                del_counter = 0.8 * del_offer
            elif 1 <= offer_increase_percentage < 2:
                del_counter = 0.7 * del_offer
            else:
                del_counter = 0.6 * del_offer

        del_counter = min(del_counter, 0.05 * self.max_price)
        if self.total_discount_given + del_counter > self.discount_ceiling:
            del_counter = self.discount_ceiling - self.total_discount_given

        self.total_discount_given += del_counter
        new_counter_offer = max(self.last_counter - del_counter, self.acc_min_price, customer_offer)
        new_counter_offer = min(new_counter_offer, self.last_counter)
        new_counter_offer = round(new_counter_offer, 1)

        if abs(self.last_counter - customer_offer) < 0.01 * self.max_price:
            return customer_offer

        if self.consecutive_small_increases >= 4:
            return "final_decision"

        self.last_offer = customer_offer
        self.last_counter = new_counter_offer
        return new_counter_offer

# --- Updated /start_negotiation Endpoint ---
@app.post("/start_negotiation")
async def start_negotiation(request: StartSessionRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    try:
        pricing_data = await get_pricing_data(request.product_id)
        max_price = pricing_data["max_price"]
        min_price = pricing_data["min_price"]
        acc_min_price = pricing_data["acc_min_price"]

        existing_session = await run_query(lambda: supabase.table("history")
                                           .select("session_id")
                                           .eq("user_id", request.user_id)
                                           .eq("product_id", request.product_id)
                                           .limit(1).execute())
        if existing_session.data:
            session_id = existing_session.data[0]["session_id"]
        else:
            session_id = str(uuid.uuid4())
            insert_response = await run_query(lambda: supabase.table("history").insert([{
                "session_id": session_id,
                "user_id": request.user_id,
                "product_id": request.product_id,
                "customer_offer": 0,                    # Log initial customer offer as 0
                "counter_offer": pricing_data["max_price"],  # Set initial counter offer as max_price
                "round_number": 0,
                "lowball_rounds": 0,
                "consecutive_small_increases": 0,
                "deal_status": "pending",
                "intent": "initial",
                "created_at": datetime.utcnow().isoformat()
            }]).execute())
            logging.info(f"Insert response in /start_negotiation: {insert_response}")

        background_tasks.add_task(logging.info, f"/start_negotiation took {time.perf_counter() - start_time:.3f} seconds")
        return {"session_id": session_id, "message": f"Welcome to the negotiation for product {request.product_id}."}
    except Exception as e:
        logging.error(f"Error in /start_negotiation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

# --- Updated /negotiate Endpoint ---
@app.post("/negotiate")
async def negotiate(offer: OfferRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    try:
        last_deal_status = await run_query(lambda: supabase.table("history")
                                             .select("deal_status", "lowball_rounds")
                                             .eq("session_id", offer.session_id)
                                             .order("created_at", desc=True)
                                             .limit(1)
                                             .execute())
        if last_deal_status.data and not offer.decision:
            deal_status = last_deal_status.data[0].get("deal_status")
            lowball_rounds = last_deal_status.data[0].get("lowball_rounds", 0) or 0
            if deal_status in ["success", "failed", "final_decision"] or lowball_rounds > 3:
                logging.info("Negotiation closed. No further offers allowed.")
                return {"status": "failed", "message": "Negotiation closed. No further offers allowed."}
        
        session_data = await fetch_session_data(offer.session_id)
        if not session_data:
            raise HTTPException(status_code=403, detail="Invalid session ID")
        user_id = session_data["user_id"]
        product_id = session_data["product_id"]

        pricing_data = await get_pricing_data(str(product_id))
        max_price = pricing_data["max_price"]
        min_price = pricing_data["min_price"]
        acc_min_price = pricing_data["acc_min_price"]

        negotiator = RuleBasedNegotiation(max_price, min_price, acc_min_price)

        conversation_history = await build_conversation_history(offer.session_id, limit=5)

        # Retrieve previous negotiation state from history
        last_negotiation = await run_query(lambda: supabase.table("history")
                                             .select("customer_offer", "counter_offer", "round_number", "consecutive_small_increases")
                                             .eq("session_id", offer.session_id)
                                             .order("created_at", desc=True)
                                             .limit(1)
                                             .execute())
        if last_negotiation.data:
            raw_offer = last_negotiation.data[0].get("customer_offer")
            last_offer = float(raw_offer) if raw_offer is not None else None
            last_counter = float(last_negotiation.data[0].get("counter_offer") or max_price)
            last_round_number = int(last_negotiation.data[0].get("round_number") or 0)
            prev_consec = int(last_negotiation.data[0].get("consecutive_small_increases") or 0)
        else:
            last_offer = None
            last_counter = max_price
            last_round_number = 0
            prev_consec = 0

        round_number = last_round_number + 1

        if offer.decision and offer.decision in ["deal", "no_deal"]:
            if offer.decision == "deal":
                final_message = f"Great! The deal is locked in at {last_counter}. Thank you for negotiating!"
                final_status = "success"
            else:
                final_message = "No deal made. Thank you for trying, maybe next time!"
                final_status = "failed"
            await run_query(lambda: supabase.table("history").insert([{
                "session_id": offer.session_id,
                "user_id": user_id,
                "product_id": product_id,
                "round_number": round_number,
                "customer_offer": last_offer,
                "counter_offer": last_counter,
                "lowball_rounds": prev_consec,
                "consecutive_small_increases": prev_consec,
                "deal_status": final_status,
                "intent": "final_decision",
                "customer_message": offer.customer_message,
                "ai_response": final_message,
                "created_at": datetime.utcnow().isoformat()
            }]).execute())
            return {"status": final_status, "human_response": final_message, "counter_offer": last_counter, "round_number": round_number}

        extracted_data = await extract_offer_intent_async(offer.customer_message)
        extracted_offer = extracted_data.get("extracted_offer")
        intent = extracted_data.get("intent", "normal")
        try:
            extracted_offer = float(extracted_offer) if extracted_offer is not None else None
        except ValueError:
            logging.error("Failed to convert extracted offer to float.")
            extracted_offer = None

        if intent == "affirmative" and (extracted_offer is None or extracted_offer == 0):
            human_response = f"Your response seems affirmative. Would you like to lock in the deal at {last_counter}?"
            counter_offer = "final_decision"
            await run_query(lambda: supabase.table("history").insert([{
                    "session_id": offer.session_id,
                    "user_id": user_id,
                    "product_id": product_id,
                    "round_number": round_number,
                    "customer_offer": 0,
                    "counter_offer": last_counter,
                    "lowball_rounds": negotiator.lowball_rounds,
                    "consecutive_small_increases": negotiator.consecutive_small_increases,
                    "deal_status": "final_decision",
                    "intent": intent,
                    "customer_message": offer.customer_message,
                    "ai_response": human_response,
                    "created_at": datetime.utcnow().isoformat()
                }]).execute())
            return {"status": "final_decision", "message": human_response, "counter_offer": "final_decision"}

        if extracted_offer is None or extracted_offer == 0:
            logging.info("Extracted offer is 0 or None, substituting with previous offer.")
            extracted_offer = last_offer

        negotiator.last_offer = last_offer
        negotiator.last_counter = last_counter
        negotiator.consecutive_small_increases = prev_consec

        if last_deal_status.data:
            previous_lowball = last_deal_status.data[0].get("lowball_rounds", 0)
            negotiator.lowball_rounds = int(previous_lowball) if previous_lowball is not None else 0

        counter_offer = negotiator.generate_counteroffer(extracted_offer, intent)
        if counter_offer == "invalid_offer":
            message = "Offer cannot be lower than the previous bid"
            await run_query(lambda: supabase.table("history").insert([{
                "session_id": offer.session_id,
                "user_id": user_id,
                "product_id": product_id,
                "round_number": round_number,
                "customer_offer": last_offer,
                "counter_offer": last_counter,
                "lowball_rounds": negotiator.lowball_rounds,
                "consecutive_small_increases": negotiator.consecutive_small_increases,
                "deal_status": "invalid_offer",
                "intent": intent,
                "customer_message": offer.customer_message,
                "ai_response": message,
                "created_at": datetime.utcnow().isoformat()
            }]).execute())
            return {"status": "invalid_offer", "message": message, "counter_offer": last_counter, "round_number": round_number}

        if counter_offer == "final_decision":
            if offer.decision and offer.decision in ["deal", "no_deal"]:
                if offer.decision == "deal":
                    final_message = f"Great! The deal is locked in at {negotiator.last_counter}. Thank you for negotiating!"
                    final_status = "success"
                else:
                    final_message = "No deal made. Thank you for trying, maybe next time!"
                    final_status = "failed"
                await run_query(lambda: supabase.table("history").insert([{
                    "session_id": offer.session_id,
                    "user_id": user_id,
                    "product_id": product_id,
                    "round_number": round_number,
                    "customer_offer": float(extracted_offer),
                    "counter_offer": negotiator.last_counter,
                    "lowball_rounds": negotiator.lowball_rounds,
                    "consecutive_small_increases": negotiator.consecutive_small_increases,
                    "deal_status": final_status,
                    "intent": intent,
                    "customer_message": offer.customer_message,
                    "ai_response": final_message,
                    "created_at": datetime.utcnow().isoformat()
                }]).execute())
                return {"status": final_status, "human_response": final_message, "counter_offer": negotiator.last_counter, "round_number": round_number}
            else:
                return {"status": "final_decision", "message": f"My final offer is {negotiator.last_counter}. Would you like to secure the deal?", "counter_offer": negotiator.last_counter}

        if counter_offer is None:
            await run_query(lambda: supabase.table("history").insert([{
                "session_id": offer.session_id,
                "user_id": user_id,
                "product_id": product_id,
                "round_number": round_number,
                "customer_offer": last_offer,
                "counter_offer": last_counter,
                "lowball_rounds": negotiator.lowball_rounds,
                "consecutive_small_increases": negotiator.consecutive_small_increases,
                "deal_status": "terminated" if negotiator.lowball_rounds > 3 else "ongoing",
                "intent": intent,
                "customer_message": offer.customer_message,
                "ai_response": "",
                "created_at": datetime.utcnow().isoformat()
            }]).execute())
            if negotiator.lowball_rounds > 3:
                return {"status": "failed", "message": "Negotiation terminated due to too many low offers."}
            return {"status": "failed", "message": "Your offer is too low. Please make a reasonable offer to continue."}

        if (abs(last_counter - extracted_offer) / last_counter) <= 0.02:
            deal_status_computed = "success"
            counter_offer = extracted_offer
            human_response = f"Great! We have a deal at {counter_offer}. Looking forward to doing business with you!"
        else:
            deal_status_computed = "ongoing"
            human_response = await generate_ai_response_async(
                offer.customer_message, extracted_offer, counter_offer, round_number, intent, "ongoing", conversation_history
            )

        extracted_offer = float(extracted_offer)
        counter_offer = float(counter_offer)

        await run_query(lambda: supabase.table("history").insert([{
            "session_id": offer.session_id,
            "user_id": user_id,
            "product_id": product_id,
            "round_number": round_number,
            "customer_offer": extracted_offer,
            "counter_offer": counter_offer,
            "lowball_rounds": negotiator.lowball_rounds,
            "consecutive_small_increases": negotiator.consecutive_small_increases,
            "deal_status": deal_status_computed,
            "intent": intent,
            "customer_message": offer.customer_message,
            "ai_response": human_response,
            "created_at": datetime.utcnow().isoformat()
        }]).execute())
        background_tasks.add_task(logging.info, f"/negotiate took {time.perf_counter() - start_time:.3f} seconds")
        return {
            "status": deal_status_computed,
            "human_response": human_response,
            "counter_offer": counter_offer,
            "round_number": round_number
        }
    
    except Exception as e:
        logging.error(f"Error in /negotiate: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")
