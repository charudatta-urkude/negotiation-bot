import os
import uuid
import random
import logging
import asyncio
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from supabase import create_client, Client
from datetime import datetime
import openai
import json
from cachetools import TTLCache
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, ValidationError

# --- Environment Setup & Global Configurations ---
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")  # Change to "gpt-4-turbo" if needed
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



async def extract_offer_intent_async(customer_message: str, last_offer: float) -> dict:
    # NEW MODIFICATION: Updated prompt with examples for each intent and additional factors.
    extraction_prompt = f"""
Extract detailed negotiation parameters from the following customer message:
- Customer message: "{customer_message}"

Your task is to extract the following fields:

1. extracted_offer: The absolute numerical offer mentioned in the message (if any).  
   - Example: "My final offer is 850" should yield extracted_offer = 850.
   - Example: "I can get it for 750" should yield extracted_offer = 750.
   
2. increment: If the customer indicates a relative increase, extract that numerical value as a positive number.
   - Example: "I can give 10 more" or "I can add 10" should yield increment = 10 (and effective_offer = last_offer + 10).
   - Example: "Add 5 more" should yield increment = 5.

3. decrement: If the customer indicates a relative decrease, extract that numerical value as a positive number.
   - Example: "I can only reduce it by 20" should yield decrement = 20.

4. quantity: If a bulk order is mentioned, extract the number of units.
   - Example: "2 for 1600" should yield quantity = 2.
   - If no quantity is mentioned, default to 1.

5. intent: Determine the negotiation intent using these categories:
***"affirmative": if the customer agrees to the last counter-offer without specifying a new number.****
         * Example: "Okay", "Sure", "Yes", "Works for me", "I'll take it", "Let's go with this."
   - "final_offer": if the customer states a definitive final price.
         * Example: "My final offer is 850."
   - "discount_request": if the customer explicitly asks for a discount.
         * Example: "Can you give me a discount? I can pay 800."
   - "hesitation": if the customer shows uncertainty or hesitation.
         * Example: "I'm not sure, maybe around 810?" or "I need to think about this."
   - "negotiate": if the customer is open to further discussion.
         * Example: "Can I get it at 800?" or "How about 820?" 
   - "other": if none of the above apply.
         * Example: "I'm interested, but not sure about the price."
   - Also include any cases not covered by these examples if needed.

6. tone: Identify the emotional tone of the message.
   - Examples: "aggressive" (e.g., "Your price is ridiculous!"), "emotional" (e.g., "I'm really desperate."), "neutral", or "friendly".

7. ambiguity: If the message is ambiguous, include a note explaining the ambiguity.
   - Example: "No clear number provided" if the message is vague.

8. competitor_offer: If the customer references a competitor’s price, extract that value.
   - Example: "Competitor X is offering 750" should yield competitor_offer = 750.

9. shipping_needed: If the message mentions shipping terms, extract that information.
   - Example: "If you include free shipping" should yield shipping_needed = true.
   - Example: "rush shipping" should capture that information (can be a boolean or descriptive string).

10. payment_terms: If the message includes conditions like installment or partial payment, extract that as a string.
    - Example: "I need to pay in installments" should yield payment_terms = "installments".

11. trade_in: If the customer mentions trading in an old item, return true.
    - Example: "I'll trade in my old phone" should yield trade_in = true.


12. effective_offer: Compute the effective offer as follows:
    - If an "increment" is provided, effective_offer = last_offer + increment.
         * Example: last_offer=800 and "I can give 10 more" results in effective_offer = 810.
    - Else if a "decrement" is provided, effective_offer = last_offer - decrement.
         * Example: last_offer=800 and "I can reduce my offer by 20" yields effective_offer = 780.
    - Else if "extracted_offer" is provided and quantity > 1, effective_offer = extracted_offer divided by quantity.
         * Example: "2 for 1600" yields effective_offer = 800.
    - Otherwise, effective_offer = last_offer.

Return a valid JSON object exactly in the following format:
{{
    "extracted_offer": null,
    "increment": 0.0,
    "decrement": 0.0,
    "quantity": 1,
    "intent": "other",
    "tone": "neutral",
    "ambiguity": "",
    "competitor_offer": null,
    "shipping_needed": false,
    "payment_terms": "",
    "trade_in": false,
    "other_factors": {{}},
    "effective_offer": null
}}
Replace null with the appropriate numerical value if found, and update the other fields accordingly.
    """
    
    try:
        response = await run_query(lambda: client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=extraction_prompt,
            temperature=0.2,
            max_tokens=300
        ))
        response_text = response.choices[0].text.strip()
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in extracting intent/offer: {e}")
        result = {
            "extracted_offer": None,
            "increment": 0.0,
            "decrement": 0.0,
            "quantity": 1,
            "intent": "other",
            "tone": "neutral",
            "ambiguity": "",
            "competitor_offer": None,
            "shipping_needed": False,
            "payment_terms": "",
            "trade_in": False,
            "other_factors": {},
            "effective_offer": None
        }
    except Exception as e:
        logging.error(f"Error in extracting intent/offer: {e}")
        result = {
            "extracted_offer": None,
            "increment": 0.0,
            "decrement": 0.0,
            "quantity": 1,
            "intent": "other",
            "tone": "neutral",
            "ambiguity": "",
            "competitor_offer": None,
            "shipping_needed": False,
            "payment_terms": "",
            "trade_in": False,
            "other_factors": {},
            "effective_offer": None
        }
    
    logging.info(f"Extracted details before effective_offer calculation: {result}")
    
    # NEW MODIFICATION: Compute effective_offer by checking increment/decrement first.
    try:
        if result.get("increment", 0) != 0:
            result["effective_offer"] = round(last_offer + float(result["increment"]), 1)
        elif result.get("decrement", 0) != 0:
            result["effective_offer"] = round(last_offer - float(result["decrement"]), 1)
        elif result.get("extracted_offer") is not None:
            qty = result.get("quantity", 1)
            result["effective_offer"] = round(float(result["extracted_offer"]) / qty, 1)
        else:
            result["effective_offer"] = last_offer
    except Exception as e:
        logging.error(f"Error calculating effective_offer: {e}")
        result["effective_offer"] = last_offer

    logging.info(f"Final extracted details: {result}")
    return result



async def generate_response_async(customer_message: str, extracted_details: dict, counter_offer: float, 
                                  round_number: int, conversation_history: list) -> str:
    system_message = (
        "You are a negotiation bot that generates friendly, casual, and engaging responses.You can occasionally be witty or humorous, but do not overdo it"
        "Your response must ALWAYS use the provided counter offer exactly as given. "
        "DO NOT generate any new offer. Your job is to acknowledge the customer's message and clearly present the counter offer. "
        "DO NOT use any currency symbol such as $, €, ₹ etc"
        "DO NOT include any smiling characters in the response"
        "Keep the tone light and natural, and limit your response to 50 tokens."
    )
    
    user_message = (
        f"Conversation History: {json.dumps(conversation_history)}\n\n"
        f"Customer Message: \"{customer_message}\"\n"
        f"Extracted Details: {json.dumps(extracted_details)}\n"
        f"Computed Counter Offer (from rules): {counter_offer}\n"
        "Based on the above, generate a final response that acknowledges the customer's message, "
        "clearly states the counter offer exactly as provided, and encourages further discussion if needed. "
        "Do NOT generate any offer other than the provided counter offer."
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = await run_query(lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6,
            max_tokens=50
        ))
        final_response = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        final_response = "I'm sorry, something went wrong. Could you please repeat your offer?"
    
    return final_response

# --------------------------------------------------------------------------------
# Rule-Based Negotiation Logic (including generate_counteroffer)
# --------------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------------
    # Updated generate_counteroffer function with additional parameters and modifications.
    # --------------------------------------------------------------------------------
    def generate_counteroffer(self, customer_offer, intent="normal", quantity=1, time_urgency=False, tone="neutral", 
                               competitor_offer=None, shipping_needed=False, payment_terms="", trade_in=False, other_factors=None):
        if other_factors is None:
            other_factors = {}
            
        self.negotiation_rounds += 1
        if self.last_offer is None:
            self.last_offer = self.max_price
        if self.last_counter is None:
            self.last_counter = self.max_price

        if intent == "affirmative":
            self.last_offer = customer_offer
            return "affirmative_decision"

        ## MODIFICATION 1 & Additional: Handle "negative" messages (e.g., "No, I cannot give at this price")
        if intent == "negative":
            minimal_discount = 0.01 * self.last_counter  # 0.2% minimal discount
            new_counter_offer = max(self.last_counter - minimal_discount, self.acc_min_price, customer_offer)
            new_counter_offer = round(new_counter_offer, 1)
            self.last_offer = customer_offer
            self.last_counter = new_counter_offer
            return new_counter_offer

        ## Existing branch for "discount_request" intent:
        if intent == "discount_request":
            if self.consecutive_small_increases < 2:
                self.consecutive_small_increases += 1
                discount_factor = random.uniform(0.03, 0.05) 
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

        del_offer = customer_offer - self.last_offer if self.last_offer is not None else 0
        logging.info(f"Round {self.negotiation_rounds} - del_offer: {del_offer}")
        if del_offer < 0:
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

        ## MODIFICATION 3: Adjust discount if time urgency is flagged.
        if time_urgency:
            del_counter *= 0.5

        ## MODIFICATION 4: Adjust discount based on conversation tone.
        if tone == "emotional":
            del_counter *= 1.1
        elif tone == "aggressive":
            del_counter *= 0.9

        del_counter = min(del_counter, 0.05 * self.max_price)
        if self.total_discount_given + del_counter > self.discount_ceiling:
            del_counter = self.discount_ceiling - self.total_discount_given

        self.total_discount_given += del_counter
        new_counter_offer = max(self.last_counter - del_counter, self.acc_min_price, customer_offer)
        new_counter_offer = min(new_counter_offer, self.last_counter)
        new_counter_offer = round(new_counter_offer, 1)

        ## MODIFICATION 2: Handle bulk/quantity orders.
        if quantity > 1:
            bulk_discount_factor = 0.02  # 2% discount per additional unit.
            additional_discount = self.last_counter * bulk_discount_factor * (quantity - 1)
            new_counter_offer = max(new_counter_offer - additional_discount, self.acc_min_price)
            new_counter_offer = round(new_counter_offer, 1)

        ## MODIFICATION 12: Apply randomization to avoid being too formulaic.
        random_variation = random.uniform(0.98, 1.02)
        new_counter_offer = max(min(new_counter_offer * random_variation, self.last_counter), self.acc_min_price)
        new_counter_offer = round(new_counter_offer, 1)

        ## NEW MODIFICATION: Adjust based on competitor_offer if provided.
        if competitor_offer is not None:
            if competitor_offer < new_counter_offer:
                new_counter_offer = min(new_counter_offer, round(competitor_offer * 0.98, 1))  # Undercut by 2%

        ## NEW MODIFICATION: Adjust for shipping needs.
        if shipping_needed:
            new_counter_offer = round(new_counter_offer * 1.05, 1)
            new_counter_offer = min(new_counter_offer, self.last_counter)

        ## NEW MODIFICATION: Adjust based on payment_terms.
        if payment_terms.lower() in ["installment", "partial payment"]:
            new_counter_offer = round(new_counter_offer * 1.05, 1)
            new_counter_offer = min(new_counter_offer, self.last_counter)

        ## NEW MODIFICATION: Adjust for trade_in scenarios.
        if trade_in:
            extra_discount = 0.03 * self.last_counter
            new_counter_offer = max(new_counter_offer - extra_discount, self.acc_min_price)
            new_counter_offer = round(new_counter_offer, 1)

        ## NEW MODIFICATION: Apply additional adjustments from other_factors.
        if "bonus_discount" in other_factors:
            bonus = other_factors["bonus_discount"]
            new_counter_offer = max(new_counter_offer - bonus, self.acc_min_price)
            new_counter_offer = round(new_counter_offer, 1)

        if abs(self.last_counter - customer_offer) <= 0.01 * self.max_price:
            return customer_offer

        if self.consecutive_small_increases >= 4:
            return "final_decision"

        self.last_offer = customer_offer
        self.last_counter = new_counter_offer
        return new_counter_offer

# --------------------------------------------------------------------------------
# Updated /start_negotiation Endpoint
# --------------------------------------------------------------------------------
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
                "customer_offer": 0,  # Log initial customer offer as 0
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

# --------------------------------------------------------------------------------
# Updated /negotiate Endpoint
# --------------------------------------------------------------------------------
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

        last_negotiation = await run_query(lambda: supabase.table("history")
                                             .select("customer_offer", "counter_offer", "round_number", "consecutive_small_increases")
                                             .eq("session_id", offer.session_id)
                                             .order("created_at", desc=True)
                                             .limit(1)
                                             .execute())
        if last_negotiation.data:
            raw_offer = last_negotiation.data[0].get("customer_offer")
            last_offer = float(raw_offer) if raw_offer is not None else max_price  ## MODIFICATION: default to max_price if None.
            last_counter = float(last_negotiation.data[0].get("counter_offer") or max_price)
            last_round_number = int(last_negotiation.data[0].get("round_number") or 0)
            prev_consec = int(last_negotiation.data[0].get("consecutive_small_increases") or 0)
        else:
            last_offer = max_price
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

        ## MODIFICATION: Use new extraction function.
        extracted_data = await extract_offer_intent_async(offer.customer_message, last_offer)
        intent_extracted = extracted_data.get("intent", "normal")
        extracted_offer = extracted_data.get("effective_offer")
        try:
            extracted_offer = float(extracted_offer) if extracted_offer is not None else last_offer
        except ValueError:
            logging.error("Failed to convert extracted offer to float.")
            extracted_offer = last_offer

        if intent_extracted == "affirmative" :
        #and (extracted_offer is None or extracted_offer == 0):
            human_response = f"Your response seems affirmative. Would you like to lock in the deal at {last_counter}?"
            counter_offer = "affirmative_decision"
            await run_query(lambda: supabase.table("history").insert([{
                "session_id": offer.session_id,
                "user_id": user_id,
                "product_id": product_id,
                "round_number": round_number,
                "customer_offer": last_offer,
                "counter_offer": last_counter,
                "lowball_rounds": negotiator.lowball_rounds,
                "consecutive_small_increases": negotiator.consecutive_small_increases,
                "deal_status": "final_decision",
                "intent": intent_extracted,
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

        ## MODIFICATION: Pass additional fields from extracted_data to generate_counteroffer.
        counter_offer = negotiator.generate_counteroffer(
            extracted_offer,
            intent=intent_extracted,
            quantity=extracted_data.get("quantity", 1),
            time_urgency=False,  ## Optionally, set based on context.
            tone=extracted_data.get("tone", "neutral"),
            competitor_offer=extracted_data.get("competitor_offer"),
            shipping_needed=extracted_data.get("shipping_needed"),
            payment_terms=extracted_data.get("payment_terms"),
            trade_in=extracted_data.get("trade_in"),
            other_factors=extracted_data.get("other_factors")
        )

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
                "intent": intent_extracted,
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
                    "intent": intent_extracted,
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
                "intent": intent_extracted,
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
            human_response = await generate_response_async(
                offer.customer_message, extracted_data, counter_offer, round_number, conversation_history
            )

        try:
            extracted_offer = float(extracted_offer)
            counter_offer = float(counter_offer)
        except Exception as e:
            logging.error(f"Conversion error: {e}")

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
            "intent": intent_extracted,
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
