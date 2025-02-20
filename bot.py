import logging
import random

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class RuleBasedNegotiation:
    def __init__(self, max_price, min_price, acc_min_price):
        self.max_price = max_price
        self.min_price = min_price
        self.acc_min_price = acc_min_price
        self.counter_offer = max_price  # Start counter near max price
        self.negotiation_rounds = 0
        self.lowball_rounds = 0
        self.last_offer = None
        self.last_counter = max_price
        self.consecutive_small_increases = 0
        self.discount_ceiling = 0.3 * self.max_price  # Prevent total discount from exceeding 30% of max_price
        self.total_discount_given = 0
        self.urgency_trigger_round = random.randint(4, 6)  # Random urgency round

    def generate_counteroffer(self, customer_offer):
        self.negotiation_rounds += 1

        # Determine del_offer safely
        if self.last_offer is None:
            del_offer = 0  # First round, no previous offer
        else:
            del_offer = customer_offer - self.last_offer  # Calculate increase

        # 🔹 **Fix 3: Prevent Decreasing Offer Exploit**
        # Line 66-68: If customer tries to reduce and increase offer to manipulate counteroffer, reset last_offer
        if del_offer < 0:
            print("Offer cannot be lower than the previous bid. Keeping last offer unchanged.")
            return self.last_counter  # Keep counteroffer unchanged

        del_counter = 0  # Default counter decrease
        del_counter = min(del_counter, 0.05 * self.max_price)  # Limit max discount per round


        logging.debug(f"Customer Offer: {customer_offer}, Last Offer: {self.last_offer}, Last Counter: {self.last_counter}")
        logging.debug(f"del_offer: {del_offer}")

        # Immediate Accept if Above Max Price
        if customer_offer >= self.max_price:
            print(f"Seller accepts the offer at {customer_offer}")
            return customer_offer

        # Reject & Warn for Lowball Offers
        if customer_offer < self.min_price:
            self.lowball_rounds += 1
            print(f"Offer too low. Please increase your bid. ({self.lowball_rounds}/3 warnings)")
            if self.lowball_rounds >= 3:
                print("Negotiation ended due to too many low offers.")
                return None
            return None

        # Detect Small Increment Patterns to Apply Patience Penalty
        offer_increase_percentage = (del_offer / self.last_offer) * 100 if self.last_offer else (del_offer / self.max_price) * 100

        if offer_increase_percentage < 2:
            self.consecutive_small_increases += 1
        else:
            self.consecutive_small_increases = 0  # Reset if a bigger increase happens

        # If customer makes many small jumps, reduce discount flexibility
        if self.consecutive_small_increases >= 3:
            print("Pattern detected: Small consecutive increases. Counteroffer flexibility reduced.")
            del_counter = 0.02 * del_offer  # Reduce price only slightly
        else:
            # Percentage-based Decrement Strategy (with protection against big-jump exploitation)
            if offer_increase_percentage >= 10:
                del_counter = 0.3 * del_offer
            elif 5 <= offer_increase_percentage < 10:
                del_counter = 0.2 * del_offer
            elif 2 <= offer_increase_percentage < 5:
                del_counter = 0.1 * del_offer
            else:
                del_counter = 0.05 * del_offer

        # Apply Discount Ceiling Protection
        if self.total_discount_given + del_counter > self.discount_ceiling:
            print("Maximum discount limit reached. Further reductions are limited.")
            del_counter = max(0, self.discount_ceiling - self.total_discount_given)

        self.total_discount_given += del_counter  # Track total discounts applied

        # Ensure counteroffer is within limits
        new_counter_offer = max(self.last_counter - del_counter, self.acc_min_price, customer_offer)
        new_counter_offer = min(new_counter_offer, self.last_counter)

        # 🔹 **Fix 1: Accept the Deal When Offer is Close**
        # Line 85-88: If customer offer and counteroffer are within 2% of max_price, close the deal at a final price
        offer_gap = abs(self.last_counter - customer_offer)
        if offer_gap < 0.01 * self.max_price:  # Within 1% → Accept directly
            print(f"Offer is very close! Seller accepts the deal at {customer_offer}.")
            return customer_offer
        #elif offer_gap < 0.02 * self.max_price:  # Within 2% → Offer a final discount
        #    final_price = (customer_offer + self.last_counter) / 2
        #    print(f"You're a tough negotiator! I'll give you a special final price: {final_price}")
        #    return final_price

        if self.negotiation_rounds >= self.urgency_trigger_round and del_offer > 0:
            print("Limited-Time Offer! Final Price:")
            new_counter_offer = max(self.acc_min_price, customer_offer)  # Offer the lowest acceptable price

        # Update last values for tracking
        self.last_offer = customer_offer
        self.last_counter = new_counter_offer

        logging.debug(f"offer_increase_percentage: {offer_increase_percentage}")
        logging.debug(f"del_counter: {del_counter}")
        logging.debug(f"New Counter Offer: {new_counter_offer} (Negotiation Round: {self.negotiation_rounds})")
        print(f"Counter Offer: {new_counter_offer} (Round: {self.negotiation_rounds})")
        return new_counter_offer

    def start_interactive_negotiation(self):
        print(f"\nNegotiation Started | Max Price: {self.max_price}, Min Price: {self.min_price}, Acceptable Min Price: {self.acc_min_price}\n")

        while True:
            try:
                # 🔹 **Fix 2: Allow Float Input Instead of Integer**
                # Line 129-131: Changed input parsing from int() to float()
                customer_offer = input("Enter your offer (or type 0 to exit): ")
                
                try:
                    customer_offer = float(customer_offer)
                except ValueError:
                    print("Invalid input! Please enter a valid number.")
                    continue

                if customer_offer == 0:
                    print("Negotiation ended. No deal was made.")
                    logging.info("Negotiation ended by customer.")
                    break

                counter_offer = self.generate_counteroffer(customer_offer)

                if self.lowball_rounds > 3:
                    print("Negotiation ended due to too many low offers.")
                    logging.warning("Negotiation ended - too many low offers")
                    break 

                if counter_offer is not None and counter_offer == customer_offer:
                    print(f"Deal closed at {customer_offer}")
                    logging.info(f"Deal finalized at {customer_offer}")
                    break

            except Exception as e:
                print("An error occurred. Please try again.")
                logging.error(f"Error: {e}")

# --- Run Interactive Negotiation ---
if __name__ == "__main__":
    seller = RuleBasedNegotiation(max_price=183.99, min_price=122.66, acc_min_price=149.65)
    seller.start_interactive_negotiation()
