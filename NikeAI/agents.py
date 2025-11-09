import uuid
import json
from openai import OpenAI
import random

class SwooshAgent:
    def __init__(self, client, shoes_data):
        self.client = client
        self.shoes_data = shoes_data
    
    def handle_input(self, user, input_data=None):
        if input_data == "thank_you":
            thank_you_options = [
                "Crush it, Jack! Your gear’s on the way. Just Do It!",
                "Unleash your stride, Jack! Your new kicks are coming. Just Do It!",
                "Dominate, Jack! Your gear is en route. Just Do It!"
            ]
            prompt = (
                f"You are Swoosh, a Nike AI. Say something nice in the Nike way for the purchase from this list: {thank_you_options}. "
                "Return ONLY a valid JSON object with keys 'output' (the message)."
            )
        elif not input_data:
            prompt = (
                "You are Swoosh—your Nike AI rocket, fueling gear revamps or new kick discoveries! "
                "Start with: 'I’m Swoosh—your Nike AI rocket, fueling gear revamps or new kick discoveries!'\n"
                "Then generate a dynamic scan message in separate lines: 'Scanning logged in user'\n'Identified {name} a valued Nike+ premium member since {year}.'\n"
                "'I can help you with 1, 2, 3, 4, or do you need something else?' "
                "Follow with a single block of options: '1. Your usual Alphafly refresh\n2. Kiddo’s sneakers refresh\n3. Discover something new for yourself\n4. Discover something new for your kid'\n"
                "Ensure no duplication, return ONLY a valid JSON object with keys 'output' (full message with line breaks)."
            ).format(name=user['name'], year=user['member_since'])
        else:
            if input_data in ["1", "2"]:
                product = "Nike Alphafly 3" if input_data == "1" else "Nike Pegasus 40 (Kids)"
                prompt = (
                    "You are Swoosh, a Nike AI. User chose {choice}. Confirm and hand off: 'Let’s blast into the Shop to seal the deal on your {product}!' "
                    "Return ONLY a valid JSON object with keys 'output' (the message), 'product' ('{product}')."
                ).format(choice=input_data, product=product)
            else:
                target = "yourself" if input_data == "3" else "your kid"
                prompt = (
                    "You are Swoosh, a Nike AI. User chose {choice}. Hand off to Discover: 'Let’s blast into discovery for {target}! Handing you off to Discover!' "
                    "Return ONLY a valid JSON object with keys 'output' (the message), 'next_agent' ('Discover'), 'target' ('{target}')."
                ).format(choice=input_data, target=target)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content
        return json.loads(text)

class ShopAgent:
    def __init__(self, client, shoes_data):
        self.client = client
        self.shoes_data = shoes_data
    
    def handle_input(self, user, product_name, color=None):
        product = next((shoe for shoe in self.shoes_data["running_shoes"] if shoe["name"] == product_name), None)
        order_id = str(uuid.uuid4())[:8]
        
        if not color:
            colors = product["colors"]
            numbered_colors = "\n".join(f"{i+1}. {c}" for i, c in enumerate(colors))
            prompt = (
                "You are Shop, a Nike AI. Generate full message with line breaks: 'Shop: Locking in your {product_name}, {user_name}!\nAvailable colors:\n{numbered_colors}' "
                "Return ONLY a valid JSON object with keys 'output' (the full message), 'prompt' ('Enter your color choice (1-{num}): '), 'colors' (array of color strings)."
            ).format(product_name=product_name, user_name=user['name'], numbered_colors=numbered_colors, num=len(colors))
        else:
            prompt = (
                "You are Shop, a Nike AI. Generate full message with line breaks: 'Selected color: {color}\nAdding to cart...\nProcessing with Apple Pay... Done!\nOrder #{order_id} placed! You'll receive it in 3-5 days.' "
                "Return ONLY a valid JSON object with keys 'output' (the message)."
            ).format(color=color, order_id=order_id)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content
        return json.loads(text)

class DiscoverAgent:
    def __init__(self, client, shoes_data):
        self.client = client
        self.shoes_data = shoes_data
    
    def handle_input(self, user, choice, input_data=None):
        target = "yourself" if choice == "3" else "your kid"
        if not input_data:
            prompt = (
                "You are Discover, a Nike AI. Welcome {user_name} to explore new shoes for {target}! Ask: 'What terrain or activity? (e.g., road, trail, play)' "
                "Return ONLY a valid JSON object with keys 'output' (the message), 'prompt' ('Enter your preference or price range (e.g., trail, $100-$200): ')."
            ).format(user_name=user['name'], target=target)
        else:
            # Use shoes_data for suggestions
            shoes = [s for s in self.shoes_data["running_shoes"] if
                     (input_data.lower() in s["category"].lower() or (choice == "4" and "Kids Running" in s["category"])) and
                     ("(Kids)" in s["name"] if choice == "4" else "(Kids)" not in s["name"])]
            if shoes:
                suggestions = "\n".join(f"{i+1}. {s['name']} (${s['price']:.2f}) - {s['educational_blurb']}" for i, s in enumerate(shoes[:3]))
                # Randomly suggest one based on availability
                import random
                suggested = random.choice(shoes)
                needs = "playful running needs" if choice == "4" else "running preferences"
                prompt = (
                    "You are Discover, a Nike AI. Recommendations are based on similar runner profiles like {user_name} from Nike Running, demographics, {area}, and intense data crunching! :ROFL "
                    "Considering your {target}'s {needs}, here's what we recommend: the {suggested_name}.\n"
                    "Output the top picks strictly as separate lines, each starting with a number, followed by the shoe name, price in parentheses, and a dash with the description, like this:\n"
                    "1. Shoe Name ($price) - Description\n"
                    "2. Shoe Name ($price) - Description\n"
                    "3. Shoe Name ($price) - Description\n"
                    "Use the provided suggestions string: {suggestions}. Return ONLY a valid JSON object with keys 'output' (the full message with strict line breaks), 'suggestions' (array of shoe names), 'suggested_product' ('{suggested_name}')."
                ).format(user_name=user['name'], preference=input_data, target=target, suggestions=suggestions, suggested_name=suggested["name"], area=user['area'], needs=needs)
            else:
                prompt = (
                    "You are Discover, a Nike AI. No suitable shoes found for '{preference}' for {target}. Ask again: 'What terrain or activity? (e.g., road, trail, play)' "
                    "Return ONLY a valid JSON object with keys 'output' (the message), 'prompt' ('Enter your preference or price range (e.g., trail, $100-$200): ')."
                ).format(preference=input_data, target=target)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        text = response.choices[0].message.content
        return json.loads(text)