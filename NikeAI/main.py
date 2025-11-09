import os
from dotenv import load_dotenv
from openai import OpenAI
from agents import SwooshAgent, ShopAgent, DiscoverAgent
from data import load_shoes_data
import time

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("Error: OPENAI_API_KEY not found in .env")
    exit(1)
client = OpenAI(api_key=api_key)

def main():
    print("Firing up your Nike AI! Just Do It!\n")
    shoes_data = load_shoes_data("shoes.json")
    user = {"name": "Jack Lakkapragada", "nike_plus": True, "member_since": 2011, "past_purchase": "Alphafly", "metrics": "long-distance", "area": "NYC"}
    
    swoosh = SwooshAgent(client, shoes_data)
    shop = ShopAgent(client, shoes_data)
    discover = DiscoverAgent(client, shoes_data)
    
    # Initial Swoosh welcome and options with visual feedback
    response = swoosh.handle_input(user)
    lines = response["output"].split('\n')
    print("[Swoosh]")
    intro_end = 0
    for i, line in enumerate(lines):
        if line.strip() and "1." in line:
            intro_end = i
            break
    # Intro block with visual feedback
    for line in lines[:intro_end]:
        if line.strip():
            if "Scanning" in line:
                print(line + "...", end="", flush=True)
                time.sleep(0.5)
                print("\r" + line)  # Overwrite with complete line
            else:
                print(line)
                time.sleep(0.5)
    # Options block
    options = "\n".join(lines[intro_end:])
    time.sleep(0.5)
    print(options + "\n")
    
    # Get user choice with exit option
    while True:
        user_input = input("Enter your choice (1, 2, 3, 4, or 'quit'): ").lower()
        if user_input in ["1", "2", "3", "4"]:
            break
        elif user_input == "quit" or user_input == "exit":
            print("[Swoosh]\nThanks for stopping by, Jack! Catch you next time. Just Do It!\n")
            return
        print("Invalid choice! Pick 1, 2, 3, 4, or 'quit'.\n")
    
    # Handle choice
    if user_input in ["1", "2"]:
        response = swoosh.handle_input(user, user_input)
        print("[Swoosh]\n" + response["output"] + "\n")
        product = response["product"]
        response = shop.handle_input(user, product)  # Color prompt
        print("[Shop]\n" + response["output"] + "\n")
        while True:
            color_input = input("Enter your color choice (1-" + str(len(response["colors"])) + ") or 'quit': ").lower()
            if color_input == "quit":
                print("[Shop]\nWrapping up, Jack! See you soon. Just Do It!\n")
                return
            elif color_input.isdigit() and 1 <= int(color_input) <= len(response["colors"]):
                color = response["colors"][int(color_input) - 1]
                break
            print("Invalid color choice! Pick a number from the list or 'quit'.\n")
        response = shop.handle_input(user, product, color)  # Checkout
        print("[Shop]\n" + response["output"] + "\n")
    elif user_input in ["3", "4"]:
        response = swoosh.handle_input(user, user_input)
        print("[Swoosh]\n" + response["output"] + "\n")
        preference = None
        while True:
            response = discover.handle_input(user, user_input, preference)
            print("[Discover]\n" + response["output"] + "\n")
            if "prompt" in response:
                pref_input = input("Enter your preference or price range (e.g., trail, $100-$200) or 'quit': ").lower()
                if pref_input == "quit":
                    print("[Discover]\nTaking a break, Jack! Come back anytime. Just Do It!\n")
                    return
                if pref_input not in ["road", "trail", "play", "soccer"]:  # Fallback for invalid input
                    print("[Discover]\nInvalid preference! Please try road, trail, play, or soccer.\n")
                    continue
                preference = pref_input
            elif "suggestions" in response:
                user_choice = input("Enter the number of your preferred shoe (1-3) or press Enter to accept our suggestion or 'quit': ").lower()
                if user_choice == "quit":
                    print("[Discover]\nStepping out, Jack! Hit me up later. Just Do It!\n")
                    return
                elif user_choice.strip() == "":
                    product = response["suggested_product"]
                elif user_choice.isdigit() and 1 <= int(user_choice) <= len(response["suggestions"]):
                    product = response["suggestions"][int(user_choice) - 1]
                else:
                    print("Invalid choice! Accepting our suggestion.\n")
                    product = response["suggested_product"]
                response = shop.handle_input(user, product)  # Color prompt
                print("[Shop]\n" + response["output"] + "\n")
                while True:
                    color_input = input("Enter your color choice (1-" + str(len(response["colors"])) + ") or 'quit': ").lower()
                    if color_input == "quit":
                        print("[Shop]\nWrapping up, Jack! See you soon. Just Do It!\n")
                        return
                    elif color_input.isdigit() and 1 <= int(color_input) <= len(response["colors"]):
                        color = response["colors"][int(color_input) - 1]
                        break
                    print("Invalid color choice! Pick a number from the list or 'quit'.\n")
                response = shop.handle_input(user, product, color)  # Checkout
                print("[Shop]\n" + response["output"] + "\n")
                break
            else:
                break
    
    # Swoosh thank-you and end
    response = swoosh.handle_input(user, "thank_you")
    if user_input in ["3", "4"]:
        print("[Swoosh]\nUnleash your kiddo's stride! Their new kicks are coming. Just Do It!\n")
    else:
        print("[Swoosh]\n" + response["output"] + "\n")

if __name__ == "__main__":
    main()