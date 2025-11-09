import json

def load_shoes_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: shoes.json not found!")
        return {"running_shoes": []}