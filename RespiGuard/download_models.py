import os
from huggingface_hub import snapshot_download, login

# UNCOMMENT AND SET YOUR TOKEN VIA ENV VAR IF NEEDED:
# login(token=os.environ.get("HF_TOKEN"))

print("⬇️  Downloading Google HeAR (Audio)...")
snapshot_download(repo_id="google/hear", local_dir="models/hear")

print("⬇️  Downloading MedGemma 4B (Medical Text)...")
snapshot_download(repo_id="google/medgemma-1.5-4b-it", local_dir="models/medgemma")

print("✅ Downloads Complete!")
