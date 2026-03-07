# EC2 Single-Server Deploy (Hackathon)

This project can run in two modes:

- `mock` mode: fully local data flow, no AWS resources required.
- `aws/auto` mode: Bedrock/S3/DynamoDB capable when credentials and resources exist.

For the fastest and most reliable judge demo, deploy in `mock` mode unless you specifically need live Bedrock responses.

## 1) Launch EC2

- AMI: Amazon Linux 2023
- Instance type: `t3.micro` (or larger for smoother demo)
- Key pair: create/download `.pem`

Security group inbound rules:

- `22` (SSH): `My IP` only (recommended)
- `8000` (Custom TCP): `0.0.0.0/0`

Notes:

- Port `80` is optional and only needed if you put Nginx in front of Uvicorn.
- For rule `8000`, source must be `0.0.0.0/0` (or your allowed CIDR), not another security-group ID, if judges access from the public internet.

## 2) IAM Role (Only if you want Bedrock/AWS mode)

If you keep `.env` as `AWS_MODE=mock`, IAM role is not required for core demo flow.

If you want Bedrock-enabled chat/tooling:

- Attach an EC2 IAM role with Bedrock invoke permissions.
- `AmazonBedrockFullAccess` works for hackathon speed (broad permissions).

## 3) Copy + Install

On your laptop:

```bash
chmod 400 ~/Downloads/silentsignal-key.pem
scp -i ~/Downloads/silentsignal-key.pem -r /path/to/SilentSignal ec2-user@EC2_PUBLIC_IP:~/SilentSignal
ssh -i ~/Downloads/silentsignal-key.pem ec2-user@EC2_PUBLIC_IP
```

On EC2:

```bash
sudo dnf install -y python3.11 python3.11-pip git screen
cd ~/SilentSignal
python3.11 -m pip install -r requirements.txt
cp .env.example .env
mkdir -p data/generated data/s3_mock
```

## 4) Configure `.env`

### Reliable demo setup (recommended)

```env
AWS_MODE=mock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0
S3_BUCKET=silent-signal-media
DYNAMODB_TABLE=silent-signal-mvp
LOCAL_DYNAMO_PATH=data/generated/mock_dynamo.json
LOCAL_S3_DIR=data/s3_mock
SIGNALS_CSV_PATH=data/generated/signals.csv
MEDIA_MANIFEST_PATH=data/generated/media_manifest.json
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*
```

## 5) Start API

```bash
cd ~/SilentSignal
screen -S silentsignal
python3.11 -m uvicorn services.api.app.main:app --host 0.0.0.0 --port 8000
```

Detach from screen: `Ctrl+A`, then `D`.

## 6) Share URL

- `http://EC2_PUBLIC_IP:8000`
- App redirects to `/dashboard/`

## Quick sanity checks

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"question":"start monitoring"}'
```

If chat returns Bedrock warnings in logs, the app still runs and falls back to local command routing.
