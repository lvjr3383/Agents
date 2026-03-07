# SilentSignal MVP Scaffold

SilentSignal is an AWS hackathon MVP for disaster response decision support.

This scaffold demonstrates three flows end-to-end in **mock mode** (no AWS credentials required):
- `SilenceMapperAgent`: detects negative-space "silence zones" from synthetic cellular/social activity.
- `DroneAnalystAgent`: evaluates zone media and labels likely signs of life.
- Dashboard: zone map/list, zone detail, and chat-with-data responses with citations.

## Architecture

Signals -> Silence Mapper -> Zones -> Drone Analyst -> Events -> Dashboard/Chat

## Repository Layout

```text
/apps/dashboard            # Dashboard UI (static HTML/CSS/JS)
/services/api              # FastAPI backend
/agents                    # Strands-style agent modules + local runner
/infra                     # AgentCore and hosting stubs
/data                      # Synthetic data generator + generated artifacts
```

Deployment note:
- EC2 single-server runbook: `infra/ec2-single-server.md`

## Data Contracts

`SilenceZone`
- `zone_id`
- `geometry` (bbox + cell_ids)
- `start_time`
- `end_time`
- `drop_score` (0-1)
- `confidence` (0-1)
- `reason_codes[]`

`DroneEvent`
- `event_id`
- `zone_id`
- `media_uri`
- `event_type` (`SHOUT|TAP_PATTERN|HUMAN_FORM|UNKNOWN`)
- `time_offset_start`
- `time_offset_end`
- `likelihood` (0-1)
- `notes`

## Quick Start (Mock Mode)

1. Create environment file:
   - `cp .env.example .env`
2. Install dependencies:
   - `make setup`
3. Generate synthetic data:
   - `make generate-data`
4. Run automated verification:
   - `make verify`
5. Start API:
   - `make api`
6. In a second terminal, start dashboard static server (optional):
   - `make dashboard`

Open either:
- `http://localhost:8000/dashboard/` (served by FastAPI)
- `http://localhost:5173/?api=http://localhost:8000` (standalone dashboard server)

## Verification Command

`make verify` runs:
- syntax checks (`make lint`)
- synthetic data generation (`make generate-data`)
- mock-mode end-to-end test (`make test`)

Use this command before each demo recording session.

If `make verify` fails with `No module named fastapi` or `uvicorn not found`, your current virtual environment is missing project dependencies. Run:
- `make setup`

## Run Analysis Flow

1. In chat, click/type `Start monitoring`.
2. In chat, click/type `Show all zones`.
3. Select `Z-...` zone in the zone list.
4. Click `Analyze Drone Media` or ask chat: `Analyze drone footage for Z-001`.
5. Ask chat prompts like:
   - `Which zone is highest priority?`
   - `Show top event for Z-001`

Chat output always includes citations, for example:
- `Citations: zone_id=Z-001, event_id=E-003`

## AWS Mode (Optional)

Set `.env`:
- `AWS_MODE=aws` (or `auto`)
- `AWS_REGION`
- `S3_BUCKET`
- `DYNAMODB_TABLE`
- `BEDROCK_MODEL_ID`

If AWS clients or credentials are missing, code falls back to local mock interfaces.

## API Endpoints

- `POST /generate-data`
- `POST /run-analysis`
- `GET /zones`
- `GET /zones/{zone_id}`
- `POST /zones/{zone_id}/analyze-drone`
- `POST /chat`
- `GET /health`

## Demo Script (<5 min)

1. Show architecture slide/diagram (5 seconds).
2. Open dashboard and ask chat to `Start monitoring`.
3. Ask chat to `Show all zones`; show ranked zones and map overlays.
4. Open top zone and run drone analysis.
5. Show new events (`SHOUT`, `TAP_PATTERN`, `HUMAN_FORM`).
6. Ask chat: `Which zone should we search first and why?`
7. Read response with citation IDs.

## Notes

- This scaffold intentionally prioritizes demo reliability over production hardening.
- No live telco ingest or live drone stream is included in this MVP.
