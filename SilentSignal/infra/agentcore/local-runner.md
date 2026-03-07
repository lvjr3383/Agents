# Bedrock AgentCore Local Runner Notes

Use this flow while preparing for AgentCore deployment:

1. Start FastAPI:
   - `make api`
2. Run Silence Mapper directly:
   - `PYTHONPATH=. python3 -m agents.runner run-silence --signals data/generated/signals.csv`
3. Run Drone Analyst directly:
   - `PYTHONPATH=. python3 -m agents.runner run-drone --zone-id Z-001 --media-uri s3://silent-signal-media/media/drone_a_shout.mp4`

When switching to managed runtime, keep agent entrypoints and data contracts unchanged.
