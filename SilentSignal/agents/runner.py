from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.drone_analyst_agent import DroneAnalystAgent
from agents.silence_mapper_agent import SilenceMapperAgent


def run() -> None:
    parser = argparse.ArgumentParser(description="SilentSignal local agent runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    silence = subparsers.add_parser("run-silence", help="Run Silence Mapper on signals.csv")
    silence.add_argument("--signals", required=True, help="Path to signals.csv")

    drone = subparsers.add_parser("run-drone", help="Run Drone Analyst on media URIs")
    drone.add_argument("--zone-id", required=True)
    drone.add_argument("--media-uri", action="append", required=True)
    drone.add_argument("--aws-mode", default="mock", choices=["mock", "auto", "aws"])
    drone.add_argument("--region", default="us-east-1")
    drone.add_argument("--model-id", default="amazon.nova-pro-v1:0")

    args = parser.parse_args()

    if args.command == "run-silence":
        agent = SilenceMapperAgent()
        zones = agent.analyze(Path(args.signals))
        print(json.dumps(zones, indent=2))
        return

    if args.command == "run-drone":
        agent = DroneAnalystAgent(
            aws_mode=args.aws_mode,
            region=args.region,
            model_id=args.model_id,
        )
        zone = {
            "zone_id": args.zone_id,
            "geometry": {"type": "bbox", "bbox": [-95.37, 29.74, -95.33, 29.78]},
        }
        media = [{"media_uri": uri} for uri in args.media_uri]
        events = agent.analyze_zone_media(zone=zone, media_records=media)
        print(json.dumps(events, indent=2))


if __name__ == "__main__":
    run()
