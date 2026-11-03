"""Serve the evaluated RoboCasa policy adapter over OpenPI websocket."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import pathlib
from .pi05_adapter import Pi05Adapter
from .smol_adapter import SmolAdapter
from .server_common import metadata_default, write_metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pi05", "smolvla"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, choices=[1, 2, 4, 10, 15, 20], default=10)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata-path", help="Save resolved runtime configuration and normalization hashes")
    parser.add_argument("--skip-sampler-parity", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    adapter = (Pi05Adapter(args.checkpoint, args.steps, not args.skip_sampler_parity)
               if args.model == "pi05" else SmolAdapter(args.checkpoint, args.steps, args.device))
    adapter.metadata["adapter_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    # Websocket metadata uses msgpack, so materialize enum/path values as JSON scalars.
    adapter.metadata = json.loads(json.dumps(adapter.metadata, default=metadata_default))
    adapter.metadata_path = args.metadata_path
    if args.metadata_path:
        write_metadata(args.metadata_path, adapter.metadata)
    logging.info("CAMPAIGN_POLICY_READY %s", json.dumps(adapter.metadata, sort_keys=True, default=metadata_default))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(adapter, host=args.host, port=args.port, metadata=adapter.metadata).serve_forever()


if __name__ == "__main__":
    main()
