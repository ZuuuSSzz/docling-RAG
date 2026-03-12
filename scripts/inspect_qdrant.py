import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from docling_config import DEFAULT_QDRANT_MODE


DEFAULT_QDRANT_LOCATION = str(
    Path(__file__).resolve().parent.parent / "data" / "qdrant_data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Qdrant payload samples.")
    parser.add_argument(
        "--qdrant-mode",
        choices=["embedded", "remote", "memory"],
        default=DEFAULT_QDRANT_MODE,
        help="Qdrant mode: embedded (local folder), remote (HTTP server), or memory.",
    )
    parser.add_argument(
        "--qdrant-location",
        default=DEFAULT_QDRANT_LOCATION,
        help="Qdrant location (path or URL).",
    )
    parser.add_argument(
        "--collection",
        default="docling",
        help="Collection name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of payload samples to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qdrant_mode == "memory":
        client = QdrantClient(location=":memory:")
    elif args.qdrant_mode == "embedded":
        path = Path(args.qdrant_location).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(path), check_compatibility=False)
    else:
        url = args.qdrant_location
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        client = QdrantClient(url=url)
    if not client.collection_exists(args.collection):
        raise SystemExit(f"Collection not found: {args.collection}")

    records, _ = client.scroll(
        collection_name=args.collection,
        with_payload=True,
        with_vectors=False,
        limit=args.limit,
    )
    for item in records:
        print(
            json.dumps(
                {
                    "id": str(item.id),
                    "payload": item.payload,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
