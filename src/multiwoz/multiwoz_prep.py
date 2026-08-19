"""
Download and prepare a sample from MultiWOZ 2.2.
Extracts dialogue turns with metadata into a flat CSV.
"""

import argparse
import json
import urllib.request
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2"
TRAIN_SHARDS = 17  # dialogues_001.json through dialogues_017.json


def download_dialogues(limit):
    """Download training dialogues from GitHub, up to `limit` dialogues."""
    dialogues = []
    for i in range(1, TRAIN_SHARDS + 1):
        url = f"{BASE_URL}/train/dialogues_{i:03d}.json"
        print(f"Fetching shard {i}/{TRAIN_SHARDS}...")
        data = json.loads(urllib.request.urlopen(url).read())
        dialogues.extend(data)
        if len(dialogues) >= limit:
            break
    return dialogues[:limit]


def extract_turns(dialogues):
    """Flatten dialogues into per-turn rows."""
    rows = []
    for dlg in dialogues:
        dialogue_id = dlg["dialogue_id"]
        services = dlg.get("services", [])
        domain_str = ", ".join(services) if services else "unknown"

        for turn in dlg["turns"]:
            # Try to get per-turn domain from frames
            turn_domain = domain_str
            if turn.get("frames"):
                frame_services = [f["service"] for f in turn["frames"] if f.get("service")]
                if frame_services:
                    turn_domain = ", ".join(sorted(set(frame_services)))

            rows.append({
                "dialogue_id": dialogue_id,
                "turn_index": int(turn["turn_id"]),
                "speaker": turn["speaker"].lower(),
                "domain": turn_domain,
                "utterance": turn["utterance"],
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare MultiWOZ 2.2 sample")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of dialogues to process (default: 20)")
    args = parser.parse_args()

    print(f"Downloading up to {args.limit} dialogues from MultiWOZ 2.2...")
    dialogues = download_dialogues(args.limit)
    print(f"Downloaded {len(dialogues)} dialogues")

    rows = extract_turns(dialogues)
    df = pd.DataFrame(rows)

    print(f"\nExtracted {len(df)} turns from {df['dialogue_id'].nunique()} dialogues")
    print(f"Speakers: {df['speaker'].value_counts().to_dict()}")
    print(f"Domains: {df['domain'].value_counts().head(10).to_dict()}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))

    df.to_csv("../../multiwoz/multiwoz_sample.csv", index=False)
    print(f"\nSaved multiwoz_sample.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
