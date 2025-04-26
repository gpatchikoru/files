#!/usr/bin/env python
import json
import re
from collections import Counter
import argparse

def tokenize(text):
    """Lowercase & split on word characters (drops punctuation)."""
    return re.findall(r"\w+", text.lower())

def build_vocab(captions_json_path, output_vocab_path, max_tokens=10000):
    # 1) Load COCO captions
    with open(captions_json_path, 'r') as f:
        data = json.load(f)
    captions = [ann['caption'] for ann in data['annotations']]

    # 2) Count tokens
    counter = Counter()
    for cap in captions:
        counter.update(tokenize(cap))

    # 3) Select most common tokens
    most_common = [tok for tok, _ in counter.most_common(max_tokens)]
    
    # 4) Special tokens
    specials = ['[PAD]', '[START]', '[END]', '[UNK]']
    
    # 5) Build final vocab list (ensuring no dupes)
    vocab = specials + [t for t in most_common if t not in specials]
    
    # 6) Map to IDs and save
    token2id = {tok: idx for idx, tok in enumerate(vocab)}
    with open(output_vocab_path, 'w') as f:
        json.dump(token2id, f, indent=2)
    print(f"✅ Built vocab with {len(vocab)} tokens → saved to {output_vocab_path}")

def main():
    parser = argparse.ArgumentParser(description="Build COCO caption vocab")
    parser.add_argument(
        "--captions_json", required=True,
        help="Path to COCO captions_train2017.json"
    )
    parser.add_argument(
        "--output_vocab", default="vocab.json",
        help="Where to write token2id JSON"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=10000,
        help="How many top tokens to keep (after specials)"
    )
    args = parser.parse_args()
    build_vocab(args.captions_json, args.output_vocab, args.max_tokens)

if __name__ == "__main__":
    main()
