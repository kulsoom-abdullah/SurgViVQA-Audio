import json
import collections
import re

def analyze_vocabulary(filepath, label):
    print(f"\n{'='*70}")
    print(f"Analyzing: {label}")
    print(f"{'='*70}")

    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extract all questions
    all_text = " ".join([item['question'] for item in data if 'question' in item])

    # Tokenize: convert to lowercase, extract words (only alphabetic)
    words = re.findall(r'\b[a-z]+\b', all_text.lower())

    # Filter words longer than 6 letters
    long_words = [w for w in words if len(w) > 6]

    # Count frequencies
    word_counts = collections.Counter(long_words)

    # Total statistics
    print(f"\nTotal words in questions: {len(words)}")
    print(f"Unique words: {len(set(words))}")
    print(f"Words longer than 6 letters: {len(long_words)}")
    print(f"Unique words longer than 6 letters: {len(word_counts)}")

    # Get least common (bottom 50)
    least_common = word_counts.most_common()[-50:]  # Get last 50 (least common)
    least_common.reverse()  # Show rarest first

    print(f"\n{'='*70}")
    print(f"Top 50 LEAST Common Words (>6 letters) - Medical Vocabulary Check")
    print(f"{'='*70}")
    print(f"{'Word':<25} {'Count':>10}")
    print(f"{'-'*70}")
    for word, count in least_common:
        print(f"{word:<25} {count:>10}")

    # Also show most common long words for context
    print(f"\n{'='*70}")
    print(f"Top 20 MOST Common Words (>6 letters) - For Context")
    print(f"{'='*70}")
    print(f"{'Word':<25} {'Count':>10}")
    print(f"{'-'*70}")
    for word, count in word_counts.most_common(20):
        print(f"{word:<25} {count:>10}")

    return word_counts

# Analyze both files
in_template_words = analyze_vocabulary("in_template.jsonl", "In-Template (Robotic)")
out_template_words = analyze_vocabulary("out_template.jsonl", "Out-Template (Natural)")

# Compare vocabularies
print(f"\n{'='*70}")
print(f"Vocabulary Comparison")
print(f"{'='*70}")

in_only = set(in_template_words.keys()) - set(out_template_words.keys())
out_only = set(out_template_words.keys()) - set(in_template_words.keys())
shared = set(in_template_words.keys()) & set(out_template_words.keys())

print(f"\nUnique to In-Template: {len(in_only)} words")
if in_only:
    print(f"Examples: {', '.join(list(in_only)[:10])}")

print(f"\nUnique to Out-Template: {len(out_only)} words")
if out_only:
    print(f"Examples: {', '.join(list(out_only)[:10])}")

print(f"\nShared between both: {len(shared)} words")
print(f"Total unique long words across both: {len(in_only) + len(out_only) + len(shared)}")
