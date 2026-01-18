#!/usr/bin/env python3
"""
Analyze data distribution for SurgViVQA dataset
Outputs: docs/data_distribution.md (for README) and docs/data_stats.json (for app)

Based on SurgViVQA paper structure:
- 6 reasoning domains: Instruments, Sizing, Diagnosis, Positions, Operation Notes, Movement
- Short answers (1 word avg) vs Long answers (6.8 words avg)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# Map question types to reasoning domains from SurgViVQA paper (page 5)
REASONING_DOMAINS = {
    'Instruments': ['tool_presence', 'tool_identification', 'tool_count'],
    'Sizing': ['lesion_size', 'lesion_extent', 'lesion_count'],
    'Diagnosis': ['lesion_type', 'lesion_classification'],
    'Positions': ['anatomical_location', 'organ_identification', 'lesion_location'],
    'Operation Notes': ['occlusion', 'blue_dye_presence', 'flush_action',
                        'lighting_mode', 'nbi_status', 'blood_presence',
                        'fluid_occlusion_level', 'phase_identification'],
    'Movement': ['scope_motion', 'scope_forward_motion', 'lesion_motion_direction',
                 'scope_rotation']
}

# Build reverse mapping
QTYPE_TO_DOMAIN = {}
for domain, qtypes in REASONING_DOMAINS.items():
    for qt in qtypes:
        QTYPE_TO_DOMAIN[qt] = domain

def analyze_dataset(filepath):
    """Analyze question types, answer distributions, and reasoning domains"""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]

    # Question type distribution
    qtypes = Counter([s['question_type'] for s in data])

    # Reasoning domain distribution
    domain_counts = defaultdict(int)
    qtype_to_samples = defaultdict(list)

    for sample in data:
        qt = sample['question_type']
        domain = QTYPE_TO_DOMAIN.get(qt, 'Other')
        domain_counts[domain] += 1
        qtype_to_samples[qt].append(sample)

    # Answer type analysis (Yes/No vs Open-ended)
    answer_analysis = {}
    for qt, samples in qtype_to_samples.items():
        answers = [s.get('short_answer', s.get('answer', '')).lower().strip()
                  for s in samples]
        yes_count = sum(1 for a in answers if a == 'yes')
        no_count = sum(1 for a in answers if a == 'no')
        yes_no_total = yes_count + no_count

        answer_analysis[qt] = {
            'total': len(samples),
            'yes': yes_count,
            'no': no_count,
            'yes_no': yes_no_total,
            'open_ended': len(samples) - yes_no_total,
            'is_yes_no': yes_no_total > len(samples) * 0.5  # Mostly Yes/No if >50%
        }

    # Overall answer distribution
    all_answers = [s.get('short_answer', s.get('answer', '')).lower().strip()
                   for s in data]
    total_yes = sum(1 for a in all_answers if a == 'yes')
    total_no = sum(1 for a in all_answers if a == 'no')
    total_yes_no = total_yes + total_no

    return {
        'total_samples': len(data),
        'question_types': dict(qtypes),
        'reasoning_domains': dict(domain_counts),
        'answer_analysis': answer_analysis,
        'overall_answers': {
            'yes': total_yes,
            'no': total_no,
            'yes_no_total': total_yes_no,
            'open_ended': len(data) - total_yes_no
        }
    }

def generate_markdown_report(train_stats, eval_stats, test_stats):
    """Generate markdown report for README"""

    # Total samples
    total_samples = train_stats['total_samples'] + eval_stats['total_samples'] + test_stats['total_samples']

    md = """# Data Distribution

## Dataset Overview

**Source:** [SurgViVQA Dataset](https://github.com/madratak/SurgViVQA/) - A temporally-grounded surgical VQA dataset

I used a subset of the SurgViVQA dataset (3,700 samples from 4 video IDs). Audio was generated from text questions using edge-tts to simulate spoken queries.

### Terminology Clarification
- **Video ID:** Identifier for source colonoscopy procedure (e.g., 002-001, 002-002)
- **Sample:** Individual VQA instance (question + 8 frames + answer)
- **Frames:** Temporal sequence of 8 frames extracted from video at specific timestamps

---

## Dataset Splits

| Split | Samples | Video IDs | Purpose |
|-------|---------|-----------|---------|
| **Train** | {:,} | 002-001, 002-002, 002-003 | Model training |
| **Eval** | {:,} | 002-001, 002-002, 002-003 | Validation during training |
| **Test** | {:,} | 002-004 (held-out) | Final generalization test |
| **Total** | **{:,}** | 4 colonoscopy videos | 20 question types |

---

## Reasoning Domains

Following the SurgViVQA paper structure (6 domains):

""".format(
        train_stats['total_samples'],
        eval_stats['total_samples'],
        test_stats['total_samples'],
        total_samples
    )

    # Domain breakdown (using training set as representative)
    md += "| Domain | Count | Percentage | Description |\n"
    md += "|--------|-------|------------|-------------|\n"

    domain_descriptions = {
        'Instruments': 'Tool presence, identification, counting',
        'Sizing': 'Lesion size, extent, counting',
        'Diagnosis': 'Lesion type, classification',
        'Positions': 'Anatomical location, spatial reasoning',
        'Operation Notes': 'Visibility, lighting, dye, occlusion',
        'Movement': 'Scope/lesion motion, direction'
    }

    for domain in ['Instruments', 'Sizing', 'Diagnosis', 'Positions', 'Operation Notes', 'Movement', 'Other']:
        count = train_stats['reasoning_domains'].get(domain, 0)
        if count > 0:
            pct = count / train_stats['total_samples'] * 100
            desc = domain_descriptions.get(domain, 'Miscellaneous')
            md += f"| **{domain}** | {count} | {pct:.1f}% | {desc} |\n"

    md += "\n---\n\n"
    md += "## Question Types by Answer Format\n\n"

    # Group by Yes/No vs Open-ended
    yes_no_types = []
    open_ended_types = []

    for qt, analysis in sorted(train_stats['answer_analysis'].items()):
        if analysis['is_yes_no']:
            yes_no_types.append((qt, analysis))
        else:
            open_ended_types.append((qt, analysis))

    md += "### Yes/No Questions\n\n"
    md += "| Question Type | Domain | Total | Yes | No |\n"
    md += "|---------------|--------|-------|-----|----|\n"

    for qt, analysis in sorted(yes_no_types, key=lambda x: x[1]['total'], reverse=True):
        domain = QTYPE_TO_DOMAIN.get(qt, 'Other')
        md += f"| {qt} | {domain} | {analysis['total']} | {analysis['yes']} | {analysis['no']} |\n"

    total_yn = sum(a['yes_no'] for _, a in yes_no_types)
    md += f"\n**Total Yes/No questions:** {total_yn} ({total_yn/train_stats['total_samples']*100:.1f}% of training set)\n\n"

    md += "### Open-Ended Questions\n\n"
    md += "| Question Type | Domain | Total | Example Answers |\n"
    md += "|---------------|--------|-------|----------------|\n"

    # Sample some answers for open-ended
    for qt, analysis in sorted(open_ended_types, key=lambda x: x[1]['total'], reverse=True):
        domain = QTYPE_TO_DOMAIN.get(qt, 'Other')
        md += f"| {qt} | {domain} | {analysis['total']} | Varies (e.g., polyp, cecum, grasper) |\n"

    total_oe = sum(a['open_ended'] for _, a in open_ended_types)
    md += f"\n**Total open-ended questions:** {total_oe} ({total_oe/train_stats['total_samples']*100:.1f}% of training set)\n\n"

    md += "---\n\n"
    md += "## Overall Answer Distribution\n\n"
    md += "| Answer Type | Count | Percentage |\n"
    md += "|-------------|-------|------------|\n"

    overall = train_stats['overall_answers']
    total = train_stats['total_samples']

    md += f"| Yes | {overall['yes']} | {overall['yes']/total*100:.1f}% |\n"
    md += f"| No | {overall['no']} | {overall['no']/total*100:.1f}% |\n"
    md += f"| Open-ended | {overall['open_ended']} | {overall['open_ended']/total*100:.1f}% |\n"
    md += f"| **Total** | **{total}** | **100.0%** |\n\n"

    md += "---\n\n"
    md += "## Detailed Question Type Breakdown\n\n"
    md += "All 20 question types across train/eval/test splits:\n\n"
    md += "| Question Type | Domain | Train | Eval | Test | Total |\n"
    md += "|---------------|--------|-------|------|------|-------|\n"

    # Get all unique question types
    all_qtypes = set(train_stats['question_types'].keys()) | \
                 set(eval_stats['question_types'].keys()) | \
                 set(test_stats['question_types'].keys())

    for qtype in sorted(all_qtypes):
        domain = QTYPE_TO_DOMAIN.get(qtype, 'Other')
        train_count = train_stats['question_types'].get(qtype, 0)
        eval_count = eval_stats['question_types'].get(qtype, 0)
        test_count = test_stats['question_types'].get(qtype, 0)
        total_count = train_count + eval_count + test_count
        md += f"| {qtype} | {domain} | {train_count} | {eval_count} | {test_count} | {total_count} |\n"

    return md

def generate_json_for_app(eval_stats, test_stats):
    """Generate JSON with stats for app display"""
    return {
        'data_distribution': {
            'eval': eval_stats,
            'test': test_stats
        },
        'reasoning_domains': REASONING_DOMAINS,
        'domain_mapping': QTYPE_TO_DOMAIN
    }

def main():
    print("📊 Analyzing SurgViVQA Data Distribution...")
    print("=" * 60)

    # Check if data files exist
    required_files = ['data/train_multivideo.jsonl', 'data/eval_multivideo.jsonl', 'data/test_multivideo.jsonl']
    for f in required_files:
        if not Path(f).exists():
            print(f"❌ Error: {f} not found!")
            print("   Make sure you're running this from the project root and data/ contains the dataset files")
            return

    # Load all three splits
    train_stats = analyze_dataset('data/train_multivideo.jsonl')
    eval_stats = analyze_dataset('data/eval_multivideo.jsonl')
    test_stats = analyze_dataset('data/test_multivideo.jsonl')

    print(f"\n✓ Train: {train_stats['total_samples']} samples")
    print(f"✓ Eval: {eval_stats['total_samples']} samples")
    print(f"✓ Test: {test_stats['total_samples']} samples")

    total = train_stats['total_samples'] + eval_stats['total_samples'] + test_stats['total_samples']
    print(f"✓ Total: {total} samples")

    # Ensure docs directory exists
    import os
    os.makedirs('docs', exist_ok=True)

    # Generate markdown report for README
    print("\n📝 Generating markdown report...")
    md_report = generate_markdown_report(train_stats, eval_stats, test_stats)

    with open('docs/data_distribution.md', 'w') as f:
        f.write(md_report)
    print("✓ Saved: docs/data_distribution.md")

    # Generate JSON for app
    print("\n💾 Generating JSON for app...")
    app_data = generate_json_for_app(eval_stats, test_stats)

    with open('docs/data_stats.json', 'w') as f:
        json.dump(app_data, f, indent=2)
    print("✓ Saved: docs/data_stats.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {total:,}")
    print(f"Unique question types: {len(train_stats['question_types'])}")
    print(f"Reasoning domains: {len(train_stats['reasoning_domains'])}")
    print(f"\nAnswer distribution:")
    print(f"  Yes/No: {train_stats['overall_answers']['yes_no_total']} ({train_stats['overall_answers']['yes_no_total']/train_stats['total_samples']*100:.1f}%)")
    print(f"    - Yes: {train_stats['overall_answers']['yes']}")
    print(f"    - No: {train_stats['overall_answers']['no']}")
    print(f"  Open-ended: {train_stats['overall_answers']['open_ended']} ({train_stats['overall_answers']['open_ended']/train_stats['total_samples']*100:.1f}%)")

    print(f"\nReasoning domains:")
    for domain, count in sorted(train_stats['reasoning_domains'].items(), key=lambda x: x[1], reverse=True):
        pct = count / train_stats['total_samples'] * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")

    print("="*60)

    print("\n✅ Done! Use these files:")
    print("  - docs/data_distribution.md → Linked from README.md")
    print("  - docs/data_stats.json → App can load this for stats display")
    print("\n💡 To add performance metrics:")
    print("  Run evaluate_checkpoint.py and paste results into README")

if __name__ == '__main__':
    main()
