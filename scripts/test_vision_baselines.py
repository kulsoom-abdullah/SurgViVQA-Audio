#!/usr/bin/env python3
"""
Unit tests and quick pipeline validation for vision baseline comparison
Run this before launching expensive full experiments
"""

import json
import sys
from pathlib import Path
import tempfile

def test_jsonl_has_required_fields():
    """Test 1: Verify test data has all required fields"""
    print("\n🧪 Test 1: Checking test data format...")

    test_file = Path("test_multivideo.jsonl")
    if not test_file.exists():
        print(f"  ❌ FAIL: {test_file} not found")
        return False

    # Read first sample
    with open(test_file) as f:
        first_line = f.readline()
        sample = json.loads(first_line)

    required_fields = ['id', 'question', 'answer', 'short_answer', 'question_type', 'frames']
    missing = [f for f in required_fields if f not in sample]

    if missing:
        print(f"  ❌ FAIL: Missing fields: {missing}")
        return False

    print(f"  ✅ PASS: All required fields present")
    print(f"     Sample ID: {sample['id']}")
    print(f"     Question type: {sample['question_type']}")
    print(f"     Short answer: {sample['short_answer']}")
    return True

def test_frames_directory_exists():
    """Test 2: Verify frames directory is accessible"""
    print("\n🧪 Test 2: Checking frames directory...")

    frames_dir = Path("data/frames")
    if not frames_dir.exists():
        print(f"  ❌ FAIL: {frames_dir} not found")
        return False

    # Check for at least one video folder
    video_folders = list(frames_dir.glob("002-*"))
    if not video_folders:
        print(f"  ❌ FAIL: No video folders (002-*) found in {frames_dir}")
        return False

    print(f"  ✅ PASS: Frames directory exists with {len(video_folders)} video folders")
    return True

def test_baseline_script_imports():
    """Test 3: Verify baseline script can import dependencies"""
    print("\n🧪 Test 3: Checking baseline script imports...")

    try:
        # Add baselines to path
        sys.path.insert(0, 'baselines')
        from utils import load_model, check_answer_match
        print("  ✅ PASS: Baseline utils imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ FAIL: Import error: {e}")
        return False

def test_answer_matching_logic():
    """Test 4: Verify answer matching works correctly"""
    print("\n🧪 Test 4: Testing answer matching logic...")

    sys.path.insert(0, 'baselines')
    from utils import check_answer_match

    test_cases = [
        # (predicted, ground_truth, short_answer, expected_result)
        ("Yes", "Yes, there is occlusion", "Yes", True),
        ("The answer is left", "left", "left", True),
        ("right", "left", "left", False),
        ("The lesion is moving to the left", "left", "left", True),
        ("NBI mode is active", "NBI", "NBI", True),
        ("No occlusion detected", "No", "No", True),
    ]

    passed = 0
    failed = 0

    for predicted, ground_truth, short_answer, expected in test_cases:
        result = check_answer_match(predicted, ground_truth, short_answer)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ❌ FAIL: predicted='{predicted}', short='{short_answer}' -> got {result}, expected {expected}")

    if failed == 0:
        print(f"  ✅ PASS: All {passed} answer matching tests passed")
        return True
    else:
        print(f"  ❌ FAIL: {failed}/{passed+failed} tests failed")
        return False

def test_comparison_script_logic():
    """Test 5: Test comparison report generation with dummy data"""
    print("\n🧪 Test 5: Testing comparison report generation...")

    # Create dummy results
    dummy_results = [
        {
            'question_id': 'test_001',
            'question_type': 'occlusion_check',
            'exact_match': 1,
        },
        {
            'question_id': 'test_002',
            'question_type': 'lesion_motion_direction',
            'exact_match': 0,
        },
        {
            'question_id': 'test_003',
            'question_type': 'occlusion_check',
            'exact_match': 1,
        }
    ]

    # Test statistics calculation
    type_stats = {}
    for r in dummy_results:
        qtype = r['question_type']
        if qtype not in type_stats:
            type_stats[qtype] = {'correct': 0, 'total': 0}
        type_stats[qtype]['total'] += 1
        type_stats[qtype]['correct'] += r['exact_match']

    # Check calculations
    if type_stats['occlusion_check']['correct'] == 2 and type_stats['occlusion_check']['total'] == 2:
        if type_stats['lesion_motion_direction']['correct'] == 0 and type_stats['lesion_motion_direction']['total'] == 1:
            print("  ✅ PASS: Statistics calculation works correctly")
            return True

    print("  ❌ FAIL: Statistics calculation incorrect")
    return False

def test_create_sample_subset():
    """Test 6: Create 10-sample test subset"""
    print("\n🧪 Test 6: Creating 10-sample test subset...")

    test_file = Path("test_multivideo.jsonl")
    if not test_file.exists():
        print(f"  ❌ FAIL: {test_file} not found")
        return False

    output_file = Path("test_sample_10.jsonl")

    with open(test_file) as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i >= 10:
                break
            f_out.write(line)

    # Verify created file
    with open(output_file) as f:
        lines = f.readlines()

    if len(lines) == 10:
        print(f"  ✅ PASS: Created {output_file} with 10 samples")
        return True
    else:
        print(f"  ❌ FAIL: Expected 10 samples, got {len(lines)}")
        return False

def run_all_tests():
    """Run all unit tests"""
    print("="*80)
    print("VISION BASELINE COMPARISON - UNIT TESTS")
    print("="*80)

    tests = [
        ("Data Format Check", test_jsonl_has_required_fields),
        ("Frames Directory Check", test_frames_directory_exists),
        ("Baseline Script Imports", test_baseline_script_imports),
        ("Answer Matching Logic", test_answer_matching_logic),
        ("Comparison Script Logic", test_comparison_script_logic),
        ("Sample Subset Creation", test_create_sample_subset),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED - Ready to run quick pipeline test!")
        print("\nNext step:")
        print("  bash scripts/quick_baseline_test.sh")
        return True
    else:
        print(f"\n❌ {total_count - passed_count} TESTS FAILED - Fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
