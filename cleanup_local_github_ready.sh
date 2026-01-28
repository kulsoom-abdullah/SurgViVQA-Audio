#!/bin/bash
################################################################################
# Cleanup Script - GitHub Ready Version
# Removes development/session notes, keeps useful documentation
################################################################################

echo "================================================================================"
echo "Cleaning Up for GitHub - Keeping Useful Documentation"
echo "================================================================================"
echo ""

# ============================================================================
# Files to REMOVE (Development/Session Notes - Not Needed for GitHub)
# ============================================================================
FILES_TO_REMOVE=(
    # Session/Planning Notes (personal, not useful for others)
    "RESUME_SESSION.md"
    "NEXT_STEPS.md"

    # Development Debug Notes (not needed in final repo)
    "PROMPT_FIX_SUMMARY.md"
    "BASELINE1_FIX_SUMMARY.md"
    "GEMINI_FEEDBACK_FIXES.md"

    # Redundant with RUNPOD_CHECKLIST.md
    "UPLOAD_CHECKLIST.md"
    "FRAME_UPLOAD_OPTIONS.md"
    "RUNPOD_SETUP.md"
    "RUNPOD_GUIDE.md"
    "RUNPOD_VENV_GUIDE.md"

    # Old versions of scripts (replaced by better versions)
    "setup_runpod.sh"              # Old conda version
    "DOWNLOAD_FRAMES_RUNPOD.sh"    # Redundant with download_frames_figshare.sh

    # Cleanup scripts (not needed in repo)
    "cleanup_local.sh"             # Old version of this script
)

echo "📝 Files to REMOVE (development notes, redundant docs):"
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -e "$file" ]; then
        echo "  ❌ $file"
    fi
done

echo ""
echo "✅ Files to KEEP (useful for GitHub documentation):"

# ============================================================================
# Files to KEEP (Useful Documentation for GitHub)
# ============================================================================
FILES_TO_KEEP=(
    # Core experiment documentation
    "EXPERIMENT_DESIGN.md"         # Explains 3-baseline approach
    "dataset_description.md"       # Dataset info
    "VERTICAL_SLICE_SETUP.md"      # Methodology explanation

    # User guides
    "RUNPOD_CHECKLIST.md"          # Complete setup guide
    "ESSENTIAL_FILES.md"           # File organization guide

    # Core scripts
    "baselines/"
    "test_set/"
    "verify_setup.py"
    "generate_audio_subset.py"
    "setup_runpod_venv.sh"
    "activate_venv.sh"
    "download_frames_figshare.sh"
    "run_all_baselines.sh"
)

for file in "${FILES_TO_KEEP[@]}"; do
    if [ -e "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "================================================================================"
read -p "Remove development files? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    for file in "${FILES_TO_REMOVE[@]}"; do
        if [ -e "$file" ]; then
            rm -rf "$file"
            echo "✓ Removed: $file"
        fi
    done
    echo ""
    echo "================================================================================"
    echo "✓ Cleanup Complete - Repository is GitHub Ready!"
    echo "================================================================================"
    echo ""
    echo "📁 What's left (all useful for GitHub):"
    echo ""
    echo "Documentation:"
    echo "  - EXPERIMENT_DESIGN.md      (explains your 3-baseline comparison)"
    echo "  - VERTICAL_SLICE_SETUP.md   (methodology for vertical slice testing)"
    echo "  - dataset_description.md    (SurgViVQA dataset info)"
    echo "  - RUNPOD_CHECKLIST.md       (complete setup guide for reproducibility)"
    echo "  - ESSENTIAL_FILES.md        (file organization reference)"
    echo ""
    echo "Code:"
    echo "  - baselines/                (3 baseline implementations)"
    echo "  - test_set/                 (50-sample test sets for quick validation)"
    echo "  - *.py scripts              (utilities)"
    echo "  - *.sh scripts              (automation)"
    echo ""
    echo "Next Steps:"
    echo "  1. Write a comprehensive README.md (use EXPERIMENT_DESIGN.md as guide)"
    echo "  2. Add results/ folder with sample outputs"
    echo "  3. Consider adding requirements.txt (extract from setup_runpod_venv.sh)"
    echo "  4. Add LICENSE file"
    echo "  5. Commit to GitHub"
    echo ""
    echo "================================================================================"
else
    echo "Cleanup cancelled."
fi
