#!/bin/bash
# Check if transformers fork is complete on RunPod

echo "🔍 Checking transformers fork installation..."
echo ""

echo "1. Transformers import location:"
python -c "import transformers; print(transformers.__file__)"
echo ""

echo "2. Checking for data directory:"
if [ -d "transformers_fork/src/transformers/data" ]; then
    echo "✅ transformers/data directory EXISTS"
    echo "   Files inside:"
    ls -la transformers_fork/src/transformers/data/
else
    echo "❌ transformers/data directory MISSING"
    echo "   This is the problem!"
fi
echo ""

echo "3. Checking for data_collator.py:"
if [ -f "transformers_fork/src/transformers/data/data_collator.py" ]; then
    echo "✅ data_collator.py EXISTS"
else
    echo "❌ data_collator.py MISSING"
fi
echo ""

echo "4. Checking if transformers fork was committed to git:"
if [ -d ".git" ]; then
    echo "Git status of transformers_fork:"
    git ls-files transformers_fork/src/transformers/data/ | head -10
else
    echo "Not a git repo"
fi
