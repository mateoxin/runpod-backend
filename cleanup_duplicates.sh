#!/bin/bash
# 🧹 Cleanup duplicate files for GitHub deploy readiness

echo "🧹 Cleaning up duplicate files for GitHub deploy..."

# Remove all files with (1) in the name
find . -name "*\(1\)*" -type f -delete

echo "✅ Removed duplicate files"

# List remaining files
echo "📋 Remaining files:"
ls -la | grep -E "(\.md|\.py|\.sh|\.txt|\.json)$" | wc -l
echo "files found"

echo "🎉 Cleanup complete!"