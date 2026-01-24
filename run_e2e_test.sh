#!/bin/bash
# Run the E2E Chromium test for DiveAnalyzer

set -e

echo "📦 Setting up test environment..."
cd /Users/mcauchy/workflow/DiveAnalizer

# Activate venv
source venv/bin/activate

echo "🧪 Running E2E Chromium test..."
python3 tests/integration/test_e2e_chromium_pipeline.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TEST PASSED"
else
    echo ""
    echo "❌ TEST FAILED"
    echo ""
    echo "📊 Test artifacts:"
    echo "  - Full log: /tmp/e2e_test.log"
    echo "  - Metrics: /tmp/e2e_metrics.json"
    echo "  - Screenshots: /tmp/e2e_*.png"
    echo ""
    echo "View log with: tail -f /tmp/e2e_test.log"
fi
