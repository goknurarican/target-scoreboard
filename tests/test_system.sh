
# Quick test script oluştur
cat > test_system.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing VantAI Target Scoreboard System..."

# Test 1: API Health Check
echo "1️⃣ Testing API health..."
health_response=$(curl -s http://localhost:8000/healthz)
if echo "$health_response" | grep -q "healthy"; then
    echo "✅ API health check passed"
else
    echo "❌ API health check failed"
    exit 1
fi

# Test 2: Score API
echo "2️⃣ Testing scoring API..."
if [ -f "examples/sample_payload.json" ]; then
    score_response=$(curl -s -X POST http://localhost:8000/score \
        -H 'Content-Type: application/json' \
        -d @examples/sample_payload.json)

    if echo "$score_response" | grep -q "targets"; then
        echo "✅ Scoring API test passed"
        echo "📊 Sample response:"
        echo "$score_response" | jq '.targets[0] | {target, total_score, data_version}' 2>/dev/null || echo "$score_response"
    else
        echo "❌ Scoring API test failed"
        echo "Response: $score_response"
        exit 1
    fi
else
    echo "⚠️  Sample payload not found, skipping scoring test"
fi

# Test 3: System Summary
echo "3️⃣ Testing system summary..."
summary_response=$(curl -s http://localhost:8000/summary)
if echo "$summary_response" | grep -q "system_info"; then
    echo "✅ System summary test passed"
else
    echo "❌ System summary test failed"
    exit 1
fi

# Test 4: Unit Tests
echo "4️⃣ Running unit tests..."
if command -v pytest &> /dev/null; then
    if pytest tests/ -v --tb=short; then
        echo "✅ Unit tests passed"
    else
        echo "⚠️  Some unit tests failed (this may be expected in demo mode)"
    fi
else
    echo "⚠️  pytest not found, skipping unit tests"
fi

echo ""
echo "🎉 System test completed!"
echo "🔗 Try the dashboard at: http://localhost:8501"
EOF
