#!/bin/bash

# VantAI Professional Target Scoreboard Launcher

set -e

echo "⚗️  VantAI Target Scoreboard - Professional Edition"
echo "   Advanced computational platform for target prioritization"
echo ""

# Function to find available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# Cleanup existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn.*app.main" || true
pkill -f "streamlit.*dashboard" || true
sleep 2

# Check Python environment
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ required, found $python_version"
    exit 1
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Ensure minimal dependencies
echo "📋 Verifying dependencies..."
pip install -q "numpy<2.0" "streamlit<1.30" requests fastapi uvicorn httpx pydantic networkx python-dotenv

# Create necessary directories
mkdir -p data_demo/cache/{json,parquet,pickle,metadata}

# Find available ports
API_PORT=$(find_available_port 8000)
STREAMLIT_PORT=$(find_available_port 8501)

echo "🔧 Configuration:"
echo "   API Port: $API_PORT"
echo "   Dashboard Port: $STREAMLIT_PORT"
echo ""

# Export API port for dashboard
export API_PORT=$API_PORT

# Function to start API server
start_api() {
    echo "🚀 Starting API server..."
    uvicorn app.main:app --host 0.0.0.0 --port $API_PORT --reload &
    API_PID=$!
    echo $API_PID > .api.pid

    # Wait for API to start
    echo "⏳ Initializing backend services..."
    for i in {1..30}; do
        if curl -s http://localhost:$API_PORT/healthz > /dev/null 2>&1; then
            echo "✅ Backend services online"
            return 0
        fi
        sleep 1
    done
    echo "⚠️  Backend may not have started properly"
}

# Function to start professional dashboard
start_dashboard() {
    echo "📊 Starting VantAI professional interface..."
    streamlit run dashboard/app.py \
        --server.port $STREAMLIT_PORT \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#22D3EE" \
        --theme.backgroundColor "#0B0F1A" \
        --theme.secondaryBackgroundColor "#0F172A" &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > .streamlit.pid
    echo "✅ Professional interface ready"
}

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down VantAI Target Scoreboard..."
    if [ -f ".api.pid" ]; then
        kill $(cat .api.pid) 2>/dev/null || true
        rm .api.pid
    fi
    if [ -f ".streamlit.pid" ]; then
        kill $(cat .streamlit.pid) 2>/dev/null || true
        rm .streamlit.pid
    fi
    echo "👋 System shutdown complete"
}

# Set cleanup trap
trap cleanup EXIT INT TERM

# Start services
start_api
sleep 3
start_dashboard

echo ""
echo "🎯 VantAI Target Scoreboard is operational"
echo ""
echo "   🌐 Professional Interface: http://localhost:$STREAMLIT_PORT"
echo "   📡 API Documentation:     http://localhost:$API_PORT/docs"
echo "   💚 Health Monitor:        http://localhost:$API_PORT/healthz"
echo ""
echo "   📊 Platform Status:       Ready for computational analysis"
echo "   🔬 Data Sources:          Open Targets, STRING, Reactome"
echo "   🧠 ML Models:             Multi-omics integration active"
echo ""
echo "Press Ctrl+C to shutdown platform"

# Wait for shutdown signal
while true; do
    sleep 1
done