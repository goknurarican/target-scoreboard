#!/bin/bash

# VantAI Target Scoreboard - Interview Demo Version
# Demonstrates advanced data integration and visualization capabilities

set -e

echo "🎯 VantAI Target Scoreboard - Interview Demo"
echo "   Showcasing: Data Integration | ML Insights | Business Intelligence"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

# Check for required tools
check_requirements() {
    print_info "Checking system requirements..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    required_version="3.8"

    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_warning "Python $python_version found, recommend 3.9+"
    else
        print_status "Python $python_version detected"
    fi
}

# Setup environment
setup_environment() {
    print_info "Setting up environment..."

    # Create virtual environment if needed
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Created virtual environment"
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies
    print_info "Installing dependencies..."
    pip install -q streamlit pandas requests fastapi uvicorn httpx pydantic networkx python-dotenv plotly numpy

    print_status "Dependencies installed"
}

# Generate enhanced demo data
generate_demo_data() {
    print_info "Generating enhanced demo datasets..."

    # Create demo data directory
    mkdir -p data_demo/{cache,enhanced}

    # Run demo data enhancement script
    python3 scripts/enhance_demo_data.py

    print_status "Enhanced demo data generated"
}

# Port management
find_available_port() {
    local start_port=$1
    local port=$start_port
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# Cleanup existing processes
cleanup_processes() {
    print_info "Cleaning up existing processes..."

    pkill -f "uvicorn.*app.main" 2>/dev/null || true
    pkill -f "streamlit.*dashboard" 2>/dev/null || true

    # Wait for processes to terminate
    sleep 2

    print_status "Cleanup complete"
}

# Start backend services
start_backend() {
    local api_port=$1

    print_info "Starting VantAI backend services..."

    # Set environment variables
    export API_PORT=$api_port
    export DEMO_MODE=true
    export ENHANCED_DATA=true

    # Start FastAPI backend
    uvicorn app.main:app --host 0.0.0.0 --port $api_port --reload --log-level warning &
    API_PID=$!
    echo $API_PID > .api.pid

    # Wait for API to be ready
    print_info "Waiting for backend to initialize..."
    for i in {1..30}; do
        if curl -s http://localhost:$api_port/healthz > /dev/null 2>&1; then
            print_status "Backend services online"
            return 0
        fi
        sleep 1
    done

    print_warning "Backend may not have started properly"
}

# Start frontend dashboard
start_frontend() {
    local streamlit_port=$1
    local api_port=$2

    print_info "Starting VantAI professional interface..."

    # Set API configuration for dashboard
    export API_PORT=$api_port

    # Start Streamlit with professional theming
    streamlit run dashboard/app.py \
        --server.port $streamlit_port \
        --server.headless true \
        --theme.base dark \
        --theme.primaryColor "#22D3EE" \
        --theme.backgroundColor "#0B0F1A" \
        --theme.secondaryBackgroundColor "#0F172A" \
        --theme.textColor "#E2E8F0" \
        --server.fileWatcherType none &

    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > .streamlit.pid

    print_status "Professional interface ready"
}

# Display demo information
show_demo_info() {
    local api_port=$1
    local streamlit_port=$2

    echo ""
    echo "🚀 VantAI Target Scoreboard Demo Ready!"
    echo ""
    echo -e "${BLUE}🌐 Professional Interface:${NC} http://localhost:$streamlit_port"
    echo -e "${BLUE}📡 API Documentation:${NC}    http://localhost:$api_port/docs"
    echo -e "${BLUE}💚 Health Monitor:${NC}       http://localhost:$api_port/healthz"
    echo ""
    echo -e "${GREEN}🎯 Demo Features:${NC}"
    echo "   • Multi-modal target scoring algorithm"
    echo "   • Interactive network visualizations"
    echo "   • Patent landscape analysis"
    echo "   • Drug discovery intelligence"
    echo "   • Competitive assessment tools"
    echo ""
    echo -e "${YELLOW}💡 Interview Talking Points:${NC}"
    echo "   • Data integration from multiple APIs (Open Targets, ChEMBL)"
    echo "   • Machine learning for target prioritization"
    echo "   • Business intelligence for strategic decision making"
    echo "   • Full-stack development (FastAPI + Streamlit)"
    echo "   • Scientific data visualization and analysis"
    echo ""
    echo -e "${BLUE}📊 Usage:${NC} Select targets → Adjust weights → Execute Analysis"
    echo ""
    echo "Press Ctrl+C to shutdown demo"
}

# Cleanup on exit
cleanup_on_exit() {
    echo ""
    print_info "Shutting down VantAI Target Scoreboard demo..."

    if [ -f ".api.pid" ]; then
        kill $(cat .api.pid) 2>/dev/null || true
        rm .api.pid
    fi

    if [ -f ".streamlit.pid" ]; then
        kill $(cat .streamlit.pid) 2>/dev/null || true
        rm .streamlit.pid
    fi

    print_status "Demo shutdown complete"
    echo "Thanks for exploring VantAI Target Scoreboard! 🎯"
}

# Main execution
main() {
    # Setup trap for cleanup
    trap cleanup_on_exit EXIT INT TERM

    # Run setup steps
    check_requirements
    setup_environment
    cleanup_processes

    # Generate enhanced data if needed
    if [ ! -f "data_demo/data_summary.json" ]; then
        generate_demo_data
    else
        print_status "Using existing enhanced demo data"
    fi

    # Find available ports
    API_PORT=$(find_available_port 8000)
    STREAMLIT_PORT=$(find_available_port 8501)

    print_info "Using ports: API=$API_PORT, Dashboard=$STREAMLIT_PORT"

    # Start services
    start_backend $API_PORT
    sleep 3
    start_frontend $STREAMLIT_PORT $API_PORT

    # Show demo information
    show_demo_info $API_PORT $STREAMLIT_PORT

    # Wait for user input
    while true; do
        sleep 1
    done
}

# Check if script is being run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi