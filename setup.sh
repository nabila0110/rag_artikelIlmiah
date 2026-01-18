# Check Python
echo ""
echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo "Python found: $(python3 --version)"

# Create virtual environment
echo ""
echo "[2/7] Creating virtual environment..."
python3 -m venv venv
source venv\Scripts\activate
echo "Virtual environment created"

# # Install dependencies
# echo ""
# echo "[3/7] Installing dependencies..."
# pip install -r requirements.txt
# echo "Dependencies installed"

# Check Ollama
echo ""
echo "[4/7] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install from https://ollama.ai"
    echo "After installing, run: ollama pull gemma2:9b"
else
    echo "Ollama found"
fi

# Create directories
echo ""
echo "[5/7] Creating directories..."
mkdir -p data models static/css static/js static/images templates utils
echo "Directories created"

# Prepare data
echo ""
echo "[6/7] Preparing data..."
python prepare_data.py
if [ $? -ne 0 ]; then
    echo "Data preparation failed"
    exit 1
fi
echo "Data prepared"

# Test setup
echo ""
echo "[7/7] Testing setup..."
python -c "from app import app; print('App imports successfully')"

echo ""
echo "----------------------------------"
echo "SETUP COMPLETE!"
echo "----------------------------------"
echo ""
echo "To run the application:"
echo "  1. Make sure Ollama is running: ollama serve"
echo "  2. Run: python run.py"
echo "  3. Open: http://localhost:5000"
echo ""