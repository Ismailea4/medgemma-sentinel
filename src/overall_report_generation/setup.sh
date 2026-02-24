#!/bin/bash

# MedGemma Sentinel - Setup Script
# Automates environment setup and dependency installation

set -e

echo "🏥 MedGemma Sentinel - Setup Script"
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📥 Installing core dependencies..."
pip install -r requirements.txt

# Install UI dependencies
echo "📥 Installing UI dependencies..."
pip install -r requirements-ui.txt

# Download MedGemma model (optional)
read -p "📥 Download MedGemma 2 model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⏳ Downloading MedGemma 2 model..."
    python scripts/download_medgemma.py
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the Streamlit UI, run:"
echo "  streamlit run ui/app.py"
echo ""
