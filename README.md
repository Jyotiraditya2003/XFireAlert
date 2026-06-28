# XFireAlert

AI-powered forest fire prediction and detection system with explainable AI (XAI) capabilities.

## Quick Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure MongoDB (optional)
cp .env.example .env
# Edit .env with your MongoDB Atlas credentials

# Run the app
streamlit run app/streamlit_app.py
```

## Requirements

- Python 3.10+
- MongoDB Atlas account (optional, for data logging)
