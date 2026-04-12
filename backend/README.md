# Backend Setup

## Install
```bash
pip install -r requirements.txt
```

For Apple Silicon (Metal support):
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

## Model
Place GGUF model inside `backend/models/`

## 🔁 Inference Modes

### Mock Mode (default)
- Instant responses
- Deterministic demo
- No GPU required

### Real Mode (Kaggle + Qwen 7B)
- Uses remote model via ngrok
- Enables real stochastic behavior
- Fallback to local GGUF if Kaggle fails

Set in `.env`:
```env
USE_MOCK=false
KAGGLE_URL=https://your-ngrok-url.ngrok-free.app/generate
```

## Run
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
