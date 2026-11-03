# RESIDE - run on a new machine

Needs **Python 3.12** and **Node.js (LTS)** on PATH, and internet for the first install.

1. Double-click `setup.bat` (one time; the face-service install is large).
2. Double-click `start_all.bat`.
3. Open http://localhost:5173

Three windows open: face service (8010), backend (8000), frontend (5173).

## Notes
- Data (users, cameras, settings) is in `backend/sentinel.db`; models and the demo video are under `backend/storage`.
- `.env` holds the JWT secret and settings. No absolute paths: the project works from any folder.
- `_model_cache/` holds the offline face/OCR/ResNet models; `setup.bat` copies them into your user profile.
- A GPU is optional. For an NVIDIA GPU, install the CUDA build of torch (pytorch.org) before `setup.bat`.
- Not included on purpose: training datasets, training samples, logs, virtualenvs and `node_modules` (setup recreates the environments).
