# Backend README

This document describes the `Backend` folder for the Face-Recognition-Based-Attendance-System project. It explains the purpose of each file and folder, setup steps and how to run the backend.

## Project overview

The backend provides the API and utilities for face embedding extraction, storage, and attendance functionality. It integrates a database, face models, dataset utilities and a simple client for webcam capture.

## Repository structure (inside `Backend`)

- `create_db.py`  
  - Utility script to (re)create and initialize the database schema and sample data. Typically imports `database.py`, `models.py`, and runs table creation routines.

- `crud.py`  
  - Contains Create/Read/Update/Delete database functions used by the API or background utilities. Handles operations such as adding users, embeddings, marking attendance, and querying records.

- `database.py`  
  - Database connection and session factory. Likely reads the `DATABASE_URL` from `Backend/.env` or environment variables and exposes a session/engine for SQLAlchemy.

- `fix_dataset_faces.py`  
  - Utility to clean/normalize the `dataset` image folders, detect and fix face crop issues or filenames. Used to prepare the dataset used to compute embeddings.

- `main.py`  
  - The main application entry point (likely a FastAPI or Flask app). Defines the API endpoints, mounts routes and starts the app. Can be started with a WSGI server (e.g. `uvicorn`) or run directly if it contains a runnable block.

- `models.py`  
  - Database model definitions (SQLAlchemy ORM models). Defines tables such as users, embeddings, attendance logs, etc.

- `schema.py`  
  - Pydantic schemas / request\response models used by the API to validate and serialize data.

- `webcam_client.py`  
  - A simple client to capture frames from a webcam and send them to the backend for recognition or registration. Useful for manual testing or demoing the system.

- `requirements.txt`  
  - Python dependencies required to run the backend. Install with:
    - `pip install -r Backend/requirements.txt`

- `__pycache__/`  
  - Compiled Python bytecode files generated during execution. No source changes needed here.

### `assets/` (important runtime files)
- `assets/class_labels.pkl`  
  - Serialized mapping of class indices to person names (used for recognition labeling).

- `assets/embeddings.pkl`  
  - Precomputed face embeddings for known people. Used to speed up recognition queries.

- `assets/facenet_keras.h5`  
  - FaceNet Keras model used to compute face embeddings from aligned face crops.

- `assets/yolov8n-face.pt`  
  - YOLOv8 face detection model weights used for detecting faces in images/frames.

> Note: These files are large and required for face detection/embedding. Keep them in `assets` for runtime.

### `dataset/`
- Folder containing subfolders per identity (e.g. `dataset/Aaron_Eckhart/`, `dataset/Aaron_Sorkin/`, ...).  
- Each identity folder contains images used to compute embeddings and to build the recognition database. Use `fix_dataset_faces.py` to prepare this dataset.

### `dataset_backup_20250613_003430/`
- Backup copy of the dataset (timestamped). Useful for restoring original data or auditing dataset changes.

### `routes/` and `services/` (if present)
- `routes/`  
  - If implemented, contains modular route definitions for the API (e.g. `auth.py`, `attendance.py`, `recognition.py`).

- `services/`  
  - Business logic separate from routes/crud, such as face processing pipelines, model inference wrappers, or background tasks.

### `Backend/.env`
- Environment file that stores `DATABASE_URL` and other environment variables used by `database.py`. Do not commit secrets to VCS. Update this file for your local DB credentials.

## Setup

1. Create a Python virtual environment (recommended):
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate` (Windows)

2. Install dependencies:
   - `pip install -r Backend/requirements.txt`

3. Configure environment:
   - Copy `Backend/.env` or set environment variables for `DATABASE_URL` and other secrets. Ensure the Postgres server and the configured database `attendance_db` exist (or adjust settings accordingly).

4. Initialize the database:
   - Run `python Backend/create_db.py` (or run the appropriate initialization routine in `main.py`).

## Run the backend

- Typical FastAPI start (if `main.py` exposes `app`):
  - `uvicorn Backend.main:app --reload --host 0.0.0.0 --port 8000`

- Or run directly (if `main.py` contains a runnable entrypoint):
  - `python Backend/main.py`

## Common workflows

- Recompute embeddings:
  - Prepare images in `dataset/`, run any dataset cleanup (`fix_dataset_faces.py`), then a script (noted in project) to compute embeddings using `assets/facenet_keras.h5` and store into `assets/embeddings.pkl`.

- Add a new user:
  - Use an API endpoint (defined in `main.py`/`routes`) or a CLI script to register a new identity and compute/store their embeddings.

- Run the webcam client:
  - `python Backend/webcam_client.py` to stream captures to the backend for recognition.

## Notes and best practices

- Keep large model files in `assets/` and do not version them in source control if they exceed repo size limits; use an artifact store or Git LFS.
- Ensure `Backend/.env` is excluded from VCS (add to `.gitignore`) because it contains DB credentials.
- Backup `dataset/` before running destructive operations; a backup exists in `dataset_backup_20250613_003430/`.

## Troubleshooting

- Database connection errors:
  - Verify `DATABASE_URL` in `Backend/.env` and ensure Postgres is running and accessible.
- Missing model files:
  - Ensure `assets/facenet_keras.h5` and `assets/yolov8n-face.pt` are present before running inference.
- Dependency issues:
  - Reinstall with `pip install -r Backend/requirements.txt` inside the activated virtual environment.

## Summary

This `Backend` layout supports face detection, embedding generation, storage and an API for attendance operations. The key files to check when debugging are `database.py`, `models.py`, `crud.py`, `main.py`, and the model assets inside `assets/`.
