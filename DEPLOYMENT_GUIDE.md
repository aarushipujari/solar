# 🚀 100% Free Cloud Deployment Guide for SIH

You can deploy the **Aditya-L1 Space Weather Early Warning System** for **100% FREE** using any of the following platforms.

---

## 🥇 Option 1: Hugging Face Spaces (Recommended for AI / Hackathons)
* **Cost:** 100% Free forever
* **Specs:** **16 GB RAM + 2 vCPU** (Best for PyTorch Deep Learning)
* **Live URL:** `https://huggingface.co/spaces/your-username/aditya-l1-solar-warning`

### Steps to Deploy:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) (create a free account if you haven't).
2. Click **Create new Space**.
3. Name your Space (e.g. `aditya-l1-solar-warning`).
4. Select **Docker** (Blank) as the Space SDK.
5. Push this GitHub repository to the Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/aditya-l1-solar-warning
   git push space main
   ```
6. Hugging Face will automatically build the Dockerfile and launch your **FastAPI Command Center + Interactive Swagger API Docs** at `/docs`.

---

## 🥈 Option 2: Render (Free Web Service)
* **Cost:** 100% Free
* **Automatic Deploy:** Auto-deploys on every `git push` to your GitHub repo.
* **Live URL:** `https://your-app-name.onrender.com`

### Steps to Deploy:
1. Push this code to your GitHub account (`https://github.com/your-username/solar-flare-warning-system`).
2. Go to [render.com](https://render.com) and log in with GitHub.
3. Click **New +** $\rightarrow$ **Web Service**.
4. Select your `solar-flare-warning-system` repository.
5. Configure:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt fastapi uvicorn pydantic && python generate_sample_data.py && python train.py`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
6. Click **Create Web Service**. Your live URL will be ready in 2 minutes!

---

## 🥉 Option 3: Local Command Center (Offline Presentation Backup)

If you are presenting offline or during venue judging rounds:

### Start FastAPI Microservice:
```bash
uvicorn api:app --reload --port 8000
```
* **Interactive Command Center:** Open `http://localhost:8000/` in your browser.
* **Interactive Swagger API Docs (Show to Judges!):** Open `http://localhost:8000/docs`.

### Start Streamlit Version (Alternative):
```bash
streamlit run app.py
```
