# 🚀 Deployment Guide - Physical AI & Humanoid Robotics

This guide covers deploying the frontend to Vercel and the backend to Railway.

---

## ✅ Prerequisites Checklist

Before deploying, ensure you have:

- [ ] GitHub repository: `safaaslam96/Physical-AI-Humanoid-Robotics`
- [ ] Vercel account connected to GitHub
- [ ] Railway account connected to GitHub
- [ ] Google Gemini API key
- [ ] Qdrant Cloud cluster (URL + API key)
- [ ] Neon Postgres database URL

---

## 🎨 Frontend Deployment (Vercel)

### Current Deployment
**URL**: https://physical-ai-humanoid-robotics-wine.vercel.app/

### ✅ Configuration Status

Your Vercel project is already configured with:
- ✅ Root `vercel.json` for monorepo structure
- ✅ Build command: `cd docusaurus && npm ci && npm run build`
- ✅ Output directory: `docusaurus/build`
- ✅ Auto-deployment enabled on `main` branch

### Vercel Configuration

The `vercel.json` at the root of your repository:

```json
{
  "version": 2,
  "buildCommand": "cd docusaurus && npm ci && npm run build",
  "outputDirectory": "docusaurus/build",
  "installCommand": "npm install --prefix ./docusaurus",
  "devCommand": "cd docusaurus && npm start",
  "cleanUrls": true,
  "trailingSlash": false
}
```

### Environment Variables (Vercel)

After deploying the backend, add this environment variable in Vercel:

1. Go to: https://vercel.com/safas-projects-e8c2e149/physical-ai-humanoid-robotics/settings/environment-variables
2. Add variable:
   - **Name**: `REACT_APP_BACKEND_URL`
   - **Value**: `https://your-railway-backend.up.railway.app` (get this from Railway after deployment)
3. Redeploy: Trigger a new deployment for the change to take effect

### How Auto-Deployment Works

Every push to `main` branch automatically:
1. ✅ Triggers Vercel build
2. ✅ Runs `npm ci && npm run build` in `docusaurus/`
3. ✅ Deploys `docusaurus/build/` to production
4. ✅ Updates live site at https://physical-ai-humanoid-robotics-wine.vercel.app/

---

## 🚂 Backend Deployment (Railway)

### Step 1: Create Railway Project

1. **Go to Railway**: https://railway.app/new
2. **Deploy from GitHub**:
   - Click "Deploy from GitHub repo"
   - Select `safaaslam96/Physical-AI-Humanoid-Robotics`
   - Click "Deploy Now"

3. **Configure Root Directory**:
   - After project creation, go to **Settings**
   - Scroll to "Service Settings"
   - Set **Root Directory** to: `backend`
   - Click "Save Changes"

### Step 2: Configure Environment Variables

Railway will automatically detect the `Procfile` and `railway.json`. Now add environment variables:

1. **In Railway Dashboard**, go to **Variables** tab
2. **Add the following variables**:

```env
# Google Gemini API
GEMINI_API_KEY=your_actual_gemini_api_key

# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_actual_qdrant_api_key
QDRANT_COLLECTION_NAME=physical_ai_book

# Neon Postgres Database
NEON_DATABASE_URL=postgresql://user:password@host.neon.tech/database

# Port (Railway provides this automatically)
PORT=8000
```

3. Click **"Add"** for each variable

### Step 3: Deploy

Railway will automatically:
1. ✅ Detect `backend/` as root directory
2. ✅ Read `backend/requirements.txt`
3. ✅ Use Python 3.11 (from `runtime.txt`)
4. ✅ Run `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. ✅ Monitor health at `/health` endpoint

### Step 4: Get Backend URL

1. After deployment completes, go to **Settings** → **Networking**
2. Copy your **Public URL**: `https://your-app-name.up.railway.app`
3. Save this URL - you'll need it for Vercel configuration

### Railway Configuration Files

Your backend includes these Railway-ready files:

**`backend/Procfile`**:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**`backend/railway.json`**:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**`backend/runtime.txt`**:
```
python-3.11
```

---

## 🔗 Connect Frontend to Backend

After both deployments are complete:

### Step 1: Update Vercel Environment Variable

1. Go to Vercel: https://vercel.com/safas-projects-e8c2e149/physical-ai-humanoid-robotics/settings/environment-variables
2. Add/Update:
   - **Name**: `REACT_APP_BACKEND_URL`
   - **Value**: `https://your-railway-backend.up.railway.app`
   - **Scope**: Production, Preview, Development
3. Click "Save"

### Step 2: Redeploy Frontend

1. Go to Vercel Deployments: https://vercel.com/safas-projects-e8c2e149/physical-ai-humanoid-robotics
2. Click on the latest deployment → **"Redeploy"**
3. Or push any commit to `main` to trigger auto-deployment

---

## 🧪 Testing the Deployment

### 1. Test Backend Health

```bash
curl https://your-railway-backend.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Test Frontend Chatbot

1. Visit: https://physical-ai-humanoid-robotics-wine.vercel.app/
2. Click the chatbot button (🤖) in bottom-right corner
3. Test with questions:
   - **English**: "What is Physical AI?"
   - **Urdu**: "Physical AI کیا ہے؟"
4. Verify:
   - ✅ Detailed 3-5 paragraph responses
   - ✅ Language consistency (English → English, Urdu → Urdu)
   - ✅ Book content citations

### 3. Test RAG Endpoint Directly

```bash
curl -X POST https://your-railway-backend.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Physical AI?",
    "selected_text": null,
    "from_selected_text": false
  }'
```

---

## 📊 Monitoring & Logs

### Vercel Monitoring

- **Deployments**: https://vercel.com/safas-projects-e8c2e149/physical-ai-humanoid-robotics
- **Logs**: Click any deployment → "View Function Logs"
- **Analytics**: Built-in analytics for page views

### Railway Monitoring

- **Dashboard**: https://railway.app/dashboard
- **Logs**: Click your service → "Deployments" → View logs
- **Metrics**: CPU, Memory, Network usage in dashboard
- **Health**: Automatic monitoring of `/health` endpoint

---

## 🔧 Troubleshooting

### Frontend Issues

| Issue | Solution |
|-------|----------|
| **404 errors** | Verify `vercel.json` configuration |
| **Chatbot not connecting** | Check `REACT_APP_BACKEND_URL` in Vercel env vars |
| **Build fails** | Check CI/CD pipeline in GitHub Actions |

### Backend Issues

| Issue | Solution |
|-------|----------|
| **Deployment fails** | Check Railway logs for Python dependency errors |
| **Health check fails** | Ensure backend starts on `$PORT` variable |
| **CORS errors** | Backend has CORS enabled for all origins |
| **API errors** | Check Railway logs for Gemini/Qdrant connection issues |

### Database Issues

| Issue | Solution |
|-------|----------|
| **Qdrant connection** | Verify `QDRANT_URL` and `QDRANT_API_KEY` in Railway |
| **Neon database** | Verify `NEON_DATABASE_URL` connection string |
| **Missing data** | Run `backend/ingest.py` locally to populate Qdrant |

---

## 🔄 Redeployment

### Redeploy Frontend (Vercel)

**Option 1 - Automatic**: Push to `main` branch
```bash
git push origin main
```

**Option 2 - Manual**: Use Vercel dashboard
1. Go to deployments
2. Click latest deployment → "Redeploy"

### Redeploy Backend (Railway)

**Option 1 - Automatic**: Push to `main` branch (if GitHub integration enabled)
```bash
git push origin main
```

**Option 2 - Manual**: Use Railway dashboard
1. Go to your service
2. Click "Deploy" → "Redeploy"

---

## 📝 Deployment Checklist

### Pre-Deployment
- [x] `vercel.json` configured at repository root
- [x] `backend/Procfile` exists
- [x] `backend/railway.json` configured
- [x] `backend/runtime.txt` specifies Python 3.11
- [x] `backend/requirements.txt` has all dependencies
- [x] CI/CD pipeline passing

### Vercel Deployment
- [ ] Project connected to GitHub repo
- [ ] Auto-deployment enabled on `main` branch
- [ ] Build succeeds with no errors
- [ ] Site accessible at https://physical-ai-humanoid-robotics-wine.vercel.app/

### Railway Deployment
- [ ] Project created from GitHub repo
- [ ] Root directory set to `backend`
- [ ] All environment variables configured:
  - [ ] `GEMINI_API_KEY`
  - [ ] `QDRANT_URL`
  - [ ] `QDRANT_API_KEY`
  - [ ] `QDRANT_COLLECTION_NAME`
  - [ ] `NEON_DATABASE_URL`
  - [ ] `PORT`
- [ ] Deployment succeeds
- [ ] Health check passing at `/health`
- [ ] Public URL obtained

### Post-Deployment
- [ ] `REACT_APP_BACKEND_URL` set in Vercel to Railway URL
- [ ] Frontend redeployed with backend URL
- [ ] Chatbot working on production site
- [ ] English questions → English detailed answers
- [ ] Urdu questions → Urdu detailed answers
- [ ] RAG endpoint returning 3-5 paragraph responses

---

## 🌐 Deployment URLs

**Frontend (Vercel)**: https://physical-ai-humanoid-robotics-wine.vercel.app/
**Backend (Railway)**: `https://your-app-name.up.railway.app` *(update after deployment)*

**Last Updated**: 2025-12-28
