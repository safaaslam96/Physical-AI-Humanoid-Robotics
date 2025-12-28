# Deployment Guide

## 🚀 Vercel Deployment (Frontend)

### Automatic Deployment
Your project is already connected to Vercel at:
https://physical-ai-humanoid-robotics-wine.vercel.app/

Every push to `main` branch will automatically deploy.

### Environment Variables
In Vercel dashboard, add:
- `REACT_APP_BACKEND_URL` = Your Railway backend URL (e.g., `https://your-app.railway.app`)

### Manual Deployment
```bash
cd docusaurus
npm install
npm run build
npx vercel --prod
```

---

## 🚂 Railway Deployment (Backend)

### Initial Setup
1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select `Physical-AI-Humanoid-Robotics` repository
4. Set **Root Directory** to `backend`
5. Railway will auto-detect `Procfile` and deploy

### Required Environment Variables
Add these in Railway dashboard under Variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=postgresql://user:password@host/database
PORT=8000
```

### Deployment Command
Railway uses the `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Health Check
Railway will monitor: `https://your-app.railway.app/health`

---

## 📝 Post-Deployment Steps

1. **Get Railway Backend URL**:
   - After Railway deployment, copy your app URL (e.g., `https://your-app-name.up.railway.app`)

2. **Update Vercel Environment Variable**:
   - Go to Vercel dashboard → Your project → Settings → Environment Variables
   - Set `REACT_APP_BACKEND_URL` to your Railway URL
   - Redeploy frontend

3. **Test the Deployment**:
   - Visit your Vercel site: https://physical-ai-humanoid-robotics-wine.vercel.app/
   - Open the chatbot (🤖 button)
   - Test with a question like "What is Physical AI?"
   - Verify it returns a detailed, language-consistent answer

---

## 🔧 Troubleshooting

### Frontend Issues
- **Chatbot not connecting**: Check `REACT_APP_BACKEND_URL` in Vercel env vars
- **Build fails**: Ensure all dependencies in `docusaurus/package.json`
- **404 on pages**: Check `vercel.json` configuration

### Backend Issues
- **Railway deployment fails**: Check `requirements.txt` and `runtime.txt`
- **Health check fails**: Ensure backend starts on `$PORT` variable
- **CORS errors**: Backend already has CORS enabled for all origins
- **API errors**: Check Railway logs for Gemini/Qdrant connection issues

### Database Issues
- **Qdrant connection**: Verify `QDRANT_URL` and `QDRANT_API_KEY`
- **Neon database**: Verify `NEON_DATABASE_URL` connection string
- **Missing data**: Run `backend/ingest.py` to populate Qdrant

---

## 📊 Monitoring

### Vercel
- Deployments: https://vercel.com/dashboard
- Logs: Click on deployment → View Logs
- Analytics: Built-in analytics for page views

### Railway
- Logs: Railway dashboard → Your project → Deployments → Logs
- Metrics: CPU, Memory, Network usage in dashboard
- Health: Automatic health checks on `/health` endpoint

---

## 🔄 CI/CD Pipeline

### Current Setup
- **Push to main** → Vercel auto-deploys frontend
- **Push to main** → Railway auto-deploys backend (if connected to GitHub)

### Manual Deploy
```bash
# Frontend (Vercel)
cd docusaurus
npx vercel --prod

# Backend (Railway CLI)
cd backend
railway up
```

---

## ✅ Deployment Checklist

- [ ] Railway project created with backend deployed
- [ ] Railway environment variables set (Gemini, Qdrant, Neon, PORT)
- [ ] Railway backend URL copied
- [ ] Vercel environment variable `REACT_APP_BACKEND_URL` updated
- [ ] Frontend redeployed on Vercel
- [ ] Health check passing: `https://your-backend.railway.app/health`
- [ ] Chatbot working on production site
- [ ] Language detection working (English/Urdu)
- [ ] Detailed answers (3-5 paragraphs) being returned

---

Last updated: 2024-12-28
Frontend: https://physical-ai-humanoid-robotics-wine.vercel.app/
Backend: https://your-railway-backend.railway.app (update after deployment)
