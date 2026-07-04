# QuakeSense — your runbook

## 1. GitHub repo (5 min) — PowerShell inside the `quakesense` folder
```powershell
git init
git add .
git commit -m "QuakeSense: AI earthquake situation room on USGS public data"
```
Create empty **public** repo `quakesense` on github.com, then:
```powershell
git remote add origin https://github.com/<your-username>/quakesense.git
git branch -M main
git push -u origin main
```
`.gitignore` already blocks service-account keys and the large history.csv.

## 2. Load 50 years of USGS data into BigQuery (10 min)
```powershell
pip install -r requirements.txt
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\...\gcp-service-account.json"
python scripts/load_history.py
```
Expect ~50 yearly downloads then "Loaded ~100,000 rows". Also writes `data/baseline.csv` (commit that file — the anomaly tab needs it; it's small).

## 3. Run and verify locally (10 min)
```powershell
streamlit run app.py
```
Check each tab: Live Monitor shows this week's real quakes · AI Briefings returns `source: gemini` · Ask the Data answers "How many M6+ earthquakes hit Myanmar since 1990?" and shows SQL · Anomaly Watch loads.
If Gemini errors: enable **Vertex AI API**, give the service account **Vertex AI User**.

## 4. Deploy to Cloud Run (20 min)
```powershell
gcloud auth login
gcloud config set project usar-decision-intel
gcloud services enable run.googleapis.com cloudbuild.googleapis.com aiplatform.googleapis.com bigquery.googleapis.com
gcloud run deploy quakesense --source . --region us-central1 --allow-unauthenticated --memory 1Gi
```
Grant the Cloud Run service account: BigQuery Data Editor, BigQuery Job User, Vertex AI User.
The URL it prints = your **working prototype link**.

## 5. Demo video script (3 min)
1. **Hook (20 s):** "After the 2025 Mandalay earthquake, millions searched for answers and found raw numbers. QuakeSense turns official USGS data into decisions."
2. **Live Monitor (30 s):** world map, real quakes from this week, filter to M4.5+.
3. **AI Briefing (45 s):** pick the week's biggest quake → generate briefing → read the recommended actions aloud.
4. **Ask the Data (60 s):** the money shot. Type "How many M6+ earthquakes hit Myanmar since 1990?" → show answer → expand "Show the SQL Gemini generated". Ask one more complex question (yearly trend → bar chart appears).
5. **Anomaly Watch (30 s):** flagged region → AI explanation of the swarm.
6. **Close (15 s):** architecture + "real data, real decisions, deployed on Google Cloud."

## Cost
USGS APIs: free. BigQuery: ~100k rows = pennies, queries within free tier. Gemini Flash: <$0.01 per interaction (your $1000 GenAI credit). Cloud Run: free tier. **Total < $5.**

## Judge Q&A prep
- "Why not predict earthquakes?" → Prediction is scientifically impossible; we optimize the decisions that ARE possible: awareness, communication, preparedness. (Responsible AI point!)
- "How does it scale?" → Stateless Cloud Run + BigQuery; add Cloud Scheduler ingestion and Pub/Sub alerts.
- "Data quality?" → USGS is the global authoritative source; catalog is complete for M5+ since ~1975.
