# AIRCHECK Pipeline on Vertex AI Workbench

This guide describes how to run the AIRCHECK pipeline using Google Cloud's Vertex AI Workbench, including setting up the instance, preparing environments, running the code, and saving results.

---

## Step 1: Create a Vertex AI Workbench Instance

1. Go to [Vertex AI > Workbench > Instances](https://console.cloud.google.com/vertex-ai/workbench/instances).
2. Click **"Create"**.
3. Use the default **Jupyter (Python 3.x)** environment.
4. Choose a machine type (e.g., `n1-standard-4`).
5. Click **"Create"**.
6. Once it's running, click **"Open JupyterLab"**.

---

## Step 2: Set Up the Environment

### 🟢 If you **don’t need conformal prediction (MAPIE)**:

You can use the default Jupyter environment.

```bash
# Clone the repository
git clone https://github.com/bhklab/ml-pipeline-aircheck
cd ml-pipeline-aircheck

# Install required packages
pip install -r requirements.txt
```

---

### 🔵 If you **want to use conformal prediction (MAPIE)**:

You must create and activate a Python 3.11 conda environment.

```bash
# Check if conda is installed
conda --version

# Create a Python 3.11 conda environment
conda create -y -n py311 python=3.11

# Activate the environment
conda activate py311

# Clone the repository
git clone https://github.com/bhklab/ml-pipeline-aircheck
cd ml-pipeline-aircheck

# Install MAPIE-compatible packages
pip install -r requirements_mapie_conformal.txt
```

---

## Step 3: Run the Pipeline (from Terminal in JupyterLab)

### Option A: Simple run (for quick testing)

```bash
python aircheck_pipeline.py
```

---

### Option B: Recommended for long runs (won’t crash if terminal disconnects)

```bash
# Run the pipeline in the background and log output
nohup python aircheck_pipeline.py > output.log 2>&1 &
```

#### Monitor the run:
```bash
ps aux | grep aircheck_pipeline.py
```

#### View live output:
```bash
tail -f output.log
```

#### Stop the background process:
```bash
kill <process_id>
```

(Replace `<process_id>` with the actual number from `ps aux`)

---

## Step 4: Save Results to Google Cloud Storage (GCS)

### Copy result folder to GCS:
```bash
gsutil -m cp -r run_test gs://<your-bucket-name>/testresult
```

### Copy MLflow logs to GCS:
```bash
gsutil -m cp -r mlruns gs://<your-bucket-name>/mlruns
```

---

## (Optional) View MLflow Tracking UI

### Launch MLflow server:
```bash
mlflow ui --port 8081 --backend-store-uri ./mlruns
```

### Open the MLflow UI in browser:
1. In JupyterLab, go to **Vertex AI --> Web preview --> Port 8081**
2. The MLflow UI will open in a new browser tab.

---
