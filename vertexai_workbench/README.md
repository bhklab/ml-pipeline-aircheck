# AIRCHECK Pipeline on Vertex AI Workbench

This guide describes how to run the AIRCHECK pipeline using Google Cloud's Vertex AI Workbench, including setting up the instance, preparing a Python 3.11 environment, running the code, and saving results.

---

## Step 1: Create a Vertex AI Workbench Instance

1. Go to [Vertex AI > Workbench > Instances](https://console.cloud.google.com/vertex-ai/workbench/instances).
2. Click **"Create"**.
3. Use the default **Jupyter (Python 3.x)** environment.
4. Choose a machine type (e.g., `n1-standard-4`).
5. Click **"Create"**.
6. Once it's running, click **"Open JupyterLab"**.

---

## Step 2: Set Up Python 3.11 Conda Environment

### Check if Conda is installed
```bash
conda --version
```

### Create a new environment with Python 3.11
```bash
conda create -y -n py311 python=3.11
```

### Activate the environment
```bash
conda activate py311
```

### Install required packages
```bash
pip install -r requirements.txt
```

---

## Step 3: Run the Pipeline (from Terminal in JupyterLab)

### Clone or update the repo
```bash

# Clone if not already cloned
git clone https://github.com/bhklab/ml-pipeline-aircheck

# Or update if it's already cloned
cd ml-pipeline-aircheck
git pull

# Navigate to your workspace
cd ml-pipeline-aircheck
```

### Remove previous run folder (optional)
```bash
rm -rf run_test
```

### Run the pipeline
```bash
python aircheck_pipeline.py
```

---

## Step 4: Save Results to Google Cloud Storage (GCS)

### Copy result folder to GCS
```bash
gsutil -m cp -r run_test gs://gs_bucket_name/testresult
```

### Copy MLflow logs to GCS
```bash
gsutil -m cp -r mlruns gs://gs_bucket_name/mlruns
```

---

## (Optional) View MLflow Tracking UI

### Launch MLflow server
```bash
mlflow ui --port 8081 --backend-store-uri ./mlruns
```

### Open the MLflow UI in browser
1. In the JupyterLab menu bar, go to  
   **Vertex AI --> Web preview --> Port 8081**
2. The UI will open in a new browser tab.

---

