# Workshop FastAPI service — quick start

This file contains exact commands and steps to create the model, start the FastAPI service that exposes `/predict` on port 9696, test it, and run the service inside Docker using the included `Dockerfile`.

Paths used in these examples
- Workshop folder: `/workspaces/machine-learning-zoomcamp/05-deployment/workshop/`
- Server module: `predict.py`
- Training script: `train.py`
- Model file: `model.bin`

Prerequisites
- Python 3 (3.8+ recommended)
- pip
- docker (optional, only for Docker section)

1) Install Python dependencies

Run this in your environment (or inside a virtualenv):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install fastapi uvicorn pandas scikit-learn numpy requests
```

Note: The training script uses scikit-learn, pandas and numpy. The server uses FastAPI (which depends on pydantic). The `requests` package is optional but convenient for testing from Python.

2) Create (train) the model

From the workshop folder run:

```bash
cd /workspaces/machine-learning-zoomcamp/05-deployment/workshop
python3 train.py
# -> this will create model.bin in the current folder
ls -l model.bin
```

3) Start the FastAPI server (foreground)

From the same folder you can run the module directly — `predict.py` runs uvicorn when executed as __main__:

```bash
cd /workspaces/machine-learning-zoomcamp/05-deployment/workshop
python3 predict.py
# This runs uvicorn on 0.0.0.0:9696 in the foreground (CTRL+C to stop)
```

4) Start uvicorn directly (foreground alternative)

```bash
cd /workspaces/machine-learning-zoomcamp/05-deployment/workshop
uvicorn predict:app --host 0.0.0.0 --port 9696 --app-dir .
# or: python3 -m uvicorn predict:app --host 0.0.0.0 --port 9696 --app-dir .
```

5) Start uvicorn detached (background, recommended for long runs)

Example using `nohup` to keep the process running after you close the terminal:

```bash
cd /workspaces/machine-learning-zoomcamp/05-deployment/workshop
nohup uvicorn predict:app --host 0.0.0.0 --port 9696 --app-dir . > /tmp/uvicorn_workshop.log 2>&1 &
echo $! > /tmp/uvicorn_workshop.pid
tail -n 200 /tmp/uvicorn_workshop.log
```

6) Test the `/predict` endpoint

Curl (single-line JSON payload):

```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"gender":"female","seniorcitizen":0,"partner":"yes","dependents":"no","phoneservice":"no","multiplelines":"no_phone_service","internetservice":"dsl","onlinesecurity":"no","onlinebackup":"yes","deviceprotection":"no","techsupport":"no","streamingtv":"no","streamingmovies":"no","contract":"month-to-month","paperlessbilling":"yes","paymentmethod":"electronic_check","tenure":1,"monthlycharges":29.85,"totalcharges":29.85}' \
  http://localhost:9696/predict

# Expect JSON similar to:
# {"churn_probability":0.66...,"churn":true}
```

Python (requests) snippet to run from a notebook or script:

```python
import requests

url = 'http://localhost:9696/predict'
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85,
}

response = requests.post(url, json=customer)
print(response.json())
```

7) Run inside Docker (using the included `Dockerfile`)

The repository includes `05-deployment/workshop/Dockerfile` which contains an ENTRYPOINT that runs `uvicorn predict:app --host 0.0.0.0 --port 9696`. The Dockerfile also expects `predict.py` and `model.bin` to be present (they are copied into the image during build). There are two ways to use Docker:

Option A — Build after you produced `model.bin` locally (simple):

```bash
cd /workspaces/machine-learning-zoomcamp/05-deployment/workshop
# Ensure model.bin exists (run train.py if needed)
python3 train.py

# Build the image (from the workshop folder, where Dockerfile is located)
docker build -t churn-workshop .

# Run the container, map port 9696
docker run --rm -p 9696:9696 churn-workshop

# Now the endpoint will be available at http://localhost:9696/predict
```

Option B — (Alternative) Modify the Dockerfile to train the model during image build
- This is optional and increases image build time. If you prefer, change the Dockerfile to run `train.py` during build and produce `model.bin` inside the image. If you do that, the build command will produce an image that can run without you copying `model.bin` beforehand.

8) Troubleshooting notes
- If you see ModuleNotFoundError for `pydantic`, `fastapi`, or `uvicorn`, install them (see step 1).
- If your notebook runs in a different container or remote kernel, `localhost` may not be the same host as the server — ensure both the notebook and server share the same network/host or use the container's IP.
- The server loads `model.bin` from its working directory. If you move `predict.py`, update paths or use an absolute path to `model.bin` in `predict.py`.

9) Exact commands I executed during this session (log)

- Installed FastAPI and Uvicorn (into the active environment):

```bash
python3 -m pip install fastapi uvicorn
```

- Trained and saved the model:

```bash
python3 /workspaces/machine-learning-zoomcamp/05-deployment/workshop/train.py
# Output: "Model saved to model.bin"
```

- Started the server (foreground, then detached):

```bash
# Foreground run (used initially):
python3 /workspaces/machine-learning-zoomcamp/05-deployment/workshop/predict.py

# Detached run (recommended to keep running):
nohup uvicorn predict:app --host 0.0.0.0 --port 9696 --app-dir /workspaces/machine-learning-zoomcamp/05-deployment/workshop > /tmp/uvicorn_workshop.log 2>&1 &
echo $! > /tmp/uvicorn_workshop.pid
```

- Verified the endpoint responded:

```bash
curl -s -X POST -H "Content-Type: application/json" -d '{"gender":"female","seniorcitizen":0,...}' http://localhost:9696/predict
# Example response: {"churn_probability":0.6638167617162171,"churn":true}
```

10) File references
- `predict.py` — FastAPI app and entrypoint
- `train.py` — training script that saves `model.bin`
- `Dockerfile` — builds image and runs `uvicorn predict:app` (port 9696)

If you'd like, I can:
- Add a one-line cell to the notebook (workshop-uv-fastapi.ipynb) that checks whether the server is reachable and starts it automatically if not (I can implement this safely so it doesn't spawn multiple servers), or
- Modify the Dockerfile to train the model during image build and provide that Dockerfile change as a patch.

---

End of `readme_new.md`.
