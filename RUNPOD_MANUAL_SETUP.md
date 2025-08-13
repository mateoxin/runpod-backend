RunPod Serverless – Manual Setup Guide

This guide describes how to deploy the Backend to RunPod Serverless and what environment variables are required.

1) Required Environment Variables

Set these as Endpoint Secrets (preferred) or keep Docker defaults and override as needed.

- S3_BUCKET: tqv92ffpc5
- S3_REGION: eu-ro-1
- S3_ENDPOINT_URL: https://s3api-eu-ro-1.runpod.io
- S3_PREFIX: lora-dashboard
- AWS_ACCESS_KEY_ID: your RunPod S3 access key
- AWS_SECRET_ACCESS_KEY: your RunPod S3 secret key
- HF_TOKEN: Hugging Face token (required for model pulls/training). Set this as an Endpoint Secret; do not commit.
- Optional:
  - DEBUG: false
  - MAX_CONCURRENT_JOBS: 10

Notes:
- Dockerfile sets defaults for S3 variables so the service boots with sane values. In production, always override via Endpoint Secrets.

2) Build and Startup

- Image base: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
- Entry point: python -u /rp_handler.py
- Files copied into image:
  - rp_handler.py, utils.py, storage_utils.py, models.py
  - Dependencies from requirements_minimal.txt

3) Deploy via RunPod Console

1. Create Serverless Endpoint
2. Configure Startup Command: python -u /rp_handler.py
3. Set GPU (e.g., A40/RTX A6000/A100)
4. Add Secrets (Environment Variables):
   - HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
     S3_BUCKET, S3_REGION, S3_ENDPOINT_URL, S3_PREFIX,
     (optional) DEBUG, MAX_CONCURRENT_JOBS
5. Save and start the Endpoint

4) Quick Tests (runsync)

- Health:
{ "input": { "type": "health" } }
- Upload training data (example):
{ "input": { "type": "upload_training_data", "training_name": "demo", "files": [ { "filename": "img1.jpg", "content": "<base64>", "content_type": "image/jpeg" } ] } }
- List files:
{ "input": { "type": "list_files" } }
- Download by S3 key:
{ "input": { "type": "download_by_key", "s3_key": "lora-dashboard/results/<subject_id>/images/<file>" } }

Tips:
- For large downloads, use presigned URLs (the backend already returns them).
- download supports process_id and optional s3_key to target specific files.

5) Frontend Compatibility

- Front requests download_by_key or download to get presigned URLs.
- Avoid direct s3:// → https:// conversion on the client; presign on backend.

6) Troubleshooting

- Verify S3 credentials and variables
- Check S3_ENDPOINT_URL and S3_REGION match RunPod S3
- Ensure HF_TOKEN is present if model pulls are required
- Inspect RunPod logs for initialization messages and errors

