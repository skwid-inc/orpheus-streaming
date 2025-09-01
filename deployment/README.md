## Packaged Deployment (image contains models + nginx.conf)

### Build image (from repo root)
```bash
cd orpheus-streaming
export HF_TOKEN=your_huggingface_token_here   # if needed
bash "deployment copy/docker-build-packaged.sh"
```

### Tag & push to ECR
```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 308804103509.dkr.ecr.us-west-2.amazonaws.com
docker tag orpheus-tts:latest 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:packaged
docker push 308804103509.dkr.ecr.us-west-2.amazonaws.com/orpheus-tts:packaged
```

### Deploy via compose (pull and run)
```bash
docker compose -f docker-compose.pull.yml up -d
```

### Monitor
```bash
docker ps
docker logs deployment-orpheus-tts-1-1 --tail 50
```

### Test
```bash
curl -X POST http://localhost:8080/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a benchmark test.",
    "num_runs": 5,
    "warmup": true
  }' | jq .
```

Notes:
- Image bakes HF cache under `/workspace/hf/hub` for main + SNAC models
- Compose sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` and GPU `device_ids`