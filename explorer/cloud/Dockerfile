FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime@sha256:2428b92ebbaeceba5572b98c18c8a94e43162bead6e88588ad54471147c58a20
WORKDIR /app
ENV PYTHONPATH=/app/src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 MUQ_MODEL_DIR=/app/checkpoint PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torchaudio==2.6.0 torchvision==0.21.0 && python -m pip install --no-cache-dir muq==0.1.0 transformers==4.57.6 'numpy<3' pandas scipy soundfile
COPY src/afp /app/src/afp
COPY explorer/serve.py explorer/gpu_match.py explorer/search_trace.py /app/explorer/
COPY data/medium /app/data/medium
COPY checkpoint /app/checkpoint
EXPOSE 8080
CMD ["python","explorer/serve.py","--cloud","--port","8080"]
