# 1. Use RunPod's pre-configured PyTorch image (stable, has CUDA)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. Set the working directory
WORKDIR /app

# 3. Install basic system tools
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. DIRECT OLLAMA DOWNLOAD (Bypasses the "Install Script" entirely)
# This is the single-file binary for Linux AMD64
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# 5. Pre-download the vision model (so it's baked into the image)
RUN (ollama serve &) && sleep 5 && ollama pull gemma3:4b

# 6. Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    opencv-python-headless \
    requests \
    pillow

# 7. Copy ONLY vision AI code (re-using your existing logic)
COPY damage_service.py car_damage.pt handler.py .

# 8. Start-up script (Ollama + Handler)
RUN echo "#!/bin/bash\nollama serve &\nsleep 5\npython -u /app/handler.py" > /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
