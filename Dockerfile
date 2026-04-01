# 1. Base image with CUDA
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 2. System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama binary directly (no script, no sudo needed)
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

# 4. Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    opencv-python-headless \
    requests \
    pillow

# 5. Copy vision AI files
COPY damage_service.py car_damage.pt handler.py .

# 6. Entrypoint: starts Ollama, pulls the model at RUNTIME (when GPU is available),
#    then starts the Python handler
RUN printf '#!/bin/bash\nset -e\nollama serve &\nOLLAMA_PID=$!\nsleep 8\necho "Pulling gemma3:4b model..."\nollama pull gemma3:4b\necho "Model ready. Starting handler."\npython -u /app/handler.py\n' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

CMD ["/bin/bash", "/app/entrypoint.sh"]
