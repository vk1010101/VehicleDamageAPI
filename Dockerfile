# 1. Base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 2. System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama binary directly to a safe path
RUN curl -fsSL https://github.com/ollama/ollama/releases/download/v0.20.2/ollama-linux-amd64.tar.zst | zstd -d | tar -xf - -C /usr

# 4. BAKE the model into the image at build time
#    Start Ollama, pull the model, then stop Ollama. The model weights
#    end up in /root/.ollama/models and ship with every container.
#    This eliminates the 3+ min cold-start download entirely.
RUN ollama serve & \
    SERVER_PID=$! && \
    sleep 5 && \
    for i in $(seq 1 30); do \
        curl -s http://localhost:11434/api/tags > /dev/null && break; \
        echo "Waiting for Ollama to start... ($i/30)"; \
        sleep 2; \
    done && \
    echo "Pulling gemma3:4b into image layer..." && \
    ollama pull gemma3:4b && \
    echo "Model baked successfully!" && \
    kill $SERVER_PID && \
    wait $SERVER_PID 2>/dev/null || true

# 5. Python dependencies
RUN pip install --no-cache-dir \
    "numpy<2" \
    runpod \
    ultralytics \
    opencv-python-headless \
    requests \
    pillow

# 6. Copy vision AI files
COPY damage_service.py car_damage.pt handler.py .

# 7. Entrypoint — NO model pull needed, just start Ollama and the handler
RUN printf '#!/bin/bash\n\
echo "--- CONTAINER STARTING ---"\n\
export PATH=$PATH:/usr/local/bin\n\
echo "Starting Ollama daemon..."\n\
ollama serve > /app/ollama.log 2>&1 &\n\
\n\
echo "Waiting for Ollama to be ready..."\n\
for i in {1..30}; do\n\
    if curl -s http://localhost:11434/api/tags > /dev/null; then\n\
        echo "Ollama is UP."\n\
        break\n\
    fi\n\
    echo "Waiting for Ollama... ($i/30)"\n\
    sleep 1\n\
done\n\
\n\
echo "Verifying model is present..."\n\
ollama list\n\
\n\
echo "Starting Python handler.py..."\n\
python -u /app/handler.py\n' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Use bash explicitly
CMD ["/bin/bash", "/app/entrypoint.sh"]
