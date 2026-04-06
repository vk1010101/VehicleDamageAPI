# 1. Base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 2. System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama binary directly to a safe path
# Using -f (fail) and -L (follow) to ensure we don't save error pages as binaries
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
    && chmod +x /usr/local/bin/ollama

# 4. Python dependencies
# First install the specific numpy version to avoid NumPy 2.x breaking everything
# We explicitly install the core dependencies for damage_service.py
RUN pip install --no-cache-dir \
    "numpy<2" \
    runpod \
    ultralytics \
    opencv-python-headless \
    requests \
    pillow

# 5. Copy vision AI files
COPY damage_service.py car_damage.pt handler.py .

# 6. "Ultimate" Entrypoint: Super resilient logs + error catching
RUN printf '#!/bin/bash\n\
echo "--- CONTAINER STARTING ---"\n\
export PATH=$PATH:/usr/local/bin\n\
echo "Starting Ollama daemon..."\n\
ollama serve > /app/ollama.log 2>&1 &\n\
\n\
echo "Waiting for Ollama to wake up..."\n\
# Increase wait time or better yet, loop until responsive\n\
for i in {1..20}; do\n\
    if curl -s http://localhost:11434/api/tags > /dev/null; then\n\
        echo "Ollama is UP."\n\
        break\n\
    fi\n\
    echo "Waiting for Ollama... ($i/20)"\n\
    sleep 3\n\
done\n\
\n\
echo "Current Ollama status:"\n\
ollama --version || echo "Ollama NOT FOUND in path"\n\
\n\
echo "Attempting to pull gemma3:4b (this may take 5 mins)..."\n\
if ollama pull gemma3:4b; then\n\
    echo "Model pull SUCCESS."\n\
else\n\
    echo "Model pull FAILED - check network. Continuing anyway to see logs."\n\
fi\n\
\n\
echo "Starting Python handler.py..."\n\
python -u /app/handler.py\n' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Use bash explicitly
CMD ["/bin/bash", "/app/entrypoint.sh"]
