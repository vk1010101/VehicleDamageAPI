# 1. Use RunPod's pre-configured PyTorch image (stable, has CUDA)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. Set the working directory
WORKDIR /app

# 3. Install Ollama and system dependencies for vision/AI
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Pre-download the vision model (so it's baked into the image)
# We briefly start the daemon to pull the model during the image build
RUN (ollama serve &) && sleep 5 && ollama pull gemma3:4b

# 5. Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    opencv-python-headless \
    requests \
    pillow

# 6. Copy only the necessary vision AI logic
COPY damage_service.py .
COPY car_damage.pt .
COPY handler.py .

# 7. Add an entrypoint script to start the Ollama daemon before the handler
RUN echo "#!/bin/bash\nollama serve &\nsleep 5\npython -u /app/handler.py" > /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Start the API handler
CMD ["/app/entrypoint.sh"]
