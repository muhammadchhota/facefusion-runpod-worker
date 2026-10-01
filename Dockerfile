FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG FACEFUSION_VERSION=3.8.2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

WORKDIR /facefusion

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        pip \
        python-is-python3 \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "${FACEFUSION_VERSION}" --depth 1 \
      https://github.com/facefusion/facefusion.git . \
    && python install.py cuda@12 --skip-conda \
    && pip install --no-cache-dir "runpod==1.7.13" "requests==2.33.0"

# Bake only the models used by the face-swap pipeline. `force-download` also
# fetches models for unrelated processors, producing an unnecessarily large
# serverless image and much slower cold starts.
RUN curl -fsSL \
      https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/source.jpg \
      -o /tmp/source.jpg \
    && curl -fsSL \
      https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-240p.mp4 \
      -o /tmp/target.mp4 \
    && ffmpeg -loglevel error -i /tmp/target.mp4 -frames:v 1 /tmp/target.jpg \
    && python facefusion.py headless-run \
      --processors face_swapper \
      --face-swapper-model hyperswap_1a_256 \
      --execution-providers cpu \
      --source-paths /tmp/source.jpg \
      --target-path /tmp/target.jpg \
      --output-path /tmp/output-hyperswap.jpg \
    && python facefusion.py headless-run \
      --processors face_swapper \
      --face-swapper-model ghost_1_256 \
      --execution-providers cpu \
      --source-paths /tmp/source.jpg \
      --target-path /tmp/target.jpg \
      --output-path /tmp/output-ghost-1.jpg \
    && python facefusion.py headless-run \
      --processors face_swapper \
      --face-swapper-model ghost_3_256 \
      --execution-providers cpu \
      --source-paths /tmp/source.jpg \
      --target-path /tmp/target.jpg \
      --output-path /tmp/output-ghost-3.jpg \
    && rm -f /tmp/source.jpg /tmp/target.mp4 /tmp/target.jpg /tmp/output-*.jpg

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
