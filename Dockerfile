ARG PYTHON_IMAGE=python:3.10-slim
FROM ${PYTHON_IMAGE}

ARG INSTALL_CARLA=0
ARG CARLA_VERSION=0.9.15

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_CARLA" = "1" ]; then python -m pip install --no-cache-dir "carla==${CARLA_VERSION}"; fi

COPY . .

CMD ["python", "scripts/run_vibe_demo.py"]
