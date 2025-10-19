# docker-ml-app/ui/Dockerfile
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
# Debian runtime libs (use libjpeg62-turbo on Debian slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libjpeg62-turbo libpng16-16 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/app
RUN useradd -m -u 10001 appuser
WORKDIR $APP_HOME

# Install deps first (cache-friendly)
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app last for fast code-only rebuilds
COPY --chown=appuser:appuser app ./app

# Parametrize API base URL (override at runtime)
ENV API_URL=http://api:8000

# Gradio port
EXPOSE 7860

USER appuser
CMD ["bash","-lc","python -m app.main"]
