# Combined Dockerfile for Render (Frontend + Backend in one container)
# ===================================================================

ARG DOCKER_REGISTRY=
ARG GHCR_REGISTRY=ghcr.io/

# Stage 1: Build Frontend
FROM ${DOCKER_REGISTRY:-}node:18-alpine AS frontend-builder

WORKDIR /app

ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --frozen-lockfile || npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Install uv
FROM ${GHCR_REGISTRY}astral-sh/uv:latest AS uv

# Stage 3: Final image with Python + Nginx
FROM ${DOCKER_REGISTRY:-}python:3.10-slim

WORKDIR /app

# Install system dependencies (nginx + curl + envsubst)
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from uv stage
COPY --from=uv /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Copy and install Python dependencies
COPY pyproject.toml uv.lock* ./
ENV UV_HTTP_TIMEOUT=300
RUN if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi

# Copy backend code
COPY backend/ ./backend/

# Create necessary directories
RUN mkdir -p /app/backend/instance /app/uploads

# Copy frontend build to nginx
COPY --from=frontend-builder /app/dist /usr/share/nginx/html

# Create nginx config template (will substitute PORT at runtime)
RUN mkdir -p /etc/nginx/templates && echo 'server { \n\
    listen ${PORT}; \n\
    server_name _; \n\
    \n\
    location / { \n\
        root /usr/share/nginx/html; \n\
        index index.html; \n\
        try_files $uri $uri/ /index.html; \n\
    } \n\
    \n\
    location /api { \n\
        proxy_pass http://127.0.0.1:5001; \n\
        proxy_set_header Host $host; \n\
        proxy_set_header X-Real-IP $remote_addr; \n\
    } \n\
    \n\
    location /files { \n\
        proxy_pass http://127.0.0.1:5001; \n\
        proxy_set_header Host $host; \n\
        proxy_set_header X-Real-IP $remote_addr; \n\
    } \n\
    \n\
    location /health { \n\
        proxy_pass http://127.0.0.1:5001; \n\
    } \n\
}' > /etc/nginx/templates/default.conf.template

# Remove default nginx config
RUN rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf

ENV PYTHONPATH=/app
ENV FLASK_APP=backend/app.py
ENV IN_DOCKER=1
ENV PORT=10000

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Start: substitute PORT in nginx config, run migrations, start backend, start nginx
CMD sh -c "\
    envsubst '\$PORT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && \
    uv run --directory backend alembic upgrade head && \
    uv run --directory backend python app.py & \
    sleep 3 && \
    nginx -g 'daemon off;'"
