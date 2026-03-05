# Ollama Gateway

An authenticated reverse proxy for [Ollama](https://ollama.com) with [Langfuse](https://langfuse.com) tracing and a web-based admin UI.

## Features

- Bearer-token authentication for all Ollama endpoints
- Langfuse tracing for `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`
- Runtime management via web admin UI (tokens + Langfuse settings)
- Configuration persisted to TOML on every admin change

## Quick Start

```bash
cp config.example.toml config.toml
# Edit config.toml as needed

ADMIN_PASSWORD=secret cargo run -- --config config.toml
```

Navigate to `http://localhost:8080/admin/` and enter your admin password.

## Configuration

See [`config.example.toml`](config.example.toml) for all options.

```toml
[ollama]
upstream_url = "http://localhost:11434"

[langfuse]
enabled = false
host    = "https://cloud.langfuse.com"
public_key = ""
secret_key = ""

[[tokens]]
token    = "sk-myapp-abc123"
app_name = "my-app"

[server]
listen_addr = "0.0.0.0"
listen_port = 8080
```

## Environment Variables

| Variable         | Default | Description                                                                 |
|------------------|---------|-----------------------------------------------------------------------------|
| `ADMIN_PASSWORD` | *(none)*| Password for the `/admin/` web UI (Basic Auth). Strongly recommended.       |
| `PROXY_PORT`     | `8080`  | Port the Ollama proxy listens on. Overrides `server.listen_port` in config. |
| `ADMIN_PORT`     | `8081`  | Port the admin UI listens on. Overrides `server.admin_port` in config.      |
| `RUST_LOG`       | `info`  | Log level filter. Use `debug` for request/Langfuse flush details.           |

## Docker

```bash
docker run -d \
  -p 8080:8080 \
  -p 8081:8081 \
  -e ADMIN_PASSWORD=secret \
  -e PROXY_PORT=8080 \
  -e ADMIN_PORT=8081 \
  -v /path/to/config.toml:/etc/ollama_gateway/config.toml \
  avirtuos/ollama_gateway:latest
```

## Portainer Stack

Paste the following into **Portainer → Stacks → Add stack → Web editor**. Adjust the volume path and environment variables to suit your environment.

```yaml
version: "3.8"

services:
  ollama_gateway:
    image: avirtuos/ollama_gateway:latest
    restart: unless-stopped
    ports:
      - "${PROXY_PORT:-8080}:8080"
      - "${ADMIN_PORT:-8081}:8081"
    environment:
      - ADMIN_PASSWORD=${ADMIN_PASSWORD:?ADMIN_PASSWORD is required}
      - PROXY_PORT=${PROXY_PORT:-8080}
      - ADMIN_PORT=${ADMIN_PORT:-8081}
      - RUST_LOG=${RUST_LOG:-info}
    volumes:
      - /opt/ollama_gateway/config.toml:/etc/ollama_gateway/config.toml
```

**Stack environment variables** (set these in Portainer's "Environment variables" panel below the editor):

| Variable         | Example          | Description                                    |
|------------------|------------------|------------------------------------------------|
| `ADMIN_PASSWORD` | `changeme`       | Admin UI password — required                   |
| `PROXY_PORT`     | `8080`           | Host port for the Ollama proxy                 |
| `ADMIN_PORT`     | `8081`           | Host port for the admin UI                     |
| `RUST_LOG`       | `info`           | Log verbosity (`error`, `warn`, `info`, `debug`)|

> **Note:** The config file at `/opt/ollama_gateway/config.toml` on the host must exist before starting the stack. Copy [`config.example.toml`](config.example.toml) as a starting point.

## Admin UI

Browse to `/admin/` (e.g. `http://localhost:8080/admin/`). The browser will prompt for credentials — use username `admin` and the value of `ADMIN_PASSWORD`.

From the UI you can:
- Enable/disable Langfuse and update all Langfuse settings
- Add or remove Bearer tokens at runtime

All changes are persisted immediately to the TOML config file.

## CI/CD — GitHub Secrets Required

The GitHub Actions workflow (`.github/workflows/docker.yml`) publishes the image to Docker Hub on every merge to `main`. Configure these secrets in your repository settings (**Settings → Secrets and variables → Actions**):

| Secret               | Description                                                      |
|----------------------|------------------------------------------------------------------|
| `DOCKERHUB_USERNAME` | Docker Hub username (e.g. `avirtuos`)                           |
| `DOCKERHUB_TOKEN`    | Docker Hub access token — generate at hub.docker.com → Account Settings → Security |

The workflow produces two tags on each push:
- `avirtuos/ollama_gateway:latest`
- `avirtuos/ollama_gateway:<git-sha>`
