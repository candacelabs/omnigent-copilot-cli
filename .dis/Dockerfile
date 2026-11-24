FROM ghcr.io/github/copilot-cli:1.0.114@sha256:5083f9892836b4d1cc975386b0fbe9ded2658f22ae43f81c0a36fe917f85b1a9 AS copilot
FROM ghcr.io/astral-sh/uv:0.13.9-python3.12-trixie-slim@sha256:b12436aceed37b21524b327362bd67e73a284b5d52f4f6be0e047dd8e4c4eb90

COPY --from=copilot /usr/local/bin/copilot /usr/local/bin/copilot
COPY --from=copilot /usr/share/licenses/copilot-cli/ /usr/share/licenses/copilot-cli/
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap git procps tini tmux \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace

# dis mounts the checkout at /workspace. Keep its host virtualenv and caches separate.
ENV UV_PROJECT_ENVIRONMENT=/opt/omnigent-venv \
    PATH="/opt/omnigent-venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest-cache" \
    RUFF_CACHE_DIR=/tmp/ruff-cache \
    COPILOT_AUTO_UPDATE=false \
    OMNIGENT_SKIP_WEB_UI=true

LABEL dis.shell="/bin/bash"
LABEL dis.post_create="uv sync --frozen --group dev"
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
