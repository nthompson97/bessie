# syntax=docker/dockerfile:1.3.1

FROM dev-environment:latest

# Installation for UV + python environment
# https://docs.astral.sh/uv/getting-started/installation/
# https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion
ENV UV_VERSION="0.9.26"
RUN curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh && \
    echo 'eval "$(uv generate-shell-completion bash)"' >> $HOME/.bashrc

# Install the uv project
# https://docs.astral.sh/uv/guides/integration/docker/#installing-a-project
# https://github.com/astral-sh/uv-docker-example
WORKDIR /app
ENV UV_NO_DEV=1 UV_LINK_MODE=copy UV_WORKING_DIR=/app

# FIXME: get caching to work, and enable this step
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --locked --no-install-project

COPY --chown=$USER:$USER pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY --chown=$USER:$USER . /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR $HOME

# Use the entrypoint script to install the project in editable mode
COPY --chown=$USER:$USER entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
