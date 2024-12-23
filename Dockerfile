FROM ghcr.io/astral-sh/uv:python3.12-alpine
LABEL authors="wwagner4"

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --frozen

RUN mkdir -p /.cache
RUN chmod -R 777 /.cache
RUN mkdir -p /.config
RUN chmod -R 777 /.config
