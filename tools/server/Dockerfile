FROM ghcr.io/a-earthperson/mcscram:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app/server

COPY src/server/pyproject.toml src/server/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/server/src ./src
RUN uv sync --frozen --no-dev
COPY input/* /app/input/

ENV PATH="/app/server/.venv/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.12/dist-packages" \
    PYTHONUNBUFFERED=1 \
    ACPP_VISIBILITY_MASK=cuda \
    MCSCRAM_INPUT_FILES=/app/input/SmallTree.xml \
    MCSCRAM_NUM_TRIALS=0 \
    MCSCRAM_EARLY_STOP=false \
    MCSCRAM_BURN_IN=0 \
    MCSCRAM_DELTA=0.000000001

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--frozen", "mcscram-server"]
