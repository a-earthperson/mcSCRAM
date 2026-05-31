FROM node:22-bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json vite.config.ts ./
COPY benchmarks ./benchmarks
COPY derived ./derived
COPY input ./input
COPY plots ./plots
COPY site ./site

ARG VITE_MCSCRAM_WS_PORT=
ARG VITE_MCSCRAM_WS_PATH=/ws
ARG VITE_MCSCRAM_WS_URL=
ARG VITE_BASE_PATH=/
ENV VITE_MCSCRAM_WS_URL=${VITE_MCSCRAM_WS_URL} \
    VITE_MCSCRAM_WS_PORT=${VITE_MCSCRAM_WS_PORT} \
    VITE_MCSCRAM_WS_PATH=${VITE_MCSCRAM_WS_PATH} \
    VITE_BASE_PATH=${VITE_BASE_PATH}

RUN npm run build:app && npm run build:data

EXPOSE 5173

CMD ["npm", "exec", "vite", "--", "preview", "--host", "0.0.0.0", "--port", "5173"]
