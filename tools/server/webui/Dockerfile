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

ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=${VITE_BASE_PATH}

RUN npm run build:ssr

EXPOSE 5173

CMD ["npm", "run", "start:ssr"]
