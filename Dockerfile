FROM rust:1.85-bookworm AS builder
WORKDIR /app

# Cache dependencies by building a dummy binary first
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release
RUN rm -rf src

# Build the real binary
COPY src/ src/
COPY assets/ assets/
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/ollama_gateway /usr/local/bin/ollama_gateway
ENTRYPOINT ["ollama_gateway"]
CMD ["--config", "/etc/ollama_gateway/config.toml"]
