FROM golang:1.22-alpine AS builder

WORKDIR /src

COPY coordinator/go.mod ./coordinator/go.mod
COPY coordinator/src ./coordinator/src

WORKDIR /src/coordinator
RUN go build -o /out/axon-coordinator ./src

FROM alpine:3.20

WORKDIR /app

COPY --from=builder /out/axon-coordinator /usr/local/bin/axon-coordinator

EXPOSE 8000

ENTRYPOINT ["axon-coordinator"]
