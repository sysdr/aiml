FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY src/*.go .
RUN go mod init substrate-freeze-service
RUN go build -o substrate-freeze-app .

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/substrate-freeze-app .
EXPOSE 8080
CMD ["./substrate-freeze-app"]
