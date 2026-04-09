.PHONY: build run up shell logs restart clean help transcribe cli health

# Default target
.DEFAULT_GOAL := help

help:
	@echo "GLM-ASR Docker Development Commands"
	@echo ""
	@echo "Usage: make [target] [args]"
	@echo ""
	@echo "Container Management:"
	@echo "  build       Build the Docker image"
	@echo "  run         Run the container with GPU support"
	@echo "  up          Build and run (production best practice)"
	@echo "  shell       Open a shell in a running container"
	@echo "  logs        Show container logs"
	@echo "  restart     Restart the containers"
	@echo "  clean       Stop and remove the container"
	@echo ""
	@echo "CLI Commands (require running container):"
	@echo "  transcribe  Transcribe audio file(s)"
	@echo "  cli         Show CLI help"
	@echo "  health      Check server health"
	@echo ""
	@echo "Examples:"
	@echo "  make transcribe input.mp3"
	@echo "  make transcribe input.mp3 OUTPUT=output.txt"
	@echo "  make transcribe /data/audio/ LANGUAGE=en"
	@echo ""
	@echo "Or use docker compose exec directly:"
	@echo "  docker compose exec glm-asr glm-asr transcribe /app/input.mp3"

build:
	docker build -t glm-asr .

run:
	docker compose up -d

up:
	docker compose up -d --build --remove-orphans

shell:
	docker exec -it glm-asr /bin/bash

logs:
	docker compose logs -f

restart:
	docker compose restart

clean:
	docker compose down
	docker rmi glm-asr || true

# CLI commands
transcribe:
	docker compose exec glm-asr python glm_asr_cli.py transcribe $(INPUT) $(if $(OUTPUT),-o $(OUTPUT),) $(if $(LANGUAGE),-l $(LANGUAGE),) $(if $(FORMAT),-f $(FORMAT),)

cli:
	docker compose exec glm-asr python glm_asr_cli.py --help

health:
	docker compose exec glm-asr python glm_asr_cli.py health
