# Name of the container and image
NAME := bessie
VERSION := $(shell git describe --always --abbrev=0)

# Default build target
.PHONY: build
build:
	podman build \
		--build-arg UID=$$(id -u) \
		--build-arg GID=$$(id -g) \
		--tag $(NAME):$(VERSION) \
		--tag $(NAME):latest \
		.

# Start the container detached (e.g. for background server)
# FIXME: git does not work in the container right now, not a big deal
.PHONY: up
up:
	@if ! podman volume exists $(NAME)-bash-history 2>/dev/null; then \
		podman volume create $(NAME)-bash-history; \
	fi

	podman pod create \
		--name $(NAME)-pod \
		--userns keep-id

	podman run -d \
		--pod $(NAME)-pod \
		--name $(NAME) \
		--volume $(NAME)-bash-history:/home/ubuntu/bash_history \
		--volume ./:/home/ubuntu/repo/:z \
		--volume ~/.claude:/home/ubuntu/.claude:z \
		--volume ~/.claude.json:/home/ubuntu/.claude.json:z \
		--volume /app/.venv \
		--workdir /home/ubuntu/repo \
		$(NAME):latest

# Stop and remove a named container
.PHONY: down
down:
	@if podman pod exists $(NAME)-pod 2>/dev/null; then \
		podman pod stop $(NAME)-pod --time 1; \
		podman pod rm $(NAME)-pod; \
	fi

# Shell into the container (must already be running)
.PHONY: shell
shell:
	podman exec -it $(NAME) bash

# Restart the container
.PHONY: restart
restart: down up
