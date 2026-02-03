#!/bin/bash

set -e

uv pip install -e $HOME/repo/

# Hand off to the CMD command
echo "$@"
exec "$@"
