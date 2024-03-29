#!/bin/bash

set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

export D_UID=$UID
export D_GID=$GID

./patch.sh
docker compose up llamacpp-builder --exit-code-from llamacpp-builder

git submodule foreach git reset --hard
