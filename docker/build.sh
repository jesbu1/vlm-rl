#!/usr/bin/env bash
# Build the training image from the *committed* tree and push it.
#   docker/build.sh <image-uri>                # e.g. $(python scripts/sagemaker_submit.py --image)
#   PUSH=0 docker/build.sh vlmrl:dev           # build only
# The build context is `git archive HEAD` (tracked files only), so uncommitted edits and untracked files
# (local configs, run outputs, keys) can never end up in the image. Log in to the registry first, e.g.
#   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <registry>
set -euo pipefail
IMAGE=${1:?image uri}
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short=12 HEAD)
if ! git diff --quiet HEAD -- . ':!uv.lock'; then
  echo "note: uncommitted changes are NOT in the image (building $COMMIT)" >&2
fi
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
git archive HEAD | tar -x -C "$CTX"
docker build --platform linux/amd64 -f "$CTX/docker/Dockerfile" --build-arg GIT_COMMIT="$COMMIT" -t "$IMAGE" "$CTX"
if [ "${PUSH:-1}" != "0" ]; then
  docker push "$IMAGE"
fi
echo "$IMAGE"
