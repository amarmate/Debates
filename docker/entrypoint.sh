#!/bin/bash
set -e

# If REPO_URL is set, clone or pull latest repo before running
if [ -n "${REPO_URL}" ]; then
  echo "Fetching latest version of repo: ${REPO_URL}"
  
  # Build auth URL for private HTTPS clone
  AUTH_URL="${REPO_URL}"
  if [ -n "${GITHUB_TOKEN}" ]; then
    if [[ "${REPO_URL}" == https://github.com/* ]]; then
      AUTH_URL="https://${GITHUB_TOKEN}@github.com/${REPO_URL#https://github.com/}"
    elif [[ "${REPO_URL}" == git@github.com:* ]]; then
      AUTH_URL="https://${GITHUB_TOKEN}@github.com/${REPO_URL#git@github.com:}"
    fi
  fi
  
  # SSH: add github.com to known_hosts if using git@
  if [ -d /root/.ssh ] && [[ "${REPO_URL}" == git@* ]]; then
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null || true
  fi
  
  if [ -d /app/repo/.git ]; then
    cd /app/repo && git pull && cd -
  else
    rm -rf /app/repo
    # Use AUTH_URL (with token) for GitHub HTTPS; REPO_URL for SSH or public
    CLONE_URL="${AUTH_URL}"
    if [ -z "${GITHUB_TOKEN}" ] || [[ "${REPO_URL}" == git@* ]]; then
      CLONE_URL="${REPO_URL}"
    fi
    git clone --depth 1 "${CLONE_URL}" /app/repo
  fi
  
  cd /app/repo
  uv sync --no-dev --locked 2>/dev/null || uv sync --no-dev
  # Use shared data mount: /app/data is typically mounted; link so repo sees it
  ln -sfn /app/data /app/repo/data 2>/dev/null || true
  exec "$@"
fi

# Use baked-in code
cd /app
exec "$@"
