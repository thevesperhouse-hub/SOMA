#!/usr/bin/env bash
# SOMA entrypoint. Starts an SSH server ONLY if a public key is provided (Vast.ai passes
# $PUBLIC_KEY; otherwise $SSH_PUBLIC_KEY). Without a key, no SSH runs (same behavior as
# before). Then runs the SOMA command passed to the container (serve, train, …).
set -e

KEY="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-}}"
if [ -n "$KEY" ] || [ -s /root/.ssh/authorized_keys ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  if [ -n "$KEY" ]; then
    grep -qxF "$KEY" /root/.ssh/authorized_keys 2>/dev/null || echo "$KEY" >> /root/.ssh/authorized_keys
  fi
  chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
  ssh-keygen -A >/dev/null 2>&1        # host keys if missing
  mkdir -p /run/sshd
  /usr/sbin/sshd
  echo "[soma] sshd started on :22 — SSH tunnel available (ssh -L 8765:localhost:8765 ...)"
fi

# Run SOMA. No args -> interactive launcher (docker run -it), otherwise the subcommand.
exec python cli.py "$@"
