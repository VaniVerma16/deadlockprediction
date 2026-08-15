#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
VM_DIR="$PROJECT_DIR/vm"
BASE_IMAGE="$VM_DIR/noble-server-cloudimg-arm64.img"
CHECKSUMS="$VM_DIR/SHA256SUMS"
DISK_IMAGE="$VM_DIR/deadlock-vm.qcow2"
SEED_DIR="$VM_DIR/seed"
SEED_ISO="$VM_DIR/seed.iso"
IMAGE_URL=${IMAGE_URL:-https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img}
SSH_KEY_FILE=${1:-"$HOME/.ssh/id_ed25519.pub"}

for tool in qemu-img curl hdiutil shasum; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
[[ -f "$SSH_KEY_FILE" ]] || { echo "public SSH key not found: $SSH_KEY_FILE" >&2; exit 1; }

mkdir -p "$VM_DIR" "$SEED_DIR"
if [[ ! -f "$BASE_IMAGE" ]]; then
  curl -L --fail --output "$BASE_IMAGE" "$IMAGE_URL"
fi
curl -L --fail --output "$CHECKSUMS" "${IMAGE_URL%/*}/SHA256SUMS"
EXPECTED=$(awk '$2 == "noble-server-cloudimg-arm64.img" || $2 == "*noble-server-cloudimg-arm64.img" {print $1}' "$CHECKSUMS")
ACTUAL=$(shasum -a 256 "$BASE_IMAGE" | awk '{print $1}')
[[ -n "$EXPECTED" && "$ACTUAL" == "$EXPECTED" ]] || {
  echo "cloud image checksum verification failed" >&2
  exit 1
}
if [[ ! -f "$DISK_IMAGE" ]]; then
  qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$DISK_IMAGE" 24G
fi

cp "$PROJECT_DIR/cloud-init/meta-data" "$SEED_DIR/meta-data"
sed "s|__SSH_PUBLIC_KEY__|$(tr -d '\n' < "$SSH_KEY_FILE")|" \
  "$PROJECT_DIR/cloud-init/user-data" > "$SEED_DIR/user-data"
hdiutil makehybrid -iso -joliet -default-volume-name cidata \
  -o "$SEED_ISO" "$SEED_DIR" >/dev/null

echo "created $DISK_IMAGE"
echo "start it with scripts/qemu/start_vm.sh"
