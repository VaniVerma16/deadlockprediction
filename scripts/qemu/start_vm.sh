#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
VM_DIR="$PROJECT_DIR/vm"
DISK_IMAGE="$VM_DIR/deadlock-vm.qcow2"
SEED_ISO="$VM_DIR/seed.iso"
QEMU_BIN=${QEMU_BIN:-qemu-system-aarch64}
FIRMWARE=${FIRMWARE:-}

command -v "$QEMU_BIN" >/dev/null || { echo "install QEMU first (brew install qemu)" >&2; exit 1; }
[[ -f "$DISK_IMAGE" && -f "$SEED_ISO" ]] || { echo "run scripts/qemu/create_vm.sh first" >&2; exit 1; }

if [[ -z "$FIRMWARE" ]] && command -v brew >/dev/null; then
  QEMU_PREFIX=$(brew --prefix qemu)
  FIRMWARE=$(find "$QEMU_PREFIX/share/qemu" -name 'edk2-aarch64-code.fd' -print -quit)
fi
[[ -f "$FIRMWARE" ]] || { echo "set FIRMWARE to edk2-aarch64-code.fd" >&2; exit 1; }

exec "$QEMU_BIN" \
  -machine virt,accel=hvf,highmem=on \
  -cpu host -smp 4 -m 4096 \
  -bios "$FIRMWARE" \
  -drive "if=virtio,format=qcow2,file=$DISK_IMAGE" \
  -drive "if=virtio,format=raw,readonly=on,file=$SEED_ISO" \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -nographic

