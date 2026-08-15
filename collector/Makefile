CLANG ?= clang
CC ?= cc
BPFTOOL ?= bpftool
ARCH ?= arm64
BUILD_DIR ?= ../build/collector
TARGET_ARCH_arm64 := arm64
TARGET_ARCH_aarch64 := arm64
TARGET_ARCH_x86_64 := x86
TARGET_ARCH := $(TARGET_ARCH_$(ARCH))
CFLAGS := -O2 -g -Wall -Wextra -Werror
LIBS := -lbpf -lelf -lz -ldl

.PHONY: all clean

all: $(BUILD_DIR)/deadlock-collector

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/vmlinux.h: | $(BUILD_DIR)
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

$(BUILD_DIR)/deadlock.bpf.o: bpf/deadlock.bpf.c events.h $(BUILD_DIR)/vmlinux.h
	$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(TARGET_ARCH) \
		-I$(BUILD_DIR) -I. -c $< -o $@

$(BUILD_DIR)/deadlock.skel.h: $(BUILD_DIR)/deadlock.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

$(BUILD_DIR)/deadlock-collector: collector.c events.h $(BUILD_DIR)/deadlock.skel.h
	$(CC) $(CFLAGS) -I$(BUILD_DIR) -I. collector.c $(LIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)
