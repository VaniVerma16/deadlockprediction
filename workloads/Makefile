CC ?= cc
CFLAGS ?= -O0 -g -Wall -Wextra -Werror -std=c11
LDFLAGS ?=
BIN_DIR ?= ../bin
PROGRAMS := abba nway dining

.PHONY: all clean

all: $(PROGRAMS:%=$(BIN_DIR)/%)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/%: %.c common.c common.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $< common.c -pthread $(LDFLAGS) -o $@

clean:
	rm -f $(PROGRAMS:%=$(BIN_DIR)/%)

