# tinyrife: RIFE v4.6 in C. Two backends (src/cpu.c, src/gl.c) implement src/rife.h; each tool (pair, dir, video) is
# built once per backend: build/cpu, build/cpu_dir, build/cpu_video and build/gl, build/gl_dir, build/gl_video.
# Needs only a C compiler, make and sh.
CC     ?= gcc
CFLAGS ?= -O3
B      ?= build
WARN   := -Wall -Wextra -Wshadow -Wformat=2 -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes
SAN    := -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
INC    := -Isrc -Ithird_party -I$(B)
CPU    := -march=native -fopenmp
DEPS   := src/rife.h src/util.h $(wildcard third_party/*.h) | $(B)

all: $(B)/cpu $(B)/cpu_dir $(B)/cpu_video $(B)/gl $(B)/gl_dir $(B)/gl_video

# each tool with each backend (-pthread is for the reader/writer threads of dir and video)
CPU_CC = $(CC) $(CFLAGS) $(CPU) -pthread $(INC) $< src/cpu.c -o $@ -lm
GL_CC  = $(CC) $(CFLAGS) -pthread $(INC) $< src/gl.c -o $@ -ldl -lm
$(B)/cpu: src/pair.c src/cpu.c $(DEPS)
	$(CPU_CC)
$(B)/cpu_%: src/%.c src/cpu.c $(DEPS)
	$(CPU_CC)
$(B)/gl: src/pair.c src/gl.c $(B)/shaders.h $(DEPS)
	$(GL_CC)
$(B)/gl_%: src/%.c src/gl.c $(B)/shaders.h $(DEPS)
	$(GL_CC)

# GLSL in shaders/ embedded as C strings
$(B)/shaders.h: $(wildcard shaders/*) | $(B)
	shaders/embed.sh $@

# --- tests ---

# dir and video against a fake backend ("interpolation" = linear blend of the two frames): their file handling,
# threading and error paths are tested with exactly predictable output, no model and no GPU
$(B)/fake_%: src/%.c tests/fake.c $(DEPS)
	$(CC) $(CFLAGS) -pthread $(INC) $< tests/fake.c -o $@ -lm

# ThreadSanitizer is only meaningful against the fake backend: with the real ones it reports libgomp's and Mesa's threads
$(B)/tsan_%: src/%.c tests/fake.c $(DEPS)
	$(CC) -O1 -g -fsanitize=thread -pthread $(INC) $< tests/fake.c -o $@ -lm

$(B)/test_kernels: tests/test_kernels.c src/cpu.c $(DEPS)
	$(CC) $(CFLAGS) $(CPU) $(INC) $< -o $@ -lm

# the checkers are not under test themselves: always plain -O2, so `make sanitize` reports on the programs only
$(B)/pngdiff $(B)/check_dir $(B)/check_video: $(B)/%: tests/%.c $(DEPS)
	$(CC) -O2 $(INC) $< -o $@ -lm

# kernel unit tests (no model needed), then end-to-end golden tests (need ./fetch_model.sh; GL tests skip without a GL 4.3 context)
test: all $(B)/test_kernels $(B)/pngdiff $(B)/check_dir $(B)/check_video $(B)/fake_dir $(B)/fake_video
	$(B)/test_kernels
	BUILD=$(B) tests/run.sh

# the whole test suite again under AddressSanitizer + UBSan, plus the two threaded tools under ThreadSanitizer
sanitize:
	$(MAKE) B=build/san CFLAGS='$(SAN)' build/san/tsan_dir build/san/tsan_video test

# warnings as errors (no conversion warnings: too noisy for numeric code)
lint: $(B)/shaders.h
	$(CC) $(WARN) -Werror $(CPU) -pthread $(INC) -fsyntax-only src/*.c tests/*.c

$(B):
	mkdir -p $@

clean:
	rm -rf build

.PHONY: all test sanitize lint clean
