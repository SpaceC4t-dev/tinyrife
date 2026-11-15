# tinyrife: two C programs (CPU and OpenGL compute). Needs only a C compiler, make and sh.
# python/ holds optional helpers (reference, tests, benchmarks, video tools); nothing here depends on it.
CC     ?= gcc
CFLAGS ?= -O3
STB    := third_party/stb_image.h third_party/stb_image_write.h
WARN   := -Wall -Wextra -Wshadow -Wformat=2 -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes
SAN    := -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer

all: build/rife_cpu build/rife_gl

build/rife_cpu: src/rife_cpu.c $(STB) | build
	$(CC) $(CFLAGS) -march=native -fopenmp -Ithird_party $< -o $@ -lm

build/rife_gl: src/rife_gl.c build/shaders.h $(STB) | build
	$(CC) $(CFLAGS) -Ithird_party -Ibuild $< -o $@ -ldl -lm

# GLSL in shaders/ embedded as C strings
build/shaders.h: $(wildcard shaders/*) | build
	shaders/embed.sh $@

build/pngdiff: tests/pngdiff.c third_party/stb_image.h | build
	$(CC) -O2 -Ithird_party $< -o $@ -lm

build/test_kernels: tests/test_kernels.c src/rife_cpu.c $(STB) | build
	$(CC) -O2 -march=native -fopenmp -Ithird_party $< -o $@ -lm

# kernel unit tests (no model needed), then end-to-end golden tests (need ./fetch_model.sh; GL tests skip without a GL 4.3 context)
test: all build/pngdiff build/test_kernels
	build/test_kernels
	tests/run.sh

# --- checking tools (gcc only needs: sanitizers, -Wall..., -fanalyzer; clang-tidy is optional) ---

# everything under AddressSanitizer + UBSan (+ leak check): unit tests and the end-to-end suite
build/san/%: | build/san
build/san/rife_cpu: src/rife_cpu.c $(STB) | build/san
	$(CC) $(SAN) -march=native -fopenmp -Ithird_party $< -o $@ -lm
build/san/rife_gl: src/rife_gl.c build/shaders.h $(STB) | build/san
	$(CC) $(SAN) -Ithird_party -Ibuild $< -o $@ -ldl -lm
build/san/test_kernels: tests/test_kernels.c src/rife_cpu.c $(STB) | build/san
	$(CC) $(SAN) -march=native -fopenmp -Ithird_party $< -o $@ -lm
build/san/pngdiff: tests/pngdiff.c third_party/stb_image.h | build/san
	$(CC) -O2 -Ithird_party $< -o $@ -lm

sanitize: build/san/rife_cpu build/san/rife_gl build/san/test_kernels build/san/pngdiff
	build/san/test_kernels
	BUILD=build/san tests/run.sh

# warnings as errors on the product code and tests (no conversion warnings: too noisy for numeric code)
lint: build/shaders.h
	$(CC) $(WARN) -Werror -march=native -fopenmp -Ithird_party -Ibuild -fsyntax-only src/rife_cpu.c
	$(CC) $(WARN) -Werror -Ithird_party -Ibuild -fsyntax-only src/rife_gl.c
	$(CC) $(WARN) -Wno-missing-prototypes -Werror -march=native -fopenmp -Ithird_party -fsyntax-only tests/test_kernels.c tests/pngdiff.c

# gcc's static analyzer (slow: ~1 minute); only the file you pass is reported on, stb is vendored
analyze: build/shaders.h
	$(CC) -O1 -fanalyzer -march=native -fopenmp -Ithird_party -Ibuild -c src/rife_cpu.c -o /dev/null
	$(CC) -O1 -fanalyzer -Ithird_party -Ibuild -c src/rife_gl.c -o /dev/null

# clang-tidy (optional, see .clang-tidy); OpenMP is off because clang usually has no omp.h, the sources have stubs for that
tidy: build/shaders.h
	clang-tidy --quiet src/rife_cpu.c -- -std=gnu11 -march=native -Ithird_party
	clang-tidy --quiet src/rife_gl.c -- -std=gnu11 -Ithird_party -Ibuild

build:
	mkdir -p $@

build/san:
	mkdir -p $@

clean:
	rm -rf build

.PHONY: all test sanitize lint analyze tidy clean
