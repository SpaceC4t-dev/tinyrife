# tinyrife: two C programs (CPU and OpenGL compute). Needs only a C compiler, make and sh.
# python/ holds optional helpers (reference, tests, benchmarks, video tools); nothing here depends on it.
CC     ?= gcc
CFLAGS ?= -O3
STB    := third_party/stb_image.h third_party/stb_image_write.h
WARN   := -Wall -Wextra -Wshadow -Wformat=2 -Wundef -Wpointer-arith -Wcast-align -Wstrict-prototypes
SAN    := -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer

all: build/cpu build/gl build/gl_dir

build/cpu: src/cpu.c $(STB) | build
	$(CC) $(CFLAGS) -march=native -fopenmp -Ithird_party $< -o $@ -lm

GLHOST := src/gl_host.c src/gl_host.h build/shaders.h

build/gl: src/gl.c $(GLHOST) $(STB) | build
	$(CC) $(CFLAGS) -pthread -Ithird_party -Isrc -Ibuild src/gl.c src/gl_host.c -o $@ -ldl -lm

build/gl_dir: src/gl_dir.c $(GLHOST) $(STB) | build
	$(CC) $(CFLAGS) -pthread -Ithird_party -Isrc -Ibuild src/gl_dir.c src/gl_host.c -o $@ -ldl -lm

# GLSL in shaders/ embedded as C strings
build/shaders.h: $(wildcard shaders/*) | build
	shaders/embed.sh $@

build/pngdiff: tests/pngdiff.c third_party/stb_image.h | build
	$(CC) -O2 -Ithird_party $< -o $@ -lm

# gl_dir built against a fake GL host (tests/fake_gl_host.c: "interpolation" = average of the two frames): tests the
# directory handling and threading with exactly predictable output and no GPU
build/gl_dir_fake: src/gl_dir.c tests/fake_gl_host.c src/gl_host.h $(STB) | build
	$(CC) -O2 -pthread -Ithird_party -Isrc src/gl_dir.c tests/fake_gl_host.c -o $@ -lm

build/check_dir: tests/check_dir.c third_party/stb_image.h | build
	$(CC) -O2 -Ithird_party $< -o $@ -lm

build/test_kernels: tests/test_kernels.c src/cpu.c $(STB) | build
	$(CC) -O2 -march=native -fopenmp -Ithird_party $< -o $@ -lm

# kernel unit tests (no model needed), then end-to-end golden tests (need ./fetch_model.sh; GL tests skip without a GL 4.3 context)
test: all build/pngdiff build/test_kernels build/gl_dir_fake build/check_dir
	build/test_kernels
	tests/run.sh

# --- checking tools (gcc only needs: sanitizers, -Wall..., -fanalyzer; clang-tidy is optional) ---

build/san/%: | build/san
build/san/cpu: src/cpu.c $(STB) | build/san
	$(CC) $(SAN) -march=native -fopenmp -Ithird_party $< -o $@ -lm
build/san/gl: src/gl.c $(GLHOST) $(STB) | build/san
	$(CC) $(SAN) -pthread -Ithird_party -Isrc -Ibuild src/gl.c src/gl_host.c -o $@ -ldl -lm
build/san/gl_dir: src/gl_dir.c $(GLHOST) $(STB) | build/san
	$(CC) $(SAN) -pthread -Ithird_party -Isrc -Ibuild src/gl_dir.c src/gl_host.c -o $@ -ldl -lm
build/san/test_kernels: tests/test_kernels.c src/cpu.c $(STB) | build/san
	$(CC) $(SAN) -march=native -fopenmp -Ithird_party $< -o $@ -lm
build/san/pngdiff: tests/pngdiff.c third_party/stb_image.h | build/san
	$(CC) -O2 -Ithird_party $< -o $@ -lm
build/san/check_dir: tests/check_dir.c third_party/stb_image.h | build/san
	$(CC) -O2 -Ithird_party $< -o $@ -lm
build/san/gl_dir_fake: src/gl_dir.c tests/fake_gl_host.c src/gl_host.h $(STB) | build/san
	$(CC) $(SAN) -pthread -Ithird_party -Isrc src/gl_dir.c tests/fake_gl_host.c -o $@ -lm
# ThreadSanitizer, only meaningful against the fake host: with the real one it reports Mesa's own internal threads
build/san/gl_dir_fake_tsan: src/gl_dir.c tests/fake_gl_host.c src/gl_host.h $(STB) | build/san
	$(CC) -O1 -g -fsanitize=thread -pthread -Ithird_party -Isrc src/gl_dir.c tests/fake_gl_host.c -o $@ -lm

sanitize: build/san/cpu build/san/gl build/san/gl_dir build/san/test_kernels build/san/pngdiff \
          build/san/check_dir build/san/gl_dir_fake build/san/gl_dir_fake_tsan
	build/san/test_kernels
	BUILD=build/san tests/run.sh

# warnings as errors on the product code and tests (no conversion warnings: too noisy for numeric code)
lint: build/shaders.h
	$(CC) $(WARN) -Werror -march=native -fopenmp -Ithird_party -Ibuild -fsyntax-only src/cpu.c
	$(CC) $(WARN) -Werror -pthread -Ithird_party -Isrc -Ibuild -fsyntax-only src/gl_host.c src/gl.c src/gl_dir.c
	$(CC) $(WARN) -Wno-missing-prototypes -Werror -march=native -fopenmp -Ithird_party -fsyntax-only tests/test_kernels.c tests/pngdiff.c tests/check_dir.c
	$(CC) $(WARN) -Werror -pthread -Ithird_party -Isrc -fsyntax-only tests/fake_gl_host.c

# gcc's static analyzer (slow: ~1 minute); only the file you pass is reported on, stb is vendored
analyze: build/shaders.h
	$(CC) -O1 -fanalyzer -march=native -fopenmp -Ithird_party -Ibuild -c src/cpu.c -o /dev/null
	$(CC) -O1 -fanalyzer -pthread -Ithird_party -Isrc -Ibuild -c src/gl_host.c -o /dev/null
	$(CC) -O1 -fanalyzer -pthread -Ithird_party -Isrc -Ibuild -c src/gl.c -o /dev/null
	$(CC) -O1 -fanalyzer -pthread -Ithird_party -Isrc -Ibuild -c src/gl_dir.c -o /dev/null

# clang-tidy (optional, see .clang-tidy); OpenMP is off because clang usually has no omp.h, the sources have stubs for that
tidy: build/shaders.h
	clang-tidy --quiet src/cpu.c -- -std=gnu11 -march=native -Ithird_party
	clang-tidy --quiet src/gl_host.c src/gl.c src/gl_dir.c -- -std=gnu11 -pthread -Ithird_party -Isrc -Ibuild

build:
	mkdir -p $@

build/san:
	mkdir -p $@

clean:
	rm -rf build

.PHONY: all test sanitize lint analyze tidy clean
