.PHONY: build run clean

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
	+cmake --build build

run: build
	./build/vivy

clean:
	cmake --build build --target clean
