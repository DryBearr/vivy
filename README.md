# Vivy

C++20 project using [GGML](https://github.com/ggml-org/ggml) on the CPU.

Requires a C++ compiler, CMake, Make, and Git.

```sh
git submodule update --init --recursive
make run
```

`make` builds `build/vivy` with debug symbols. `make clean` removes compiled outputs.
Neovim's clangd uses `build/compile_commands.json` for GGML headers and compiler flags.

GGML is pinned by the `third_party/ggml` Git submodule. Its MIT license is kept in
[`third_party/ggml/LICENSE`](third_party/ggml/LICENSE) and copied to `build/GGML-LICENSE`.
