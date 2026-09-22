<div align="center">

# V I V Y

**A small inference engine with bigger ideas.**

An experimental C++ project exploring how language models run, how their memory is managed, and how inference might work across constrained machines.

[![C++20](https://img.shields.io/badge/C%2B%2B-20-1f2937?style=flat-square&logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![GGML](https://img.shields.io/badge/Backend-GGML-0f766e?style=flat-square)](https://github.com/ggml-org/ggml)
[![Status](https://img.shields.io/badge/Status-Early%20development-b45309?style=flat-square)](#project-status)
[![License](https://img.shields.io/badge/License-Apache--2.0-1f2937?style=flat-square)](LICENSE)

[Get started](#get-started) · [How it works](#how-it-works) · [Roadmap](#roadmap)

</div>

---

## The idea

Vivy is a hands-on attempt to build and understand a lightweight language-model inference engine rather than treat model execution as a black box. The long-term goal is to experiment with **CPU inference**, **layer-by-layer weight loading**, and **inference across multiple machines**.

The project uses [GGML](https://github.com/ggml-org/ggml) for tensor operations and backend execution. Vivy will be responsible for the model-specific logic and, eventually, its own execution and memory-management decisions.

> **Project status:** Vivy is currently a CPU tensor-computation experiment. It does **not** load language-model weights, tokenize text, or generate responses yet.

## Get started

You'll need Git, CMake, Make, and a C++20-capable compiler.

```bash
# Clone with the GGML submodule
git clone --recurse-submodules https://github.com/DryBearr/vivy.git
cd vivy

# Build and run the current CPU example
make run
```

If you've already cloned the repository without its submodule, run:

```bash
git submodule update --init --recursive
make run
```

The current program performs a small matrix–vector multiplication and prints:

```text
Vivy v0.1
Result: 17, 39
```

`make` builds the debug executable at `build/vivy`. CMake also exports `build/compile_commands.json` for editors such as Neovim with clangd.

## How it works

The current program follows a deliberately small execution path:

```text
            C++ / Vivy
                 │
      Describe tensors + graph
                 │
          GGML scheduler
                 │
             CPU backend
                 │
           Read the result
```

The tensor metadata lives in a GGML context; the backend allocates storage for the numerical data and executes the computation graph. This small example is the foundation for experimenting with real model weights and more complex graphs.

### Project layout

```text
vivy/
├── main.cpp             # Current CPU tensor experiment
├── CMakeLists.txt       # Build configuration
├── Makefile             # Convenience commands
├── third_party/
│   └── ggml/            # Pinned Git submodule
└── LICENSE              # Vivy's Apache 2.0 license
```

## Roadmap

The roadmap describes **planned work, not implemented features**.

| Stage | Goal | Status |
| :--- | :--- | :--- |
| 01 | Run a computation graph with GGML on the CPU | Current experiment |
| 02 | Read GGUF metadata and load model weights | Planned |
| 03 | Implement and validate a model's forward pass | Planned |
| 04 | Add tokenization, KV cache, and text generation | Planned |
| 05 | Experiment with layer streaming and memory budgets | Research goal |
| 06 | Explore execution across multiple machines | Research goal |

The initial focus is correctness and understanding. Performance comparisons and broader model compatibility can come after a working inference path exists.

## Build notes

```bash
make          # Configure and build in Debug mode
make run      # Build and execute Vivy
make clean    # Clean an already configured build directory
```

GGML is pinned as a Git submodule under `third_party/ggml` so the project can develop against a known dependency revision.

## License

Vivy is licensed under [Apache License 2.0](LICENSE). Its GGML dependency is separately licensed under [MIT](third_party/ggml/LICENSE). Model weights, if added later, may have their own licenses.

---

<div align="center">

<sub>Built to understand the machine, one tensor at a time.</sub>

</div>
