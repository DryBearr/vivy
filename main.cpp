#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    // 1. Initialize the CPU backend.
    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr
    );

    if (!backend) {
        std::cerr << "Failed to initialize CPU backend\n";
        return 1;
    }

    // 2. Create a memory arena for tensor metadata.
    const size_t arena_size =
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
        + ggml_graph_overhead();

    std::vector<uint8_t> arena(arena_size);

    ggml_init_params params{};
    params.mem_size = arena.size();
    params.mem_buffer = arena.data();
    params.no_alloc = true;

    ggml_context* ctx = ggml_init(params);

    if (!ctx) {
        std::cerr << "Failed to initialize GGML\n";
        ggml_backend_free(backend);
        return 1;
    }

    // 3. Define our tensors.
    ggml_tensor* weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);

    ggml_tensor* input =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

    // 4. Describe the calculation.
    ggml_tensor* output =
        ggml_mul_mat(ctx, weights, input);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    // 5. Allocate memory for the computation.
    ggml_backend_t backends[] = {backend};

    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        backends, nullptr, 1,
        GGML_DEFAULT_GRAPH_SIZE,
        false, true
    );

    if (!scheduler ||
        !ggml_backend_sched_alloc_graph(scheduler, graph)) {
        std::cerr << "Failed to allocate graph\n";

        if (scheduler) {
            ggml_backend_sched_free(scheduler);
        }

        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // 6. Supply the actual numbers.
    float weight_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float input_data[] = {5.0f, 6.0f};

    ggml_backend_tensor_set(
        weights, weight_data, 0, sizeof(weight_data)
    );

    ggml_backend_tensor_set(
        input, input_data, 0, sizeof(input_data)
    );

    // 7. Execute the computation.
    ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, graph);

    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "Computation failed\n";
        ggml_backend_sched_free(scheduler);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // 8. Read the result from backend memory.
    float result[2]{};

    ggml_backend_tensor_get(
        output, result, 0, sizeof(result)
    );

    std::cout << "Vivy v0.1\n";
    std::cout << "Result: "
              << result[0] << ", "
              << result[1] << '\n';

    // 9. Release allocated resources.
    ggml_backend_sched_free(scheduler);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
