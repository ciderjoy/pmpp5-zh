#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING

#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr unsigned unvisited = std::numeric_limits<unsigned>::max();
constexpr unsigned private_capacity = 256;

struct GraphView {
    const unsigned* row_ptr;
    const unsigned* col_idx;
    unsigned vertices;
    unsigned edges;
};

__global__ void bfs_one_grid_barrier(GraphView graph, unsigned* level,
                                     unsigned* previous_frontier,
                                     unsigned* current_frontier,
                                     unsigned* frontier_sizes) {
    const cg::grid_group grid = cg::this_grid();
    unsigned previous_count = frontier_sizes[0];
    for (unsigned current_level = 1; previous_count != 0; ++current_level) {
        __shared__ unsigned private_frontier[private_capacity];
        __shared__ unsigned private_count;
        __shared__ unsigned public_base;
        if (threadIdx.x == 0) {
            private_count = 0;
        }
        __syncthreads();

        for (unsigned long long item = grid.thread_rank();
             item < previous_count; item += grid.size()) {
            const unsigned source = previous_frontier[item];
            if (source >= graph.vertices) {
                continue;
            }
            for (unsigned edge = graph.row_ptr[source];
                 edge < graph.row_ptr[source + 1] && edge < graph.edges;
                 ++edge) {
                const unsigned destination = graph.col_idx[edge];
                if (destination < graph.vertices &&
                    atomicCAS(&level[destination], unvisited, current_level) ==
                        unvisited) {
                    const unsigned slot = atomicAdd(&private_count, 1U);
                    if (slot < private_capacity) {
                        private_frontier[slot] = destination;
                    } else {
                        const unsigned output =
                            atomicAdd(&frontier_sizes[current_level], 1U);
                        if (output < graph.vertices) {
                            current_frontier[output] = destination;
                        }
                    }
                }
            }
        }
        __syncthreads();

        const unsigned staged = min(private_count, private_capacity);
        if (threadIdx.x == 0) {
            public_base =
                atomicAdd(&frontier_sizes[current_level], staged);
        }
        __syncthreads();
        for (unsigned item = threadIdx.x; item < staged; item += blockDim.x) {
            if (public_base + item < graph.vertices) {
                current_frontier[public_base + item] = private_frontier[item];
            }
        }

        grid.sync();
        previous_count = frontier_sizes[current_level];
        unsigned* temporary = previous_frontier;
        previous_frontier = current_frontier;
        current_frontier = temporary;
    }
}

std::vector<unsigned> cooperative_bfs_gpu(
    const pmpp_examples::CsrGraph& host_graph, unsigned root) {
    pmpp_examples::validate_graph(host_graph);
    const unsigned vertices = host_graph.vertices();
    if (vertices == 0 || root >= vertices ||
        host_graph.col_idx.size() >
            static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
        throw std::invalid_argument("cooperative BFS graph/root");
    }
    int device = 0;
    int cooperative = 0;
    int multiprocessors = 0;
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperative,
                                      cudaDevAttrCooperativeLaunch, device));
    if (cooperative == 0) {
        throw std::runtime_error("device does not support cooperative launch");
    }

    device_buffer<unsigned> device_row_ptr(host_graph.row_ptr.size());
    device_buffer<unsigned> device_col_idx(host_graph.col_idx.size());
    device_buffer<unsigned> device_level(vertices);
    device_buffer<unsigned> device_previous(vertices);
    device_buffer<unsigned> device_current(vertices);
    device_buffer<unsigned> device_sizes(static_cast<std::size_t>(vertices) + 1);
    CUDA_CHECK(cudaMemcpy(device_row_ptr.get(), host_graph.row_ptr.data(),
                          host_graph.row_ptr.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
    if (!host_graph.col_idx.empty()) {
        CUDA_CHECK(cudaMemcpy(device_col_idx.get(), host_graph.col_idx.data(),
                              host_graph.col_idx.size() * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(device_level.get(), 0xff,
                          static_cast<std::size_t>(vertices) *
                              sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(device_sizes.get(), 0,
                          (static_cast<std::size_t>(vertices) + 1) *
                              sizeof(unsigned)));
    const unsigned zero = 0;
    const unsigned one = 1;
    CUDA_CHECK(cudaMemcpy(device_level.get() + root, &zero, sizeof(zero),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_previous.get(), &root, sizeof(root),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_sizes.get(), &one, sizeof(one),
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors,
                                      cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor, bfs_one_grid_barrier, threads, 0));
    const int resident_limit = multiprocessors * blocks_per_multiprocessor;
    const int work_blocks =
        static_cast<int>((static_cast<std::size_t>(vertices) + threads - 1) /
                         threads);
    const int blocks = std::min(work_blocks, resident_limit);
    if (blocks <= 0) {
        throw std::runtime_error("cooperative grid cannot be resident");
    }

    const GraphView view{device_row_ptr.get(), device_col_idx.get(), vertices,
                         static_cast<unsigned>(host_graph.col_idx.size())};
    unsigned* level = device_level.get();
    unsigned* previous = device_previous.get();
    unsigned* current = device_current.get();
    unsigned* sizes = device_sizes.get();
    void* arguments[] = {const_cast<GraphView*>(&view), &level, &previous,
                         &current, &sizes};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(bfs_one_grid_barrier), dim3(blocks),
        dim3(threads), arguments, 0, nullptr));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> result(vertices);
    CUDA_CHECK(cudaMemcpy(result.data(), device_level.get(),
                          result.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    return result;
}

const std::vector<std::pair<unsigned, unsigned>>& example_edges() {
    static const std::vector<std::pair<unsigned, unsigned>> edges{
        {0, 2}, {0, 5}, {1, 0}, {1, 4}, {1, 7}, {2, 3}, {3, 0}, {3, 6},
        {4, 3}, {5, 1}, {5, 7}, {6, 4}, {7, 2}, {7, 4}, {7, 6}};
    return edges;
}

void run_cpu_assertions(const pmpp_examples::CsrGraph& graph) {
    const std::vector<unsigned> expected{0, 2, 1, 2, 3, 1, 3, 2};
    pmpp_examples::expect(pmpp_examples::bfs_cpu(graph, 0) == expected,
                          "CPU cooperative BFS reference");
    pmpp_examples::expect(graph.col_idx.size() == 15,
                          "CPU cooperative BFS edge count");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const pmpp_examples::CsrGraph graph =
            pmpp_examples::make_csr_graph(8, example_edges());
        run_cpu_assertions(graph);
        const std::vector<unsigned> reference =
            pmpp_examples::bfs_cpu(graph, 0);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch18_ex03_cooperative_bfs");
            return EXIT_SUCCESS;
        }
        const std::vector<unsigned> actual = cooperative_bfs_gpu(graph, 0);
        pmpp_examples::expect(actual == reference,
                              "GPU cooperative BFS levels");
        std::cout << "ch18_ex03_cooperative_bfs: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch18_ex03_cooperative_bfs: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
