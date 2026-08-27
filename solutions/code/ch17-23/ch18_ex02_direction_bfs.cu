#include "common/cuda_check.hpp"
#include "ch17_23_cpu_algorithms.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr unsigned unvisited = std::numeric_limits<unsigned>::max();

struct GraphView {
    const unsigned* row_ptr;
    const unsigned* col_idx;
    unsigned vertices;
};

__global__ void bfs_push_cas(GraphView graph, unsigned* level,
                             unsigned* new_count, unsigned current) {
    const unsigned vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= graph.vertices || level[vertex] != current - 1) {
        return;
    }
    const unsigned first = graph.row_ptr[vertex];
    const unsigned last = graph.row_ptr[vertex + 1];
    for (unsigned edge = first; edge < last; ++edge) {
        const unsigned destination = graph.col_idx[edge];
        if (destination < graph.vertices &&
            atomicCAS(&level[destination], unvisited, current) == unvisited) {
            atomicAdd(new_count, 1U);
        }
    }
}

__global__ void bfs_pull(GraphView incoming, unsigned* level,
                         unsigned* new_count, unsigned current) {
    const unsigned vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= incoming.vertices || level[vertex] != unvisited) {
        return;
    }
    const unsigned first = incoming.row_ptr[vertex];
    const unsigned last = incoming.row_ptr[vertex + 1];
    for (unsigned edge = first; edge < last; ++edge) {
        const unsigned source = incoming.col_idx[edge];
        if (source < incoming.vertices && level[source] == current - 1) {
            level[vertex] = current;
            atomicAdd(new_count, 1U);
            break;
        }
    }
}

struct DeviceGraph {
    explicit DeviceGraph(const pmpp_examples::CsrGraph& graph)
        : row_ptr(graph.row_ptr.size()), col_idx(graph.col_idx.size()) {
        CUDA_CHECK(cudaMemcpy(row_ptr.get(), graph.row_ptr.data(),
                              graph.row_ptr.size() * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
        if (!graph.col_idx.empty()) {
            CUDA_CHECK(cudaMemcpy(col_idx.get(), graph.col_idx.data(),
                                  graph.col_idx.size() * sizeof(unsigned),
                                  cudaMemcpyHostToDevice));
        }
        view = {row_ptr.get(), col_idx.get(), graph.vertices()};
    }

    device_buffer<unsigned> row_ptr;
    device_buffer<unsigned> col_idx;
    GraphView view{};
};

std::vector<unsigned> direction_bfs_gpu(
    const pmpp_examples::CsrGraph& outgoing,
    const pmpp_examples::CsrGraph& incoming, unsigned root) {
    pmpp_examples::validate_graph(outgoing);
    pmpp_examples::validate_graph(incoming);
    if (outgoing.vertices() == 0 || outgoing.vertices() != incoming.vertices() ||
        root >= outgoing.vertices()) {
        throw std::invalid_argument("direction BFS graph/root");
    }
    const unsigned vertices = outgoing.vertices();
    DeviceGraph device_outgoing(outgoing);
    DeviceGraph device_incoming(incoming);
    device_buffer<unsigned> device_level(vertices);
    device_buffer<unsigned> device_new_count(1);
    CUDA_CHECK(cudaMemset(device_level.get(), 0xff,
                          static_cast<std::size_t>(vertices) *
                              sizeof(unsigned)));
    const unsigned zero = 0;
    CUDA_CHECK(cudaMemcpy(device_level.get() + root, &zero, sizeof(zero),
                          cudaMemcpyHostToDevice));

    constexpr unsigned threads = 256;
    const unsigned blocks = (vertices + threads - 1) / threads;
    unsigned frontier = 1;
    unsigned visited = 1;
    unsigned current = 1;
    bool pull = false;
    const double average_degree =
        static_cast<double>(outgoing.col_idx.size()) / vertices;
    constexpr double alpha = 3.0;
    constexpr double beta = 24.0;
    bool used_push = false;
    bool used_pull = false;

    while (frontier != 0) {
        const unsigned remaining = vertices - visited;
        if (!pull && static_cast<double>(frontier) * average_degree >
                         static_cast<double>(remaining) / alpha) {
            pull = true;
        } else if (pull &&
                   static_cast<double>(frontier) <
                       static_cast<double>(vertices) / beta) {
            pull = false;
        }
        CUDA_CHECK(cudaMemset(device_new_count.get(), 0, sizeof(unsigned)));
        if (pull) {
            bfs_pull<<<blocks, threads>>>(device_incoming.view,
                                          device_level.get(),
                                          device_new_count.get(), current);
            used_pull = true;
        } else {
            bfs_push_cas<<<blocks, threads>>>(device_outgoing.view,
                                              device_level.get(),
                                              device_new_count.get(), current);
            used_push = true;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&frontier, device_new_count.get(),
                              sizeof(frontier), cudaMemcpyDeviceToHost));
        if (frontier > remaining) {
            throw std::runtime_error("BFS discovered too many vertices");
        }
        visited += frontier;
        ++current;
    }
    pmpp_examples::expect(used_push && used_pull,
                          "direction BFS did not exercise both modes");
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
                          "CPU direction BFS levels");
    bool rejected = false;
    try {
        (void)pmpp_examples::bfs_cpu(graph, 8);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    pmpp_examples::expect(rejected, "CPU BFS root validation");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = pmpp_examples::validate_cli(argc, argv);
        const pmpp_examples::CsrGraph outgoing =
            pmpp_examples::make_csr_graph(8, example_edges());
        const pmpp_examples::CsrGraph incoming =
            pmpp_examples::make_csc_graph(8, example_edges());
        run_cpu_assertions(outgoing);
        const std::vector<unsigned> reference =
            pmpp_examples::bfs_cpu(outgoing, 0);
        if (cpu_only || !has_cuda_device()) {
            report_cpu_only("ch18_ex02_direction_bfs");
            return EXIT_SUCCESS;
        }
        const std::vector<unsigned> actual =
            direction_bfs_gpu(outgoing, incoming, 0);
        pmpp_examples::expect(actual == reference,
                              "GPU direction BFS levels");
        std::cout << "ch18_ex02_direction_bfs: GPU comparison passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch18_ex02_direction_bfs: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
