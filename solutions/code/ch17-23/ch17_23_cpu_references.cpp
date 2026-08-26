#include "ch17_23_cpu_algorithms.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

using pmpp_examples::Atom;
using pmpp_examples::ConvolutionShape;
using pmpp_examples::Triplet;

const std::vector<Triplet>& sparse_example() {
    static const std::vector<Triplet> entries{
        {2, 2, 3.0f}, {0, 2, 7.0f}, {3, 3, 1.0f}, {1, 2, 8.0f},
        {0, 0, 1.0f}, {3, 0, 2.0f}, {2, 1, 4.0f}};
    return entries;
}

const std::vector<std::pair<unsigned, unsigned>>& graph_edges() {
    static const std::vector<std::pair<unsigned, unsigned>> edges{
        {0, 2}, {0, 5}, {1, 0}, {1, 4}, {1, 7}, {2, 3}, {3, 0}, {3, 6},
        {4, 3}, {5, 1}, {5, 7}, {6, 4}, {7, 2}, {7, 4}, {7, 6}};
    return edges;
}

void test_sparse_formats() {
    using namespace pmpp_examples;
    const CsrMatrix csr = coo_to_csr_cpu(sparse_example(), 4, 4);
    expect(csr.row_ptr == std::vector<unsigned>({0, 2, 3, 5, 7}),
           "chapter 17 CSR row pointers");
    const std::vector<SparseEntry> expected{
        {0, 0, 1.0f}, {0, 2, 7.0f}, {1, 2, 8.0f}, {2, 1, 4.0f},
        {2, 2, 3.0f}, {3, 0, 2.0f}, {3, 3, 1.0f}};
    expect(canonical_entries(csr) == expected, "chapter 17 COO to CSR");

    constexpr int rows = 4;
    constexpr int nonzeros = 7;
    constexpr int max_row_length = 2;
    expect(3 * nonzeros == 21, "COO storage count");
    expect(2 * nonzeros + rows + 1 == 19, "CSR storage count");
    expect(2 * rows * max_row_length + rows == 20, "ELL storage count");
    expect(2 * nonzeros + rows + max_row_length + 1 == 21,
           "JDS storage count");

    const std::vector<float> x{2.0f, 3.0f, 5.0f, 7.0f};
    const std::vector<float> expected_y{37.0f, 40.0f, 27.0f, 11.0f};
    const HybridMatrix hybrid = make_hybrid(sparse_example(), 4, 4, 1);
    expect_near(hybrid_spmv_cpu(hybrid, x), expected_y, 0.0f, 0.0f,
                "chapter 17 HYB SpMV");
    expect(hybrid.overflow.size() == 3, "HYB overflow count");
    HybridMatrix malformed_hybrid = hybrid;
    malformed_hybrid.overflow[0].row = 4;
    bool rejected_overflow = false;
    try {
        (void)hybrid_spmv_cpu(malformed_hybrid, x);
    } catch (const std::out_of_range&) {
        rejected_overflow = true;
    }
    expect(rejected_overflow, "HYB rejects an invalid overflow coordinate");

    const JdsMatrix jds = make_jds(sparse_example(), 4, 4);
    expect(jds.row_order == std::vector<int>({0, 2, 3, 1}),
           "chapter 17 JDS stable row ordering");
    expect(jds.iter_ptr == std::vector<int>({0, 4, 7}),
           "chapter 17 JDS diagonal pointers");
    expect_near(jds_spmv_cpu(jds, x), expected_y, 0.0f, 0.0f,
                "chapter 17 JDS SpMV");

    const std::array<int, 4> row_lengths{2, 1, 2, 2};
    int padding = 0;
    int overflow = 0;
    for (const int length : row_lengths) {
        padding += std::max(1 - length, 0);
        overflow += std::max(length - 1, 0);
    }
    expect(padding == 0 && overflow == 3, "chapter 17 HYB K cost");
}

void test_graph_algorithms() {
    using namespace pmpp_examples;
    const CsrGraph graph = make_csr_graph(8, graph_edges());
    expect(graph.row_ptr ==
               std::vector<unsigned>({0, 2, 5, 6, 8, 9, 11, 12, 15}),
           "chapter 18 CSR graph pointers");
    expect(graph.col_idx == std::vector<unsigned>(
                                {2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6}),
           "chapter 18 CSR graph destinations");
    const std::vector<unsigned> expected{0, 2, 1, 2, 3, 1, 3, 2};
    expect(bfs_cpu(graph, 0) == expected, "chapter 18 BFS levels");
    const CsrGraph incoming = make_csc_graph(8, graph_edges());
    validate_graph(incoming);
    expect(incoming.col_idx.size() == graph_edges().size(),
           "chapter 18 CSC edge count");

    const std::array<unsigned, 4> push_traversals{1, 2, 3, 2};
    const std::array<unsigned, 4> pull_marks{2, 3, 2, 0};
    expect(std::accumulate(push_traversals.begin(), push_traversals.end(), 0U) ==
               8,
           "chapter 18 push activity");
    expect(std::accumulate(pull_marks.begin(), pull_marks.end(), 0U) == 7,
           "chapter 18 pull activity");
}

void test_subsampling() {
    using namespace pmpp_examples;
    const std::vector<float> input{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    const std::vector<float> output =
        subsample_forward_cpu(input, {0.25f}, 1, 1, 4, 5, 2);
    const std::vector<float> expected{
        1.0f / (1.0f + std::exp(-4.25f)),
        1.0f / (1.0f + std::exp(-6.25f)),
        1.0f / (1.0f + std::exp(-14.25f)),
        1.0f / (1.0f + std::exp(-16.25f))};
    expect_near(output, expected, 1.0e-6f, 1.0e-6f,
                "chapter 19 subsampling");
    expect(output.size() == 4, "subsampling drops incomplete right edge");
}

void test_convolution() {
    using namespace pmpp_examples;
    const ConvolutionShape shape{1, 1, 1, 3, 4, 2};
    std::vector<float> input(convolution_input_count(shape));
    std::iota(input.begin(), input.end(), 1.0f);
    const std::vector<float> filter{1.0f, -1.0f, 0.5f, 2.0f};
    const std::vector<float> upstream(convolution_output_count(shape), 1.0f);
    const std::vector<float> output =
        convolution_forward_cpu(input, filter, shape);
    expect_near(output, {13.5f, 16.0f, 18.5f, 23.5f, 26.0f, 28.5f},
                0.0f, 0.0f, "chapter 19 convolution forward");
    const ConvolutionGradients gradients =
        convolution_backward_cpu(input, filter, upstream, shape);
    expect(gradients.bias == std::vector<float>({6.0f}),
           "chapter 19 bias gradient");

    constexpr float epsilon = 1.0e-3f;
    std::vector<float> plus = filter;
    std::vector<float> minus = filter;
    plus[2] += epsilon;
    minus[2] -= epsilon;
    const auto loss = [&](const std::vector<float>& candidate) {
        const std::vector<float> value =
            convolution_forward_cpu(input, candidate, shape);
        return std::accumulate(value.begin(), value.end(), 0.0f);
    };
    const float finite_difference = (loss(plus) - loss(minus)) / (2 * epsilon);
    expect(std::fabs(finite_difference - gradients.filter[2]) < 0.02f,
           "chapter 19 finite-difference filter gradient");
}

void test_attention() {
    using namespace pmpp_examples;
    constexpr int rows = 5;
    constexpr int features = 4;
    std::vector<float> query(rows * features);
    std::vector<float> key(rows * features);
    std::vector<float> value(rows * features);
    for (std::size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
        key[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
        value[i] = static_cast<float>(i + 1) * 0.05f;
    }
    const float scaling = 1.0f / std::sqrt(static_cast<float>(features));
    const AttentionResult reference =
        attention_reference_cpu(query, key, value, rows, features, scaling);
    const AttentionResult online =
        attention_online_cpu(query, key, value, rows, features, scaling, 3);
    expect_near(online.output, reference.output, 1.0e-6f, 1.0e-5f,
                "chapter 20 online attention");
    expect_near(online.normalizer, reference.normalizer, 1.0e-6f, 1.0e-5f,
                "chapter 20 online normalizer");
    const AttentionResult causal_reference = attention_causal_reference_cpu(
        query, key, value, rows, features, scaling);
    const AttentionResult causal_online = attention_causal_online_cpu(
        query, key, value, rows, features, scaling, 3);
    expect_near(causal_online.output, causal_reference.output, 1.0e-6f,
                1.0e-5f, "chapter 20 causal online attention");
    expect_near(causal_online.normalizer, causal_reference.normalizer,
                1.0e-6f, 1.0e-5f,
                "chapter 20 causal online normalizer");
    for (int feature = 0; feature < features; ++feature) {
        expect(std::fabs(causal_reference.output[feature] - value[feature]) <
                   1.0e-6f,
               "chapter 20 causal row-zero mask");
    }
    expect(std::fabs(causal_reference.output[0] - reference.output[0]) >
               1.0e-4f,
           "chapter 20 causal and unmasked fixtures are distinguishable");

    std::vector<unsigned char> valid(
        static_cast<std::size_t>(rows) * rows, 1U);
    constexpr int all_masked_row = 2;
    constexpr int single_key_row = 3;
    for (int column = 0; column < rows; ++column) {
        valid[static_cast<std::size_t>(all_masked_row) * rows + column] = 0U;
        valid[static_cast<std::size_t>(single_key_row) * rows + column] = 0U;
    }
    constexpr int single_key_column = 1;
    valid[static_cast<std::size_t>(single_key_row) * rows +
          single_key_column] = 1U;
    const AttentionResult masked = attention_masked_online_cpu(
        query, key, value, valid, rows, features, scaling, 2);
    expect(masked.normalizer[all_masked_row] == 0.0f,
           "chapter 20 all-masked normalizer policy");
    for (int feature = 0; feature < features; ++feature) {
        const float all_masked_value =
            masked.output[static_cast<std::size_t>(all_masked_row) *
                              features +
                          feature];
        expect(all_masked_value == 0.0f && std::isfinite(all_masked_value),
               "chapter 20 all-masked finite zero policy");
        expect(std::fabs(
                   masked.output[static_cast<std::size_t>(single_key_row) *
                                     features +
                                 feature] -
                   value[static_cast<std::size_t>(single_key_column) *
                             features +
                         feature]) < 1.0e-6f,
               "chapter 20 first valid item initializes online state");
    }
    for (int row : {0, 1, 4}) {
        expect(std::fabs(masked.normalizer[static_cast<std::size_t>(row)] -
                         online.normalizer[static_cast<std::size_t>(row)]) <
                   1.0e-5f,
               "chapter 20 masked/unmasked normalizer agreement");
    }
    const std::array<float, 32> lanes = [] {
        std::array<float, 32> values{};
        std::iota(values.begin(), values.end(), 1.0f);
        return values;
    }();
    expect(std::accumulate(lanes.begin(), lanes.end(), 0.0f) == 528.0f,
           "chapter 20 warp reduction reference");
}

void test_cenergy_and_partition() {
    using namespace pmpp_examples;
    const std::vector<Atom> atoms{{0.25f, 0.5f, 0.75f, 1.0f},
                                  {2.25f, 1.5f, 0.25f, -0.4f},
                                  {4.5f, 3.25f, 1.5f, 0.7f}};
    const std::vector<float> energy = cenergy_cpu(7, 5, 3, 0.5f, atoms);
    expect(energy.size() == 105, "chapter 21 cenergy element count");
    expect(std::all_of(energy.begin(), energy.end(),
                       [](float value) { return std::isfinite(value); }),
           "chapter 21 cenergy finite values");
    const std::vector<float> singular =
        cenergy_cpu(1, 1, 1, 0.5f, {{0.0f, 0.0f, 0.0f, 1.0f}});
    expect(std::isinf(singular[0]) && singular[0] > 0.0f,
           "chapter 21 q/r singularity is not silently softened");
    const PartitionCounts counts = partition_counts(64, 512, 16);
    expect(counts.output_points == 1984 && counts.halo_points == 128 &&
               counts.stage1_boundary_points == 124 &&
               counts.stage2_internal_points == 1860 &&
               counts.sent_bytes == 512,
           "chapter 23 partition counts");
    expect(1000 * static_cast<int>(sizeof(float)) == 4000,
           "chapter 23 MPI_FLOAT byte count model");
}

}  // namespace

int main() {
    try {
        test_sparse_formats();
        test_graph_algorithms();
        test_subsampling();
        test_convolution();
        test_attention();
        test_cenergy_and_partition();
        std::cout << "ch17_23_cpu_references: all CPU assertions passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "ch17_23_cpu_references: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
