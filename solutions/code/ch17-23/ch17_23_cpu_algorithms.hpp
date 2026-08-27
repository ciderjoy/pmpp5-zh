#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pmpp_examples {

inline void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline bool validate_cli(int argc, char** argv) {
    bool cpu_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument{argv[i]};
        if (argument != "--cpu-only") {
            throw std::invalid_argument("unknown argument: " + argument);
        }
        if (cpu_only) {
            throw std::invalid_argument("--cpu-only specified more than once");
        }
        cpu_only = true;
    }
    return cpu_only;
}

inline std::size_t checked_product(
    std::initializer_list<std::size_t> factors) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (factor != 0 && result > std::numeric_limits<std::size_t>::max() /
                                      factor) {
            throw std::overflow_error("element count overflow");
        }
        result *= factor;
    }
    return result;
}

inline void expect_near(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        float absolute_tolerance, float relative_tolerance,
                        const std::string& label) {
    expect(actual.size() == expected.size(), label + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::fabs(actual[i] - expected[i]);
        const float scale = std::max(std::fabs(actual[i]),
                                     std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) ||
            difference > absolute_tolerance + relative_tolerance * scale) {
            throw std::runtime_error(label + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

struct Triplet {
    int row = 0;
    int col = 0;
    float value = 0.0f;
};

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<unsigned> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

inline void validate_triplets(const std::vector<Triplet>& entries, int rows,
                              int cols) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("negative sparse-matrix shape");
    }
    for (const Triplet& entry : entries) {
        if (entry.row < 0 || entry.row >= rows || entry.col < 0 ||
            entry.col >= cols || !std::isfinite(entry.value)) {
            throw std::out_of_range("invalid COO entry");
        }
    }
}

inline CsrMatrix coo_to_csr_cpu(const std::vector<Triplet>& entries, int rows,
                                int cols) {
    validate_triplets(entries, rows, cols);
    CsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    result.col_idx.resize(entries.size());
    result.values.resize(entries.size());
    for (const Triplet& entry : entries) {
        ++result.row_ptr[static_cast<std::size_t>(entry.row) + 1];
    }
    std::partial_sum(result.row_ptr.begin(), result.row_ptr.end(),
                     result.row_ptr.begin());
    std::vector<unsigned> next(result.row_ptr.begin(), result.row_ptr.end() - 1);
    for (const Triplet& entry : entries) {
        const unsigned position = next[static_cast<std::size_t>(entry.row)]++;
        result.col_idx[position] = entry.col;
        result.values[position] = entry.value;
    }
    return result;
}

using SparseEntry = std::tuple<int, int, float>;

inline std::vector<SparseEntry> canonical_entries(const CsrMatrix& matrix) {
    if (matrix.rows < 0 || matrix.cols < 0 ||
        matrix.row_ptr.size() != static_cast<std::size_t>(matrix.rows) + 1 ||
        matrix.col_idx.size() != matrix.values.size() ||
        matrix.row_ptr.back() != matrix.values.size()) {
        throw std::invalid_argument("malformed CSR matrix");
    }
    std::vector<SparseEntry> entries;
    entries.reserve(matrix.values.size());
    for (int row = 0; row < matrix.rows; ++row) {
        const unsigned first = matrix.row_ptr[static_cast<std::size_t>(row)];
        const unsigned last = matrix.row_ptr[static_cast<std::size_t>(row) + 1];
        if (first > last || last > matrix.values.size()) {
            throw std::invalid_argument("malformed CSR row pointer");
        }
        for (unsigned position = first; position < last; ++position) {
            const int col = matrix.col_idx[position];
            if (col < 0 || col >= matrix.cols) {
                throw std::out_of_range("invalid CSR column");
            }
            entries.emplace_back(row, col, matrix.values[position]);
        }
    }
    std::sort(entries.begin(), entries.end());
    return entries;
}

inline std::vector<float> coo_spmv_cpu(const std::vector<Triplet>& entries,
                                       int rows, int cols,
                                       const std::vector<float>& x) {
    validate_triplets(entries, rows, cols);
    if (x.size() != static_cast<std::size_t>(cols)) {
        throw std::invalid_argument("SpMV vector length mismatch");
    }
    std::vector<float> y(static_cast<std::size_t>(rows), 0.0f);
    for (const Triplet& entry : entries) {
        y[static_cast<std::size_t>(entry.row)] +=
            entry.value * x[static_cast<std::size_t>(entry.col)];
    }
    return y;
}

struct HybridMatrix {
    int rows = 0;
    int cols = 0;
    int width = 0;
    std::vector<float> ell_values;
    std::vector<int> ell_cols;
    std::vector<int> row_lengths;
    std::vector<Triplet> overflow;
};

inline HybridMatrix make_hybrid(const std::vector<Triplet>& entries, int rows,
                                int cols, int width) {
    validate_triplets(entries, rows, cols);
    if (width < 0) {
        throw std::invalid_argument("negative ELL width");
    }
    HybridMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.width = width;
    const std::size_t slots = checked_product(
        {static_cast<std::size_t>(rows), static_cast<std::size_t>(width)});
    result.ell_values.assign(slots, 0.0f);
    result.ell_cols.assign(slots, 0);
    result.row_lengths.assign(static_cast<std::size_t>(rows), 0);
    std::vector<std::vector<Triplet>> by_row(static_cast<std::size_t>(rows));
    for (const Triplet& entry : entries) {
        by_row[static_cast<std::size_t>(entry.row)].push_back(entry);
    }
    for (int row = 0; row < rows; ++row) {
        const auto& row_entries = by_row[static_cast<std::size_t>(row)];
        const int stored = std::min<int>(width, row_entries.size());
        result.row_lengths[static_cast<std::size_t>(row)] = stored;
        for (int item = 0; item < stored; ++item) {
            const std::size_t position = static_cast<std::size_t>(item) * rows +
                                         static_cast<std::size_t>(row);
            result.ell_values[position] =
                row_entries[static_cast<std::size_t>(item)].value;
            result.ell_cols[position] =
                row_entries[static_cast<std::size_t>(item)].col;
        }
        result.overflow.insert(result.overflow.end(),
                               row_entries.begin() + stored, row_entries.end());
    }
    return result;
}

inline std::vector<float> hybrid_spmv_cpu(const HybridMatrix& matrix,
                                          const std::vector<float>& x) {
    if (matrix.rows < 0 || matrix.cols < 0 || matrix.width < 0 ||
        x.size() != static_cast<std::size_t>(matrix.cols) ||
        matrix.row_lengths.size() != static_cast<std::size_t>(matrix.rows)) {
        throw std::invalid_argument("malformed HYB matrix");
    }
    const std::size_t slots = checked_product(
        {static_cast<std::size_t>(matrix.rows),
         static_cast<std::size_t>(matrix.width)});
    if (matrix.ell_values.size() != slots || matrix.ell_cols.size() != slots) {
        throw std::invalid_argument("malformed ELL storage");
    }
    std::vector<float> y(static_cast<std::size_t>(matrix.rows), 0.0f);
    for (int row = 0; row < matrix.rows; ++row) {
        const int length = matrix.row_lengths[static_cast<std::size_t>(row)];
        if (length < 0 || length > matrix.width) {
            throw std::invalid_argument("invalid ELL row length");
        }
        for (int item = 0; item < length; ++item) {
            const std::size_t position = static_cast<std::size_t>(item) *
                                             matrix.rows +
                                         static_cast<std::size_t>(row);
            const int col = matrix.ell_cols[position];
            if (col < 0 || col >= matrix.cols) {
                throw std::out_of_range("invalid ELL column");
            }
            y[static_cast<std::size_t>(row)] +=
                matrix.ell_values[position] * x[static_cast<std::size_t>(col)];
        }
    }
    for (const Triplet& entry : matrix.overflow) {
        if (entry.row < 0 || entry.row >= matrix.rows || entry.col < 0 ||
            entry.col >= matrix.cols || !std::isfinite(entry.value)) {
            throw std::out_of_range("invalid HYB overflow entry");
        }
        y[static_cast<std::size_t>(entry.row)] +=
            entry.value * x[static_cast<std::size_t>(entry.col)];
    }
    return y;
}

struct JdsMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;
    std::vector<int> col_idx;
    std::vector<int> row_order;
    std::vector<int> iter_ptr;
};

inline JdsMatrix make_jds(const std::vector<Triplet>& entries, int rows,
                          int cols) {
    validate_triplets(entries, rows, cols);
    std::vector<std::vector<Triplet>> by_row(static_cast<std::size_t>(rows));
    for (const Triplet& entry : entries) {
        by_row[static_cast<std::size_t>(entry.row)].push_back(entry);
    }
    JdsMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.row_order.resize(static_cast<std::size_t>(rows));
    std::iota(result.row_order.begin(), result.row_order.end(), 0);
    std::stable_sort(result.row_order.begin(), result.row_order.end(),
                     [&](int left, int right) {
                         return by_row[static_cast<std::size_t>(left)].size() >
                                by_row[static_cast<std::size_t>(right)].size();
                     });
    std::size_t max_length = 0;
    for (const auto& row : by_row) {
        max_length = std::max(max_length, row.size());
    }
    result.iter_ptr.push_back(0);
    for (std::size_t item = 0; item < max_length; ++item) {
        for (const int row : result.row_order) {
            const auto& row_entries = by_row[static_cast<std::size_t>(row)];
            if (item >= row_entries.size()) {
                break;
            }
            result.col_idx.push_back(row_entries[item].col);
            result.values.push_back(row_entries[item].value);
        }
        result.iter_ptr.push_back(static_cast<int>(result.values.size()));
    }
    return result;
}

inline void validate_jds(const JdsMatrix& matrix) {
    if (matrix.rows < 0 || matrix.cols < 0 ||
        matrix.values.size() != matrix.col_idx.size() ||
        matrix.row_order.size() != static_cast<std::size_t>(matrix.rows) ||
        matrix.iter_ptr.empty() || matrix.iter_ptr.front() != 0 ||
        matrix.iter_ptr.back() != static_cast<int>(matrix.values.size())) {
        throw std::invalid_argument("malformed JDS matrix");
    }
    std::vector<int> order = matrix.row_order;
    std::sort(order.begin(), order.end());
    for (int row = 0; row < matrix.rows; ++row) {
        if (order[static_cast<std::size_t>(row)] != row) {
            throw std::invalid_argument("JDS row order is not a permutation");
        }
    }
    int previous_active = matrix.rows;
    for (std::size_t item = 0; item + 1 < matrix.iter_ptr.size(); ++item) {
        const int first = matrix.iter_ptr[item];
        const int last = matrix.iter_ptr[item + 1];
        const int active = last - first;
        if (first < 0 || last < first || active > previous_active) {
            throw std::invalid_argument("invalid JDS diagonal pointers");
        }
        previous_active = active;
    }
    for (const int col : matrix.col_idx) {
        if (col < 0 || col >= matrix.cols) {
            throw std::out_of_range("invalid JDS column");
        }
    }
}

inline std::vector<float> jds_spmv_cpu(const JdsMatrix& matrix,
                                       const std::vector<float>& x) {
    validate_jds(matrix);
    if (x.size() != static_cast<std::size_t>(matrix.cols)) {
        throw std::invalid_argument("SpMV vector length mismatch");
    }
    std::vector<float> y(static_cast<std::size_t>(matrix.rows), 0.0f);
    const int iterations = static_cast<int>(matrix.iter_ptr.size()) - 1;
    for (int sorted_row = 0; sorted_row < matrix.rows; ++sorted_row) {
        float sum = 0.0f;
        for (int item = 0; item < iterations; ++item) {
            const int active = matrix.iter_ptr[static_cast<std::size_t>(item) + 1] -
                               matrix.iter_ptr[static_cast<std::size_t>(item)];
            if (sorted_row >= active) {
                break;
            }
            const int position =
                matrix.iter_ptr[static_cast<std::size_t>(item)] + sorted_row;
            sum += matrix.values[static_cast<std::size_t>(position)] *
                   x[static_cast<std::size_t>(
                       matrix.col_idx[static_cast<std::size_t>(position)])];
        }
        y[static_cast<std::size_t>(
            matrix.row_order[static_cast<std::size_t>(sorted_row)])] = sum;
    }
    return y;
}

struct CsrGraph {
    std::vector<unsigned> row_ptr;
    std::vector<unsigned> col_idx;

    unsigned vertices() const {
        return row_ptr.empty() ? 0U
                               : static_cast<unsigned>(row_ptr.size() - 1);
    }
};

inline CsrGraph make_csr_graph(
    unsigned vertices,
    const std::vector<std::pair<unsigned, unsigned>>& edges) {
    CsrGraph graph;
    graph.row_ptr.assign(static_cast<std::size_t>(vertices) + 1, 0);
    for (const auto [source, destination] : edges) {
        if (source >= vertices || destination >= vertices) {
            throw std::out_of_range("graph edge endpoint");
        }
        ++graph.row_ptr[static_cast<std::size_t>(source) + 1];
    }
    std::partial_sum(graph.row_ptr.begin(), graph.row_ptr.end(),
                     graph.row_ptr.begin());
    graph.col_idx.resize(edges.size());
    std::vector<unsigned> next(graph.row_ptr.begin(), graph.row_ptr.end() - 1);
    for (const auto [source, destination] : edges) {
        graph.col_idx[next[source]++] = destination;
    }
    for (unsigned source = 0; source < vertices; ++source) {
        std::sort(graph.col_idx.begin() + graph.row_ptr[source],
                  graph.col_idx.begin() + graph.row_ptr[source + 1]);
    }
    return graph;
}

inline CsrGraph make_csc_graph(
    unsigned vertices,
    const std::vector<std::pair<unsigned, unsigned>>& edges) {
    std::vector<std::pair<unsigned, unsigned>> reversed;
    reversed.reserve(edges.size());
    for (const auto [source, destination] : edges) {
        reversed.emplace_back(destination, source);
    }
    return make_csr_graph(vertices, reversed);
}

inline void validate_graph(const CsrGraph& graph) {
    if (graph.row_ptr.empty() || graph.row_ptr.front() != 0 ||
        graph.row_ptr.back() != graph.col_idx.size()) {
        throw std::invalid_argument("malformed CSR graph");
    }
    const unsigned vertices = graph.vertices();
    for (unsigned source = 0; source < vertices; ++source) {
        if (graph.row_ptr[source] > graph.row_ptr[source + 1]) {
            throw std::invalid_argument("nonmonotonic CSR graph pointers");
        }
    }
    for (const unsigned destination : graph.col_idx) {
        if (destination >= vertices) {
            throw std::out_of_range("CSR graph endpoint");
        }
    }
}

inline std::vector<unsigned> bfs_cpu(const CsrGraph& graph, unsigned root) {
    validate_graph(graph);
    const unsigned vertices = graph.vertices();
    if (vertices == 0 || root >= vertices) {
        throw std::invalid_argument("BFS root");
    }
    constexpr unsigned unvisited = std::numeric_limits<unsigned>::max();
    std::vector<unsigned> level(vertices, unvisited);
    std::queue<unsigned> frontier;
    level[root] = 0;
    frontier.push(root);
    while (!frontier.empty()) {
        const unsigned source = frontier.front();
        frontier.pop();
        for (unsigned edge = graph.row_ptr[source];
             edge < graph.row_ptr[source + 1]; ++edge) {
            const unsigned destination = graph.col_idx[edge];
            if (level[destination] == unvisited) {
                level[destination] = level[source] + 1;
                frontier.push(destination);
            }
        }
    }
    return level;
}

inline std::vector<float> subsample_forward_cpu(
    const std::vector<float>& input, const std::vector<float>& bias, int batch,
    int channels, int height, int width, int window) {
    if (batch < 0 || channels < 0 || height < 0 || width < 0 || window <= 0) {
        throw std::invalid_argument("subsampling shape");
    }
    const std::size_t input_count = checked_product(
        {static_cast<std::size_t>(batch), static_cast<std::size_t>(channels),
         static_cast<std::size_t>(height), static_cast<std::size_t>(width)});
    if (input.size() != input_count ||
        bias.size() != static_cast<std::size_t>(channels)) {
        throw std::invalid_argument("subsampling input size");
    }
    const int output_height = height / window;
    const int output_width = width / window;
    std::vector<float> output(checked_product(
        {static_cast<std::size_t>(batch), static_cast<std::size_t>(channels),
         static_cast<std::size_t>(output_height),
         static_cast<std::size_t>(output_width)}));
    for (int n = 0; n < batch; ++n) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int row = 0; row < output_height; ++row) {
                for (int col = 0; col < output_width; ++col) {
                    float sum = 0.0f;
                    for (int p = 0; p < window; ++p) {
                        for (int q = 0; q < window; ++q) {
                            const std::size_t index =
                                ((static_cast<std::size_t>(n) * channels +
                                  channel) *
                                     height +
                                 row * window + p) *
                                    width +
                                col * window + q;
                            sum += input[index];
                        }
                    }
                    const float activation =
                        sum / static_cast<float>(window * window) +
                        bias[static_cast<std::size_t>(channel)];
                    const std::size_t output_index =
                        ((static_cast<std::size_t>(n) * channels + channel) *
                             output_height +
                         row) *
                            output_width +
                        col;
                    output[output_index] =
                        1.0f / (1.0f + std::exp(-activation));
                }
            }
        }
    }
    return output;
}

struct ConvolutionShape {
    int batch = 0;
    int output_channels = 0;
    int input_channels = 0;
    int height = 0;
    int width = 0;
    int kernel = 0;

    int output_height() const { return height - kernel + 1; }
    int output_width() const { return width - kernel + 1; }
};

inline void validate_convolution_shape(const ConvolutionShape& shape) {
    if (shape.batch <= 0 || shape.output_channels <= 0 ||
        shape.input_channels <= 0 || shape.height <= 0 || shape.width <= 0 ||
        shape.kernel <= 0 || shape.kernel > shape.height ||
        shape.kernel > shape.width) {
        throw std::invalid_argument("convolution shape");
    }
}

inline std::size_t convolution_input_count(const ConvolutionShape& shape) {
    validate_convolution_shape(shape);
    return checked_product({static_cast<std::size_t>(shape.batch),
                            static_cast<std::size_t>(shape.input_channels),
                            static_cast<std::size_t>(shape.height),
                            static_cast<std::size_t>(shape.width)});
}

inline std::size_t convolution_filter_count(const ConvolutionShape& shape) {
    validate_convolution_shape(shape);
    return checked_product({static_cast<std::size_t>(shape.output_channels),
                            static_cast<std::size_t>(shape.input_channels),
                            static_cast<std::size_t>(shape.kernel),
                            static_cast<std::size_t>(shape.kernel)});
}

inline std::size_t convolution_output_count(const ConvolutionShape& shape) {
    validate_convolution_shape(shape);
    return checked_product({static_cast<std::size_t>(shape.batch),
                            static_cast<std::size_t>(shape.output_channels),
                            static_cast<std::size_t>(shape.output_height()),
                            static_cast<std::size_t>(shape.output_width())});
}

inline std::vector<float> convolution_forward_cpu(
    const std::vector<float>& input, const std::vector<float>& filter,
    const ConvolutionShape& shape) {
    if (input.size() != convolution_input_count(shape) ||
        filter.size() != convolution_filter_count(shape)) {
        throw std::invalid_argument("convolution input size");
    }
    const int output_height = shape.output_height();
    const int output_width = shape.output_width();
    std::vector<float> output(convolution_output_count(shape), 0.0f);
    for (int n = 0; n < shape.batch; ++n) {
        for (int m = 0; m < shape.output_channels; ++m) {
            for (int row = 0; row < output_height; ++row) {
                for (int col = 0; col < output_width; ++col) {
                    float sum = 0.0f;
                    for (int channel = 0; channel < shape.input_channels;
                         ++channel) {
                        for (int p = 0; p < shape.kernel; ++p) {
                            for (int q = 0; q < shape.kernel; ++q) {
                                const std::size_t input_index =
                                    ((static_cast<std::size_t>(n) *
                                          shape.input_channels +
                                      channel) *
                                         shape.height +
                                     row + p) *
                                        shape.width +
                                    col + q;
                                const std::size_t filter_index =
                                    ((static_cast<std::size_t>(m) *
                                          shape.input_channels +
                                      channel) *
                                         shape.kernel +
                                     p) *
                                        shape.kernel +
                                    q;
                                sum += input[input_index] * filter[filter_index];
                            }
                        }
                    }
                    const std::size_t output_index =
                        ((static_cast<std::size_t>(n) *
                              shape.output_channels +
                          m) *
                             output_height +
                         row) *
                            output_width +
                        col;
                    output[output_index] = sum;
                }
            }
        }
    }
    return output;
}

struct ConvolutionGradients {
    std::vector<float> input;
    std::vector<float> filter;
    std::vector<float> bias;
};

inline ConvolutionGradients convolution_backward_cpu(
    const std::vector<float>& input, const std::vector<float>& filter,
    const std::vector<float>& upstream, const ConvolutionShape& shape) {
    if (input.size() != convolution_input_count(shape) ||
        filter.size() != convolution_filter_count(shape) ||
        upstream.size() != convolution_output_count(shape)) {
        throw std::invalid_argument("convolution gradient input size");
    }
    const int output_height = shape.output_height();
    const int output_width = shape.output_width();
    ConvolutionGradients gradients;
    gradients.input.assign(input.size(), 0.0f);
    gradients.filter.assign(filter.size(), 0.0f);
    gradients.bias.assign(static_cast<std::size_t>(shape.output_channels),
                          0.0f);
    for (int n = 0; n < shape.batch; ++n) {
        for (int m = 0; m < shape.output_channels; ++m) {
            for (int row = 0; row < output_height; ++row) {
                for (int col = 0; col < output_width; ++col) {
                    const std::size_t output_index =
                        ((static_cast<std::size_t>(n) *
                              shape.output_channels +
                          m) *
                             output_height +
                         row) *
                            output_width +
                        col;
                    const float gradient = upstream[output_index];
                    gradients.bias[static_cast<std::size_t>(m)] += gradient;
                    for (int channel = 0; channel < shape.input_channels;
                         ++channel) {
                        for (int p = 0; p < shape.kernel; ++p) {
                            for (int q = 0; q < shape.kernel; ++q) {
                                const std::size_t input_index =
                                    ((static_cast<std::size_t>(n) *
                                          shape.input_channels +
                                      channel) *
                                         shape.height +
                                     row + p) *
                                        shape.width +
                                    col + q;
                                const std::size_t filter_index =
                                    ((static_cast<std::size_t>(m) *
                                          shape.input_channels +
                                      channel) *
                                         shape.kernel +
                                     p) *
                                        shape.kernel +
                                    q;
                                gradients.input[input_index] +=
                                    gradient * filter[filter_index];
                                gradients.filter[filter_index] +=
                                    gradient * input[input_index];
                            }
                        }
                    }
                }
            }
        }
    }
    return gradients;
}

struct AttentionResult {
    std::vector<float> output;
    std::vector<float> normalizer;
};

inline void validate_attention(const std::vector<float>& query,
                               const std::vector<float>& key,
                               const std::vector<float>& value, int rows,
                               int features, float scaling) {
    if (rows <= 0 || features <= 0 || !std::isfinite(scaling) ||
        scaling <= 0.0f) {
        throw std::invalid_argument("attention shape or scaling");
    }
    const std::size_t elements = checked_product(
        {static_cast<std::size_t>(rows), static_cast<std::size_t>(features)});
    if (query.size() != elements || key.size() != elements ||
        value.size() != elements) {
        throw std::invalid_argument("attention input size");
    }
}

inline AttentionResult attention_reference_cpu_impl(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling,
    bool causal) {
    validate_attention(query, key, value, rows, features, scaling);
    AttentionResult result;
    result.output.assign(query.size(), 0.0f);
    result.normalizer.resize(static_cast<std::size_t>(rows));
    std::vector<float> scores(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const int columns = causal ? row + 1 : rows;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int column = 0; column < columns; ++column) {
            float score = 0.0f;
            for (int feature = 0; feature < features; ++feature) {
                score += query[static_cast<std::size_t>(row) * features +
                               feature] *
                         key[static_cast<std::size_t>(column) * features +
                             feature];
            }
            scores[static_cast<std::size_t>(column)] = score * scaling;
            maximum = std::max(maximum, score * scaling);
        }
        float denominator = 0.0f;
        for (int column = 0; column < columns; ++column) {
            const float weight =
                std::exp(scores[static_cast<std::size_t>(column)] - maximum);
            denominator += weight;
            for (int feature = 0; feature < features; ++feature) {
                result.output[static_cast<std::size_t>(row) * features +
                              feature] +=
                    weight * value[static_cast<std::size_t>(column) * features +
                                   feature];
            }
        }
        result.normalizer[static_cast<std::size_t>(row)] = denominator;
        for (int feature = 0; feature < features; ++feature) {
            result.output[static_cast<std::size_t>(row) * features + feature] /=
                denominator;
        }
    }
    return result;
}

inline AttentionResult attention_reference_cpu(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling) {
    return attention_reference_cpu_impl(query, key, value, rows, features,
                                        scaling, false);
}

inline AttentionResult attention_causal_reference_cpu(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling) {
    return attention_reference_cpu_impl(query, key, value, rows, features,
                                        scaling, true);
}

inline AttentionResult attention_online_cpu_impl(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling,
    int tile_size, bool causal) {
    validate_attention(query, key, value, rows, features, scaling);
    if (tile_size <= 0) {
        throw std::invalid_argument("attention tile size");
    }
    AttentionResult result;
    result.output.assign(query.size(), 0.0f);
    result.normalizer.resize(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const int columns = causal ? row + 1 : rows;
        float maximum = -std::numeric_limits<float>::infinity();
        float denominator = 0.0f;
        std::vector<float> accumulator(static_cast<std::size_t>(features),
                                       0.0f);
        for (int tile = 0; tile < columns; tile += tile_size) {
            const int tile_end = std::min(columns, tile + tile_size);
            for (int column = tile; column < tile_end; ++column) {
                float score = 0.0f;
                for (int feature = 0; feature < features; ++feature) {
                    score += query[static_cast<std::size_t>(row) * features +
                                   feature] *
                             key[static_cast<std::size_t>(column) * features +
                                 feature];
                }
                score *= scaling;
                const float new_maximum = std::max(maximum, score);
                const float previous_scale = std::exp(maximum - new_maximum);
                const float weight = std::exp(score - new_maximum);
                denominator = denominator * previous_scale + weight;
                for (int feature = 0; feature < features; ++feature) {
                    accumulator[static_cast<std::size_t>(feature)] =
                        accumulator[static_cast<std::size_t>(feature)] *
                            previous_scale +
                        weight * value[static_cast<std::size_t>(column) *
                                           features +
                                       feature];
                }
                maximum = new_maximum;
            }
        }
        result.normalizer[static_cast<std::size_t>(row)] = denominator;
        for (int feature = 0; feature < features; ++feature) {
            result.output[static_cast<std::size_t>(row) * features + feature] =
                accumulator[static_cast<std::size_t>(feature)] / denominator;
        }
    }
    return result;
}

inline AttentionResult attention_online_cpu(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling,
    int tile_size) {
    return attention_online_cpu_impl(query, key, value, rows, features,
                                     scaling, tile_size, false);
}

inline AttentionResult attention_causal_online_cpu(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, int rows, int features, float scaling,
    int tile_size) {
    return attention_online_cpu_impl(query, key, value, rows, features,
                                     scaling, tile_size, true);
}

inline AttentionResult attention_masked_online_cpu(
    const std::vector<float>& query, const std::vector<float>& key,
    const std::vector<float>& value, const std::vector<unsigned char>& valid,
    int rows, int features, float scaling, int tile_size) {
    validate_attention(query, key, value, rows, features, scaling);
    if (tile_size <= 0) {
        throw std::invalid_argument("attention tile size");
    }
    const std::size_t mask_elements = checked_product(
        {static_cast<std::size_t>(rows), static_cast<std::size_t>(rows)});
    if (valid.size() != mask_elements) {
        throw std::invalid_argument("attention mask size");
    }

    AttentionResult result;
    result.output.assign(query.size(), 0.0f);
    result.normalizer.assign(static_cast<std::size_t>(rows), 0.0f);
    for (int row = 0; row < rows; ++row) {
        bool has_valid = false;
        float maximum = -std::numeric_limits<float>::infinity();
        float denominator = 0.0f;
        std::vector<float> accumulator(static_cast<std::size_t>(features),
                                       0.0f);
        for (int tile = 0; tile < rows; tile += tile_size) {
            const int tile_end = std::min(rows, tile + tile_size);
            for (int column = tile; column < tile_end; ++column) {
                const std::size_t mask_index =
                    static_cast<std::size_t>(row) * rows + column;
                if (valid[mask_index] == 0U) {
                    continue;
                }
                float score = 0.0f;
                for (int feature = 0; feature < features; ++feature) {
                    score += query[static_cast<std::size_t>(row) * features +
                                   feature] *
                             key[static_cast<std::size_t>(column) * features +
                                 feature];
                }
                score *= scaling;
                if (!has_valid) {
                    maximum = score;
                    denominator = 1.0f;
                    for (int feature = 0; feature < features; ++feature) {
                        accumulator[static_cast<std::size_t>(feature)] =
                            value[static_cast<std::size_t>(column) * features +
                                  feature];
                    }
                    has_valid = true;
                    continue;
                }
                const float new_maximum = std::max(maximum, score);
                const float previous_scale =
                    std::exp(maximum - new_maximum);
                const float weight = std::exp(score - new_maximum);
                denominator = denominator * previous_scale + weight;
                for (int feature = 0; feature < features; ++feature) {
                    accumulator[static_cast<std::size_t>(feature)] =
                        accumulator[static_cast<std::size_t>(feature)] *
                            previous_scale +
                        weight * value[static_cast<std::size_t>(column) *
                                           features +
                                       feature];
                }
                maximum = new_maximum;
            }
        }
        if (!has_valid) {
            continue;
        }
        result.normalizer[static_cast<std::size_t>(row)] = denominator;
        for (int feature = 0; feature < features; ++feature) {
            result.output[static_cast<std::size_t>(row) * features + feature] =
                accumulator[static_cast<std::size_t>(feature)] / denominator;
        }
    }
    return result;
}

struct Atom {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float charge = 0.0f;
};

inline std::vector<float> cenergy_cpu(int size_x, int size_y, int size_z,
                                      float spacing,
                                      const std::vector<Atom>& atoms) {
    if (size_x <= 0 || size_y <= 0 || size_z <= 0 ||
        !std::isfinite(spacing) || spacing <= 0.0f) {
        throw std::invalid_argument("cenergy grid");
    }
    for (const Atom& atom : atoms) {
        if (!std::isfinite(atom.x) || !std::isfinite(atom.y) ||
            !std::isfinite(atom.z) || !std::isfinite(atom.charge)) {
            throw std::invalid_argument("non-finite atom");
        }
    }
    std::vector<float> energy(
        checked_product({static_cast<std::size_t>(size_x),
                         static_cast<std::size_t>(size_y),
                         static_cast<std::size_t>(size_z)}),
        0.0f);
    for (int z = 0; z < size_z; ++z) {
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                float sum = 0.0f;
                for (const Atom& atom : atoms) {
                    const float dx = static_cast<float>(x) * spacing - atom.x;
                    const float dy = static_cast<float>(y) * spacing - atom.y;
                    const float dz = static_cast<float>(z) * spacing - atom.z;
                    sum += atom.charge /
                           std::sqrt(dx * dx + dy * dy + dz * dz);
                }
                const std::size_t index =
                    (static_cast<std::size_t>(z) * size_y + y) * size_x + x;
                energy[index] = sum;
            }
        }
    }
    return energy;
}

struct PartitionCounts {
    int output_points = 0;
    int halo_points = 0;
    int stage1_boundary_points = 0;
    int stage2_internal_points = 0;
    int sent_bytes = 0;
};

inline PartitionCounts partition_counts(int size_x, int global_size_y,
                                         int ranks) {
    if (size_x < 3 || global_size_y <= 0 || ranks <= 0 ||
        global_size_y % ranks != 0 || global_size_y / ranks < 2) {
        throw std::invalid_argument("partition dimensions");
    }
    const int local_rows = global_size_y / ranks;
    const int computed_columns = size_x - 2;
    return {local_rows * computed_columns,
            2 * size_x,
            2 * computed_columns,
            (local_rows - 2) * computed_columns,
            2 * size_x * static_cast<int>(sizeof(float))};
}

}  // namespace pmpp_examples
