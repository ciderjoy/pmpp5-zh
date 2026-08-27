#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pmpp::reference {

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class T>
void verify_exact(const std::vector<T>& actual,
                  const std::vector<T>& expected,
                  const std::string& label) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(label + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

inline void verify_close(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float absolute_tolerance,
                         float relative_tolerance,
                         const std::string& label) {
    require(actual.size() == expected.size(), label + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance =
            absolute_tolerance + relative_tolerance * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(label + ": mismatch at index " +
                                     std::to_string(i));
        }
    }
}

inline float reduction_sum(const std::vector<float>& input) {
    return std::accumulate(input.begin(), input.end(), 0.0f);
}

inline float reduction_max(const std::vector<float>& input) {
    if (input.empty()) {
        return -std::numeric_limits<float>::infinity();
    }
    return *std::max_element(input.begin(), input.end());
}

inline std::vector<float> inclusive_scan(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    std::partial_sum(input.begin(), input.end(), output.begin());
    return output;
}

inline std::vector<float> block_inclusive_scan(
    const std::vector<float>& input, std::size_t segment_size) {
    require(segment_size > 0, "scan segment size must be positive");
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t base = 0; base < input.size(); base += segment_size) {
        float prefix = 0.0f;
        const std::size_t end = std::min(base + segment_size, input.size());
        for (std::size_t i = base; i < end; ++i) {
            prefix += input[i];
            output[i] = prefix;
        }
    }
    return output;
}

inline std::vector<int> stable_filter_nonzero(
    const std::vector<int>& input) {
    std::vector<int> output;
    output.reserve(input.size());
    std::copy_if(input.begin(), input.end(), std::back_inserter(output),
                 [](int value) { return value != 0; });
    return output;
}

inline int co_rank(int k, const std::vector<int>& left,
                   const std::vector<int>& right) {
    require(k >= 0 &&
                static_cast<std::size_t>(k) <= left.size() + right.size(),
            "co-rank outside merged range");
    int low = std::max(0, k - static_cast<int>(right.size()));
    int high = std::min(k, static_cast<int>(left.size()));
    while (low < high) {
        const int i = low + (high - low) / 2;
        const int j = k - i;
        if (j > 0 && i < static_cast<int>(left.size()) &&
            right[static_cast<std::size_t>(j - 1)] >=
                left[static_cast<std::size_t>(i)]) {
            low = i + 1;
        } else {
            high = i;
        }
    }
    return low;
}

inline std::vector<int> stable_merge(const std::vector<int>& left,
                                     const std::vector<int>& right) {
    require(std::is_sorted(left.begin(), left.end()) &&
                std::is_sorted(right.begin(), right.end()),
            "merge inputs must be sorted");
    std::vector<int> output(left.size() + right.size());
    std::merge(left.begin(), left.end(), right.begin(), right.end(),
               output.begin());
    return output;
}

inline std::vector<unsigned> radix_sort_reference(
    std::vector<unsigned> input) {
    std::stable_sort(input.begin(), input.end());
    return input;
}

inline std::vector<int> merge_sort_reference(std::vector<int> input) {
    std::stable_sort(input.begin(), input.end());
    return input;
}

inline std::vector<float> matmul(const std::vector<float>& left,
                                 const std::vector<float>& right, int rows,
                                 int inner, int columns) {
    require(rows >= 0 && inner >= 0 && columns >= 0,
            "negative matrix dimension");
    require(left.size() == static_cast<std::size_t>(rows) * inner &&
                right.size() == static_cast<std::size_t>(inner) * columns,
            "matrix size mismatch");
    std::vector<float> output(
        static_cast<std::size_t>(rows) * columns, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            double sum = 0.0;
            for (int k = 0; k < inner; ++k) {
                sum += static_cast<double>(
                           left[static_cast<std::size_t>(row) * inner + k]) *
                       right[static_cast<std::size_t>(k) * columns + column];
            }
            output[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(sum);
        }
    }
    return output;
}

inline std::vector<std::int64_t> floyd_warshall(
    std::vector<std::int64_t> distances, int n, std::int64_t infinity) {
    require(n >= 0 &&
                distances.size() == static_cast<std::size_t>(n) * n,
            "Floyd-Warshall matrix size mismatch");
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            const std::int64_t left =
                distances[static_cast<std::size_t>(i) * n + k];
            if (left == infinity) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                const std::int64_t right =
                    distances[static_cast<std::size_t>(k) * n + j];
                if (right == infinity) {
                    continue;
                }
                if ((right > 0 &&
                     left > std::numeric_limits<std::int64_t>::max() -
                                right) ||
                    (right < 0 &&
                     left < std::numeric_limits<std::int64_t>::min() -
                                right)) {
                    throw std::overflow_error(
                        "Floyd-Warshall path sum exceeds int64 range");
                }
                const std::int64_t via = left + right;
                std::int64_t& current =
                    distances[static_cast<std::size_t>(i) * n + j];
                if (via < current) {
                    current = via;
                }
            }
        }
    }
    return distances;
}

struct sw_scores {
    int match = 2;
    int mismatch = -1;
    int gap = -2;
};

inline std::vector<int> smith_waterman(std::string_view read,
                                       std::string_view reference,
                                       sw_scores scores = {}) {
    const std::size_t rows = read.size() + 1;
    const std::size_t columns = reference.size() + 1;
    std::vector<int> table(rows * columns, 0);
    for (std::size_t row = 1; row < rows; ++row) {
        for (std::size_t column = 1; column < columns; ++column) {
            const int substitution =
                read[row - 1] == reference[column - 1] ? scores.match
                                                        : scores.mismatch;
            const int diagonal =
                table[(row - 1) * columns + column - 1] + substitution;
            const int west = table[row * columns + column - 1] + scores.gap;
            const int north =
                table[(row - 1) * columns + column] + scores.gap;
            table[row * columns + column] =
                std::max({0, diagonal, west, north});
        }
    }
    return table;
}

inline void test_ch10() {
    std::vector<float> input(1031);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 31) - 20) / 8.0f;
    }
    require(std::isfinite(reduction_sum(input)),
            "chapter 10 sum is not finite");
    require(reduction_max(input) == 1.25f,
            "chapter 10 maximum fixture failed");
    require(reduction_sum({}) == 0.0f,
            "chapter 10 empty sum identity failed");
    require(std::isinf(reduction_max({})) && reduction_max({}) < 0.0f,
            "chapter 10 empty maximum identity failed");
}

inline void test_ch11() {
    const std::vector<float> input{4, 6, 7, 1, 2, 8, 5, 2};
    const std::vector<float> expected{4, 10, 17, 18, 20, 28, 33, 35};
    verify_exact(inclusive_scan(input), expected,
                 "chapter 11 inclusive scan");
    const std::vector<float> blocked = block_inclusive_scan(input, 4);
    verify_exact(blocked, std::vector<float>{4, 10, 17, 18, 2, 10, 15, 17},
                 "chapter 11 block scan reset");
}

inline void test_ch12() {
    const std::vector<int> input{0, 4, 0, -2, 7, 0, 0, 9};
    verify_exact(stable_filter_nonzero(input),
                 std::vector<int>{4, -2, 7, 9},
                 "chapter 12 stable filter");
    require(stable_filter_nonzero({}).empty(),
            "chapter 12 empty filter failed");
}

inline void test_ch13() {
    const std::vector<int> left{1, 7, 8, 9, 10};
    const std::vector<int> right{7, 10, 10, 12};
    require(co_rank(8, left, right) == 5,
            "chapter 13 rank-8 co-rank failed");
    verify_exact(stable_merge(left, right),
                 std::vector<int>{1, 7, 7, 8, 9, 10, 10, 10, 12},
                 "chapter 13 stable merge");
}

inline void test_ch14() {
    const std::vector<unsigned> unsigned_input{
        19, 0, 7, 7, 255, 3, 1, 128, 3, 42, 1};
    const std::vector<unsigned> unsigned_expected{
        0, 1, 1, 3, 3, 7, 7, 19, 42, 128, 255};
    verify_exact(radix_sort_reference(unsigned_input), unsigned_expected,
                 "chapter 14 radix sort");
    const std::vector<int> signed_input{5, -1, 5, 3, -8, 0, 3};
    verify_exact(merge_sort_reference(signed_input),
                 std::vector<int>{-8, -1, 0, 3, 3, 5, 5},
                 "chapter 14 merge sort");
}

inline void test_ch15() {
    constexpr int rows = 7;
    constexpr int inner = 5;
    constexpr int columns = 9;
    std::vector<float> left(static_cast<std::size_t>(rows) * inner);
    std::vector<float> right(static_cast<std::size_t>(inner) * columns,
                             0.0f);
    for (std::size_t i = 0; i < left.size(); ++i) {
        left[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 4.0f;
    }
    for (int k = 0; k < inner; ++k) {
        right[static_cast<std::size_t>(k) * columns + k] = 1.0f;
    }
    const std::vector<float> output =
        matmul(left, right, rows, inner, columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < inner; ++column) {
            require(output[static_cast<std::size_t>(row) * columns + column] ==
                        left[static_cast<std::size_t>(row) * inner + column],
                    "chapter 15 rectangular identity fixture failed");
        }
    }
}

inline void test_ch16() {
    constexpr std::int64_t infinity = 1'000'000;
    const std::vector<std::int64_t> graph{
        0, 3, 10, 20,
        infinity, 0, 4, infinity,
        infinity, infinity, 0, 2,
        infinity, infinity, infinity, 0,
    };
    const std::vector<std::int64_t> shortest =
        floyd_warshall(graph, 4, infinity);
    require(shortest[3] == 9 && shortest[1 * 4 + 3] == 6,
            "chapter 16 Floyd-Warshall paths failed");

    const std::vector<int> table = smith_waterman("AC", "AC");
    require(table.size() == 9 && table.back() == 4 &&
                *std::max_element(table.begin(), table.end()) == 4,
            "chapter 16 Smith-Waterman fixture failed");
    const std::vector<int> empty = smith_waterman("", "AC");
    require(empty == std::vector<int>({0, 0, 0}),
            "chapter 16 empty Smith-Waterman boundary failed");
}

}  // namespace pmpp::reference
