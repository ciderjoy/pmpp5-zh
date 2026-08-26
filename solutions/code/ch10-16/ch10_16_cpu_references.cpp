#include "reference_algorithms.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

int main() {
    try {
        using test_case = std::pair<const char*, std::function<void()>>;
        const std::vector<test_case> tests{
            {"chapter 10 reductions", pmpp::reference::test_ch10},
            {"chapter 11 scans", pmpp::reference::test_ch11},
            {"chapter 12 filters", pmpp::reference::test_ch12},
            {"chapter 13 merge/co-rank", pmpp::reference::test_ch13},
            {"chapter 14 sorts", pmpp::reference::test_ch14},
            {"chapter 15 dense GEMM", pmpp::reference::test_ch15},
            {"chapter 16 dynamic programming", pmpp::reference::test_ch16},
        };
        for (const auto& test : tests) {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        }
        std::cout << "ch10_16_cpu_references: " << tests.size()
                  << " reference groups passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
