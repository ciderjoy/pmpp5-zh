#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    try {
        const float one = 1.0f;
        const float next = std::nextafter(one,
                                          std::numeric_limits<float>::infinity());
        const float ulp_at_one = next - one;
        if (ulp_at_one != std::ldexp(1.0f, -23)) {
            throw std::runtime_error("unexpected binary32 spacing at one");
        }

        volatile float half_ulp = std::ldexp(1.0f, -24);
        volatile float sequential = one;
        sequential = sequential + half_ulp;
        sequential = sequential + half_ulp;
        volatile float paired_small = half_ulp + half_ulp;
        volatile float paired = one + paired_small;
        if (sequential != one || paired != next ||
            (paired - sequential) != ulp_at_one) {
            throw std::runtime_error("reduction-order example failed");
        }

        constexpr int binary32_precision = 24;
        const int ulps_after_9_cycles =
            1 << (binary32_precision - 2 * 9);
        const int ulps_after_10_cycles =
            1 << (binary32_precision - 2 * 10);
        if (ulps_after_9_cycles != 64 || ulps_after_10_cycles != 16) {
            throw std::runtime_error("iterative precision count failed");
        }

        std::cout << "appendix_a_floating_point: ULP/order/iteration checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
