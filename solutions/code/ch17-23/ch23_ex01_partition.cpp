#include <cstdlib>
#include <iostream>
#include <stdexcept>

struct PartitionCounts {
    int output_points;
    int halo_points;
    int stage1_boundary_points;
    int stage2_internal_points;
    int sent_bytes;
};

PartitionCounts jacobi_partition_counts(int nx, int global_ny, int ranks) {
    if (nx < 3 || global_ny <= 0 || ranks <= 0 || global_ny % ranks != 0) {
        throw std::invalid_argument("invalid Jacobi partition dimensions");
    }
    const int owned_rows = global_ny / ranks;
    if (owned_rows < 2) {
        throw std::invalid_argument("each rank needs at least two owned rows");
    }
    const int computed_columns = nx - 2;
    return {
        owned_rows * computed_columns,
        2 * nx,
        2 * computed_columns,
        (owned_rows - 2) * computed_columns,
        2 * nx * static_cast<int>(sizeof(float)),
    };
}

int main() {
    try {
        const PartitionCounts counts = jacobi_partition_counts(64, 512, 16);
        if (counts.output_points != 1984 || counts.halo_points != 128 ||
            counts.stage1_boundary_points != 124 ||
            counts.stage2_internal_points != 1860 ||
            counts.sent_bytes != 512 ||
            counts.stage1_boundary_points + counts.stage2_internal_points !=
                counts.output_points) {
            throw std::runtime_error("chapter 23 partition count mismatch");
        }
        std::cout << "ch23_ex01_partition: 1984 128 124 1860 512\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
