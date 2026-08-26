#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float physical_boundary = -1.0f;

void mpi_check(int status, const char* expression) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(expression) + ": " +
                             std::string(message, length));
}

#define MPI_CHECK(expression) mpi_check((expression), #expression)

float encoded_value(int global_row, int x) {
    return static_cast<float>(global_row * 1000 + x);
}

void initialize_grid(std::vector<float>& grid, int nx, int owned_rows,
                     int rank) {
    std::fill(grid.begin(), grid.end(), physical_boundary);
    for (int local_row = 1; local_row <= owned_rows; ++local_row) {
        const int global_row = rank * owned_rows + local_row - 1;
        for (int x = 0; x < nx; ++x) {
            grid[static_cast<std::size_t>(local_row) * nx + x] =
                encoded_value(global_row, x);
        }
    }
}

std::array<MPI_Request, 4> post_halo_exchange(
    std::vector<float>& grid, int nx, int owned_rows, int top, int bottom) {
    std::array<MPI_Request, 4> requests{
        MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,
        MPI_REQUEST_NULL};
    MPI_CHECK(MPI_Irecv(grid.data(), nx, MPI_FLOAT, top, 11, MPI_COMM_WORLD,
                        &requests[0]));
    MPI_CHECK(MPI_Irecv(grid.data() +
                            static_cast<std::size_t>(owned_rows + 1) * nx,
                        nx, MPI_FLOAT, bottom, 10, MPI_COMM_WORLD,
                        &requests[1]));
    MPI_CHECK(MPI_Isend(grid.data() + nx, nx, MPI_FLOAT, top, 10,
                        MPI_COMM_WORLD, &requests[2]));
    MPI_CHECK(MPI_Isend(grid.data() +
                            static_cast<std::size_t>(owned_rows) * nx,
                        nx, MPI_FLOAT, bottom, 11, MPI_COMM_WORLD,
                        &requests[3]));
    return requests;
}

std::size_t compute_stencil_rows(const std::vector<float>& grid,
                                 std::vector<float>& output, int nx,
                                 int first_row, int last_row) {
    std::size_t computed = 0;
    for (int row = first_row; row <= last_row; ++row) {
        for (int x = 1; x < nx - 1; ++x) {
            const std::size_t center =
                static_cast<std::size_t>(row) * nx + x;
            output[center] = grid[center] + grid[center - 1] +
                             grid[center + 1] + grid[center - nx] +
                             grid[center + nx];
            ++computed;
        }
    }
    return computed;
}

void verify_halos(const std::vector<float>& grid, int nx, int owned_rows,
                  int rank, int top, int bottom) {
    for (int x = 0; x < nx; ++x) {
        const float expected_top =
            top == MPI_PROC_NULL
                ? physical_boundary
                : encoded_value(rank * owned_rows - 1, x);
        const float expected_bottom =
            bottom == MPI_PROC_NULL
                ? physical_boundary
                : encoded_value((rank + 1) * owned_rows, x);
        if (grid[static_cast<std::size_t>(x)] != expected_top) {
            throw std::runtime_error("top halo verification failed at x=" +
                                     std::to_string(x));
        }
        const std::size_t bottom_index =
            static_cast<std::size_t>(owned_rows + 1) * nx + x;
        if (grid[bottom_index] != expected_bottom) {
            throw std::runtime_error("bottom halo verification failed at x=" +
                                     std::to_string(x));
        }
    }
}

void verify_output(const std::vector<float>& output, int nx, int owned_rows,
                   int rank, int size) {
    const int global_rows = size * owned_rows;
    for (int local_row = 1; local_row <= owned_rows; ++local_row) {
        const int global_row = rank * owned_rows + local_row - 1;
        for (int x = 1; x < nx - 1; ++x) {
            const float above = global_row == 0
                                    ? physical_boundary
                                    : encoded_value(global_row - 1, x);
            const float below =
                global_row + 1 == global_rows
                    ? physical_boundary
                    : encoded_value(global_row + 1, x);
            const float expected =
                encoded_value(global_row, x) +
                encoded_value(global_row, x - 1) +
                encoded_value(global_row, x + 1) + above + below;
            const std::size_t index =
                static_cast<std::size_t>(local_row) * nx + x;
            if (!std::isfinite(output[index]) ||
                std::fabs(output[index] - expected) > 1.0e-3f) {
                throw std::runtime_error(
                    "stencil verification failed at local row=" +
                    std::to_string(local_row) + ", x=" + std::to_string(x));
            }
        }
        const std::size_t row_start =
            static_cast<std::size_t>(local_row) * nx;
        if (!std::isnan(output[row_start]) ||
            !std::isnan(output[row_start + nx - 1])) {
            throw std::runtime_error("fixed x boundary was overwritten");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int result = EXIT_FAILURE;
    try {
        int rank = 0;
        int size = 0;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        if (size < 2) {
            throw std::runtime_error("run with at least two MPI ranks");
        }

        constexpr int nx = 64;
        constexpr int owned_rows = 32;
        constexpr int local_rows = owned_rows + 2;
        std::vector<float> grid(static_cast<std::size_t>(local_rows) * nx);
        initialize_grid(grid, nx, owned_rows, rank);
        std::vector<float> output(
            grid.size(), std::numeric_limits<float>::quiet_NaN());

        const int top = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int bottom = rank + 1 == size ? MPI_PROC_NULL : rank + 1;
        std::array<MPI_Request, 4> requests =
            post_halo_exchange(grid, nx, owned_rows, top, bottom);

        const std::size_t interior_points =
            compute_stencil_rows(grid, output, nx, 2, owned_rows - 1);
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()),
                              requests.data(), MPI_STATUSES_IGNORE));
        verify_halos(grid, nx, owned_rows, rank, top, bottom);
        const std::size_t boundary_points =
            compute_stencil_rows(grid, output, nx, 1, 1) +
            compute_stencil_rows(grid, output, nx, owned_rows, owned_rows);
        verify_output(output, nx, owned_rows, rank, size);

        if (interior_points != 1860 || boundary_points != 124 ||
            interior_points + boundary_points != 1984) {
            throw std::runtime_error("partition point counts do not match");
        }
        if (rank == 0) {
            std::cout
                << "ch23_ex01_halo_exchange_mpi: nonperiodic halo, "
                   "interior-first stencil, and 1984 outputs passed\n";
        }
        result = EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "rank failure: " << error.what() << '\n';
    }
    MPI_Finalize();
    return result;
}
