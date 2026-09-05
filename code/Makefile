NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -lineinfo
CUDA_ARCH ?=
ARCH_FLAGS := $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH))

BUILD_DIR := build
TARGETS := \
	$(BUILD_DIR)/vec_add \
	$(BUILD_DIR)/rgb_to_grayscale \
	$(BUILD_DIR)/image_blur \
	$(BUILD_DIR)/matrix_multiply

.PHONY: all check-nvcc run clean

all: $(TARGETS)

check-nvcc:
	@command -v "$(NVCC)" >/dev/null 2>&1 || { \
		echo "error: nvcc was not found; install the CUDA Toolkit or set NVCC=/path/to/nvcc"; \
		exit 1; \
	}

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGETS): | check-nvcc $(BUILD_DIR)

$(BUILD_DIR)/vec_add: ch2/vec_add.cu common/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) "$<" -o "$@"

$(BUILD_DIR)/rgb_to_grayscale: ch3/rgb_to_grayscale.cu common/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) "$<" -o "$@"

$(BUILD_DIR)/image_blur: ch3/image_blur.cu common/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) "$<" -o "$@"

$(BUILD_DIR)/matrix_multiply: ch3/matrix_multiply.cu common/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) "$<" -o "$@"

run: all
	"$(BUILD_DIR)/vec_add" 1003
	"$(BUILD_DIR)/rgb_to_grayscale"
	"$(BUILD_DIR)/image_blur"
	"$(BUILD_DIR)/matrix_multiply"

# 只删除上面列出的四个可执行文件，不递归删除目录。
clean:
	$(RM) $(TARGETS)
