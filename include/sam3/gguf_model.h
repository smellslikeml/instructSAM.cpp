#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sam3 {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::string type;
    size_t nbytes = 0;
};

struct Metadata {
    std::string architecture;
    std::string source_repo;
    std::string source_impl;
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t vision_layers = 0;
    int32_t text_layers = 0;
};

class GgufModel {
public:
    GgufModel() = default;
    ~GgufModel();

    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;

    // Loading modes:
    //   prefer_gpu=true, use_mmap=false (default): eager load into GPU/CPU
    //     backend buffer — ~2x file size peak RAM (tmp_ctx + backend copy).
    //     Fine for smaller GGUFs (grounding half, <2GB).
    //
    //   use_mmap=true: mmap the file, tensors point directly into the
    //     mmap region. Ignores prefer_gpu (mmap is always CPU-side).
    //     Peak RAM ~= file size, but only pages actually touched are
    //     faulted in. Right for the 3.3GB LM GGUF on constrained boxes.
    bool load(const std::string & path,
              bool prefer_gpu = true,
              const std::string & tensor_map_path = {},
              bool use_mmap = false);

    const Metadata & metadata() const { return metadata_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }
    ggml_backend_t backend() const { return backend_; }
    ggml_backend_t cpu_backend() const;

    bool has_tensor(const std::string & name) const;
    const TensorInfo * find_tensor(const std::string & name) const;
    ggml_tensor * find_weight(const std::string & name) const;
    std::string resolve_tensor_name(const std::string & name) const;

private:
    Metadata metadata_{};
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    std::unordered_map<std::string, std::string> tensor_aliases_;

    ggml_context * weights_ctx_ = nullptr;
    gguf_context * gguf_ctx_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    mutable ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    // mmap-mode bookkeeping
    void * mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    void clear();
};

}  // namespace sam3
