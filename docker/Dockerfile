# syntax=docker/dockerfile:1.7
# sam3-instructsam — natural-language image segmentation, CPU-only.
#
# Builds a self-contained image whose entrypoint is `instructsam`.
# Weights are converted from CircleRadon/InstructSAM-2B AT BUILD TIME
# (~4 GB safetensors download + ~5 min conversion on a fast box), so the
# published image is a self-contained runtime — no HF pull at run-time.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t sam3-instructsam:latest .
#
# Run (mount an input image + output dir):
#   docker run --rm \
#     -v /path/to/images:/in \
#     -v /path/to/out:/out \
#     sam3-instructsam:latest \
#       --image /in/warehouse.jpg \
#       --query "Please segment 'box', 'person', 'forklift' in this image." \
#       --out-dir /out --run-vision
#
# Notes:
# - CPU-only. End-to-end ≈1.7 min on a 12-core CPU for a 640×426 input.
# - The Q4_K_M quant is the default LM; f16 is also included if you want
#   to override with --lm /models/instructsam-lm-f16.gguf.
# - The InstructSAM-2B checkpoint does NOT declare an explicit license
#   on its HF card. This Dockerfile pulls from CircleRadon at build time
#   so you inherit whatever terms apply to their upstream — we do not
#   redistribute derived weights in the image layers that get published.

# ── Stage 1: build llama.cpp + sam3cpp + convert weights ───────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        python3 python3-pip python3-venv python3-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned llama.cpp commit — the exact tree our fork was developed against.
# Bumping means re-verifying the Qwen3-VL InstructSAM tensor-filter patch
# and the mtmd API surface used by tools/sam3_instructsam_cli.cpp.
ARG LLAMA_CPP_COMMIT=1425386

# Python venv for conversion (llama.cpp pins transformers==4.57.6).
RUN python3 -m venv /opt/convert-venv \
    && /opt/convert-venv/bin/pip install --no-cache-dir --upgrade pip

WORKDIR /src

# sam3cpp source. Default is COPY from the build context so this Dockerfile
# works against the local checkout without needing to push a branch. Set
# --build-arg SAM3CPP_SOURCE=git to clone from GitHub instead (useful for
# reproducible builds off a tag).
ARG SAM3CPP_SOURCE=copy
ARG SAM3CPP_REPO=https://github.com/smellslikeml/sam3cpp.git
ARG SAM3CPP_BRANCH=feat/instructsam-decoder

# COPY is unconditional (Docker can't skip it), so we accept the copy
# either way and then optionally override with a git clone. This costs a
# few MB of layer bloat when SAM3CPP_SOURCE=git; not worth branching over.
COPY . /src/sam3cpp
RUN if [ "${SAM3CPP_SOURCE}" = "git" ]; then \
        rm -rf /src/sam3cpp && \
        git clone --depth=1 --branch "${SAM3CPP_BRANCH}" "${SAM3CPP_REPO}" /src/sam3cpp; \
    fi

# Clone + build llama.cpp at the pinned commit, applying the InstructSAM
# tensor-filter patch needed for LM + mmproj conversion.
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git checkout "${LLAMA_CPP_COMMIT}" \
    && git apply /src/sam3cpp/docs/instructsam/llamacpp-qwen3vl-instructsam-filter.patch

# Install conversion deps from llama.cpp's pinned list.
RUN /opt/convert-venv/bin/pip install --no-cache-dir \
        -r /src/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# Build llama.cpp static libs (CPU-only, minimal — mtmd + core).
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
        -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF \
        -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/llama.cpp/build \
        --target llama common mtmd ggml ggml-cpu ggml-base llama-mtmd-cli llama-quantize \
        --parallel "$(nproc)"

# ── Download source safetensors + convert to GGUF ──────────────────────
# huggingface_hub is already pulled in by the conversion requirements.
ARG INSTRUCTSAM_REPO=CircleRadon/InstructSAM-2B
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN /opt/convert-venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${INSTRUCTSAM_REPO}', local_dir='/src/instructsam-src', local_dir_use_symlinks=False)"

# Rewrite config.json + tokenizer_config.json so llama.cpp's Qwen3-VL
# converter sees a vanilla Qwen3VLForConditionalGeneration checkpoint.
# (Mirrors tools/pathb_convert_lm_mmproj.sh steps 2 + 3. Actual rewrite
# logic lives in docker/rewrite_config_for_qwen3vl.py which we already
# COPYed in as part of /src/sam3cpp.)
RUN mkdir -p /src/instructsam-as-qwen3vl && \
    cp /src/instructsam-src/model.safetensors  /src/instructsam-as-qwen3vl/ && \
    cp /src/instructsam-src/tokenizer.json     /src/instructsam-as-qwen3vl/ && \
    for f in chat_template.jinja generation_config.json processor_config.json; do \
        [ -f "/src/instructsam-src/$f" ] && cp "/src/instructsam-src/$f" /src/instructsam-as-qwen3vl/ ; \
    done && \
    /opt/convert-venv/bin/python /src/sam3cpp/docker/rewrite_config_for_qwen3vl.py \
        /src/instructsam-src /src/instructsam-as-qwen3vl

# Convert LM half (f16 for accuracy) + Q4_K_M quant (default runtime).
RUN mkdir -p /models \
    && cd /src/llama.cpp \
    && /opt/convert-venv/bin/python convert_hf_to_gguf.py \
        --outfile /models/instructsam-lm-f16.gguf \
        --outtype f16 \
        /src/instructsam-as-qwen3vl \
    && ./build/bin/llama-quantize \
        --token-embedding-type f16 \
        /models/instructsam-lm-f16.gguf \
        /models/instructsam-lm-Q4_K_M.gguf Q4_K_M
# --token-embedding-type f16: llama-quantize's default for Q4_K_M is to
# downcast token_embd.weight to Q6_K, but instructsam's
# InstructsamLmForward::embed_for_token (used to fetch the mask_start /
# mask_end / phrase-token embeddings that seed the seg-output injection)
# only supports F32 and F16. Force F16 here so both paths agree.

# Convert vision projector (mmproj).
RUN cd /src/llama.cpp \
    && /opt/convert-venv/bin/python convert_hf_to_gguf.py \
        --outfile /models/instructsam-mmproj-f16.gguf \
        --outtype f16 --mmproj \
        /src/instructsam-as-qwen3vl

# Convert grounding half (DETR + FPN + mask decoder + dot_prod_scoring)
# and extract the mask_queries sidecar.
RUN /opt/convert-venv/bin/pip install --no-cache-dir gguf safetensors torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && /opt/convert-venv/bin/python /src/sam3cpp/tools/convert_instructsam_to_gguf.py \
        --input  /src/instructsam-src/model.safetensors \
        --output /models/instructsam-grounding-f16.gguf \
    && /opt/convert-venv/bin/python /src/sam3cpp/tools/extract_mask_queries.py \
        --input  /src/instructsam-src/model.safetensors \
        --output /models/instructsam_mask_queries.f32

# Build sam3cpp against the just-built llama.cpp tree.
RUN cd /src/sam3cpp \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target instructsam --parallel "$(nproc)"

# Reclaim disk: drop safetensors + intermediate build trees. The runtime
# stage only needs the built binary, the shared libs, and /models.
RUN rm -rf /src/instructsam-src /src/instructsam-as-qwen3vl \
           /src/llama.cpp /root/.cache /opt/convert-venv

# ── Stage 2: runtime image ────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 libgomp1 libc6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary + models.
COPY --from=builder /src/sam3cpp/build/instructsam /usr/local/bin/instructsam
COPY --from=builder /models /models

# Default model paths — override with --lm / --mmproj / --grounding /
# --mask-queries if desired.
ENV SAM3_LM=/models/instructsam-lm-Q4_K_M.gguf \
    SAM3_MMPROJ=/models/instructsam-mmproj-f16.gguf \
    SAM3_GROUNDING=/models/instructsam-grounding-f16.gguf \
    SAM3_MASK_QUERIES=/models/instructsam_mask_queries.f32

# Thin wrapper that plugs the model paths in as defaults so the caller
# only needs --image / --query / --out-dir. Any explicit --lm / --mmproj
# etc. overrides these.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
