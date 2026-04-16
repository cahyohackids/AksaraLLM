#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  📦 AksaraLLM GGUF Converter — Mac M-Series Optimized
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Otomatis download model dari HuggingFace → convert ke GGUF Q4_K_M
# Hasil bisa dipakai di: Ollama, LM Studio, llama.cpp
#
# Cara pakai:
#   export HF_TOKEN=hf_xxx   # opsional untuk repo private
#   bash scripts/convert_gguf.sh
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

# Config
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPO="${1:-AksaraLLM/aksarallm-1.5b-v2}"
OUTPUT_ROOT="${AKSARALLM_OUTPUT_DIR:-$ROOT_DIR/outputs}"
WORK_DIR="${GGUF_WORK_DIR:-$OUTPUT_ROOT/gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORK_DIR/llama.cpp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

print_step() { echo -e "\n\033[1;32m[+] $1\033[0m"; }
print_info() { echo -e "\033[1;36m    → $1\033[0m"; }
print_error() { echo -e "\033[1;31m[!] $1\033[0m"; }

# ── Step 1: Prepare workspace ──
print_step "STEP 1: Menyiapkan workspace..."
mkdir -p "$WORK_DIR"

# ── Step 2: Clone/Update llama.cpp ──
print_step "STEP 2: Menyiapkan llama.cpp..."
if [ -d "$LLAMA_CPP_DIR" ]; then
    print_info "llama.cpp sudah ada, update..."
    cd "$LLAMA_CPP_DIR"
    git pull --quiet 2>/dev/null || true
else
    print_info "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
fi

# ── Step 3: Build llama.cpp (Metal acceleration for Mac) ──
print_step "STEP 3: Kompilasi llama.cpp (Metal GPU acceleration)..."
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    mkdir -p build && cd build
    cmake .. -DLLAMA_METAL=ON -DCMAKE_BUILD_TYPE=Release 2>/dev/null
    cmake --build . --config Release -j$(sysctl -n hw.ncpu) 2>/dev/null
    cd "$LLAMA_CPP_DIR"
    print_info "Build selesai!"
else
    print_info "Sudah ter-build sebelumnya."
fi

# ── Step 4: Install Python deps ──
print_step "STEP 4: Install Python dependencies..."
"$PYTHON_BIN" -m pip install huggingface_hub transformers sentencepiece protobuf gguf --quiet 2>/dev/null

# ── Step 5: Download model from HuggingFace ──
print_step "STEP 5: Download model dari $MODEL_REPO..."
MODEL_DIR="$WORK_DIR/model"
"$PYTHON_BIN" -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR', token=os.environ.get('HF_TOKEN') or None)
print('✅ Download selesai!')
"

# ── Step 6: Convert to GGUF F16 ──
print_step "STEP 6: Convert ke GGUF F16..."
GGUF_F16="$WORK_DIR/aksarallm-1.5b-v2-F16.gguf"
"$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" \
    --outfile "$GGUF_F16" \
    --outtype f16
print_info "F16: $(du -h "$GGUF_F16" | cut -f1)"

# ── Step 7: Quantize to Q4_K_M ──
print_step "STEP 7: Quantize ke Q4_K_M (optimal untuk Mac)..."
GGUF_Q4="$WORK_DIR/aksarallm-1.5b-v2-Q4_K_M.gguf"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
print_info "Q4_K_M: $(du -h "$GGUF_Q4" | cut -f1)"

# ── Step 8: Quick test ──
print_step "STEP 8: Quick test..."
"$LLAMA_CPP_DIR/build/bin/llama-cli" \
    -m "$GGUF_Q4" \
    -p "<|im_start|>system\nKamu adalah AksaraLLM, asisten AI berbahasa Indonesia.<|im_end|>\n<|im_start|>user\nSiapa kamu?<|im_end|>\n<|im_start|>assistant\n" \
    -n 100 \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.15 \
    2>/dev/null

# ── Step 9: Create Ollama Modelfile ──
print_step "STEP 9: Buat Ollama Modelfile..."
MODELFILE="$WORK_DIR/Modelfile"
cat > "$MODELFILE" << 'EOF'
FROM ./aksarallm-1.5b-v2-Q4_K_M.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.15
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
EOF

# ── Done! ──
print_step "🎉 SELESAI!"
echo ""
echo -e "\033[1;36m📁 File output:\033[0m"
echo "   F16:    $GGUF_F16 ($(du -h "$GGUF_F16" | cut -f1))"
echo "   Q4_K_M: $GGUF_Q4 ($(du -h "$GGUF_Q4" | cut -f1))"
echo ""
echo -e "\033[1;33m🚀 Cara pakai Ollama:\033[0m"
echo "   cd $WORK_DIR"
echo "   ollama create aksarallm -f Modelfile"
echo "   ollama run aksarallm"
echo ""
echo -e "\033[1;33m🚀 Cara pakai LM Studio:\033[0m"
echo "   Drag & drop file $GGUF_Q4 ke LM Studio"
echo ""
echo -e "\033[1;33m🚀 Cara pakai llama.cpp:\033[0m"
echo "   $LLAMA_CPP_DIR/build/bin/llama-cli -m $GGUF_Q4 -cnv"
echo ""
