#!/usr/bin/env python3
"""
AksaraLLM GGUF Exporter & Quantizer
Automates the integration with llama.cpp to convert and quantize HF checkpoints.

Usage:
    python export_gguf.py --hf-dir ../aksarallm-200m-hf --out-name aksarallm-200m.gguf
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"📦 {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    if process.returncode != 0:
        print(f"❌ Error during: {desc}")
        sys.exit(process.returncode)
    print(f"✅ Success: {desc}")


def setup_llamacpp(work_dir: str) -> str:
    """Clones and builds llama.cpp."""
    repo_url = "https://github.com/ggerganov/llama.cpp.git"
    llama_dir = os.path.join(work_dir, "llama.cpp")
    
    if not os.path.exists(llama_dir):
        print(f"\n🔄 Cloning llama.cpp into {llama_dir}...")
        run_cmd(["git", "clone", repo_url, llama_dir], "Clone llama.cpp")
    else:
        print(f"\n🔄 llama.cpp already exists at {llama_dir}, pulling latest...")
        run_cmd(["git", "-C", llama_dir, "pull"], "Update llama.cpp")
        
    # Build llama.cpp (with Metal support for Mac automatically by default on mac)
    print(f"\n🔨 Building llama.cpp tools...")
    run_cmd(["make", "-C", llama_dir, "-j4"], "Build llama.cpp")
    
    # Needs pip install -r requirements.txt for the python converter
    req_file = os.path.join(llama_dir, "requirements.txt")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", req_file], "Install Python dependencies")
    
    return llama_dir


def convert_and_quantize(hf_dir: str, llama_dir: str, output_name: str, quant_type: str = "q4_k_m"):
    """Runs conversion to F16 GGUF, then quantizes."""
    
    if not os.path.exists(hf_dir) or not os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"❌ Error: HuggingFace directory not found or invalid: {hf_dir}")
        print("Please run `convert_to_hf.py` first.")
        sys.exit(1)
        
    out_dir = os.path.dirname(os.path.abspath(output_name)) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = os.path.basename(output_name).replace(".gguf", "")
    f16_gguf = os.path.join(out_dir, f"{base_name}-f16.gguf")
    quant_gguf = os.path.join(out_dir, f"{base_name}-{quant_type.lower()}.gguf")
    
    # 1. Convert HF to GGUF F16
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Fallback to older or alternative names
        alt_names = ["convert-hf-to-gguf.py", "convert.py"]
        for name in alt_names:
            alt_path = os.path.join(llama_dir, name)
            if os.path.exists(alt_path):
                convert_script = alt_path
                break
        
    cmd_convert = [
        sys.executable, convert_script,
        hf_dir,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    run_cmd(cmd_convert, "Convert HuggingFace -> GGUF (FP16)")
    
    # 2. Quantize
    quantize_bin = os.path.join(llama_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_dir, "quantize")  # Older build name
        
    cmd_quantize = [
        quantize_bin,
        f16_gguf,
        quant_gguf,
        quant_type.upper()
    ]
    run_cmd(cmd_quantize, f"Quantize to {quant_type.upper()}")
    
    # 3. Clean up F16 to save space (optional, but requested by some users)
    if os.path.exists(f16_gguf):
        os.remove(f16_gguf)
    
    print(f"\n{'='*60}")
    print(f"🎉 GGUF Export & Quantization Complete!")
    print(f"📁 Output file: {quant_gguf}")
    print(f"📏 Size: {os.path.getsize(quant_gguf) / 1024**2:.1f} MB")
    print(f"{'='*60}")
    print("\nTo test with llama.cpp:")
    print(f"  {llama_dir}/llama-cli -m {quant_gguf} -p '[INST] Apa ibukota Indonesia? [/INST]' -n 128")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--out-name", required=True, help="Output GGUF base name (e.g., aksarallm-200m)")
    parser.add_argument("--quant", type=str, default="q4_k_m", help="Quantization type (q4_k_m, q8_0, etc)")
    parser.add_argument("--work-dir", type=str, default=".", help="Directory to clone llama.cpp into")
    args = parser.parse_args()
    
    llama_dir = setup_llamacpp(args.work_dir)
    convert_and_quantize(args.hf_dir, llama_dir, args.out_name, args.quant)

if __name__ == "__main__":
    main()
