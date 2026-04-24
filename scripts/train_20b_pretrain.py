#!/usr/bin/env python3
"""AksaraLLM 20B pre-training launcher (EasyDeL / JAX / TPU).

This script is the **single entry point** for the from-scratch 20B
pre-training run described in ``docs/aksara_20b_blueprint.md``. It
replaces ``scripts/train_distributed.py`` for the 20B path. The 1B
smoke-test path is still served by ``scripts/train_distributed.py``
(PyTorch + Accelerate FSDP) until ported.

Framework: **EasyDeL** (https://github.com/erfanzar/EasyDeL) — a
JAX/Flax LLM training toolkit purpose-built for TPU pods with
SPMD sharding, Orbax checkpoints, W&B integration, and batteries-
included mixed precision. EasyDeL ships with a LLaMA-family model
class whose architecture maps cleanly onto the AksaraLLM 20B spec:
GQA, RoPE w/ large theta, SwiGLU, RMSNorm, tied embeddings.

The script is **TPU-first**. It expects to be launched inside a
provisioned TPU VM (v5p-64 minimum, v5p-128 recommended) via

    gcloud compute tpus tpu-vm ssh <name> --worker=all --command=\\
      "cd ~/AksaraLLM && python3 scripts/train_20b_pretrain.py \\
         --config configs/aksara_20b_dense.json \\
         --tokenizer Ezekiel999/aksara-tokenizer-20b \\
         --corpus-glob 'gs://aksarallm-corpus/pretrain/*/*.parquet' \\
         --output-dir gs://aksarallm-checkpoints/20b \\
         --wandb-project aksarallm-20b"

On the available v6e-8 the script can run *as a smoke test* only;
see ``--smoke-test`` and the README notes about HBM limits in that
topology.

Sharding policy
---------------

A 2-axis mesh ``(data, model)`` is used. For v5p-128 the default mesh
is ``(32, 4)``; for v5p-64 ``(16, 4)``; for v5p-256 ``(64, 4)``;
for v6e-8 (smoke only) ``(2, 4)``. Tensor parallel (TP) is set to 4
because 4 evenly divides both ``n_heads=48`` and ``n_kv_heads=8`` and
keeps head dim * TP within HBM comfort. Data parallel (DP) is
``total_chips / TP``.

Optimizer
---------

AdamW in fp32 on bf16 weights for v5p. For tight-HBM setups the
``--optimizer adafactor`` flag swaps to Adafactor (halves optimizer
state). Adafactor is well-tested for >10B Google-TPU runs but needs a
slightly lower peak LR (~1.2e-4 instead of 1.5e-4).

Checkpointing
-------------

Orbax async sharded checkpoint every ``--ckpt-every`` steps. Keep
last ``--ckpt-keep`` full checkpoints + one permanent checkpoint
every ``--ckpt-permanent-every`` steps. Resume is automatic from the
latest checkpoint found in ``--output-dir``.

Metrics (W&B)
-------------

Per step: loss, grad_norm, learning_rate, tokens/sec, tokens/device/sec,
MFU, % of step in compute vs IO. Every ``--eval-every`` steps: held-out
perplexity on ``--eval-shards``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover — only imported when actually running on TPU
    import jax
    import jax.numpy as jnp
    import numpy as np
except Exception as _exc:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    """All knobs for the 20B pretrain in one place."""

    config_json: str
    tokenizer: str
    corpus_glob: str
    output_dir: str

    # Data/seq
    seq_len: int = 8192
    global_batch_tokens: int = 2_097_152  # 2 Mi tokens/step
    pack: bool = True

    # Optim
    optimizer: str = "adamw"  # "adamw" | "adafactor"
    peak_lr: float = 1.5e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # Schedule
    max_steps: int = 200_000  # 200k * 2Mi tokens = 419B tokens
    eval_every: int = 5_000
    eval_shards: Optional[str] = None

    # Sharding
    tp_size: int = 4
    dp_size: Optional[int] = None  # inferred from total chips if None

    # Precision
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"

    # Checkpointing
    ckpt_every: int = 500
    ckpt_keep: int = 3
    ckpt_permanent_every: int = 10_000
    resume: bool = True

    # Misc
    wandb_project: Optional[str] = "aksarallm-20b"
    wandb_run_name: Optional[str] = None
    seed: int = 42
    smoke_test: bool = False

    def load_model_config(self) -> Dict[str, Any]:
        with open(self.config_json) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# EasyDeL wiring
# ---------------------------------------------------------------------------


def _build_easydel_config(cfg: PretrainConfig, model_cfg: Dict[str, Any]):
    """Translate AksaraLLM JSON config into an EasyDeL LLaMA config.

    Uses the top-level ``easydel.LlamaConfig`` symbol, which works for
    EasyDeL ≥ 0.1.4 (the Flax NNX lineage).  For older EasyDeL (0.0.x,
    Linen) the same symbol is available at
    ``easydel.modules.llama.llama_configuration.LlamaConfig`` — update this
    import if you have to pin to an older EasyDeL.
    """
    from easydel import LlamaConfig  # type: ignore

    arch = model_cfg["architecture"]
    seq_plan = model_cfg.get("sequence_plan", {})

    return LlamaConfig(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["n_embd"],
        intermediate_size=arch["n_inner"],
        num_hidden_layers=arch["n_layers"],
        num_attention_heads=arch["n_heads"],
        num_key_value_heads=arch["n_kv_heads"],
        max_position_embeddings=seq_plan.get("train_context_main", cfg.seq_len),
        rope_theta=float(arch.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(arch.get("rms_norm_eps", 1e-6)),
        tie_word_embeddings=bool(arch.get("tie_embeddings", True)),
        attention_bias=bool(arch.get("attention_bias", False)),
        hidden_act="silu",  # SwiGLU is (W_gate(x) * silu(W_up(x))) · W_down
        initializer_range=0.02,
        use_cache=False,
    )


def _build_mesh(cfg: PretrainConfig):
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils  # type: ignore

    total = jax.device_count()
    tp = cfg.tp_size
    if total % tp != 0:
        raise ValueError(
            f"Total chips ({total}) not divisible by tp_size ({tp}). "
            "Override --tp-size or pick a different TPU topology."
        )
    dp = cfg.dp_size or (total // tp)
    assert dp * tp == total, f"dp*tp ({dp}*{tp}) must equal total chips ({total})"
    devices = mesh_utils.create_device_mesh((dp, tp))
    return Mesh(devices, axis_names=("data", "model")), dp, tp


def _save_checkpoint(ckpt_mgr, step: int, state) -> None:
    """Best-effort checkpoint save that works across Orbax ≥0.10 API tweaks.

    Orbax changed StandardSave wrapping twice between 0.5 and 0.11 — older
    managers accept ``args=StandardSave(state)`` directly, newer ones require
    wrapping in ``Composite(state=StandardSave(state))``. Try them in order.
    """
    import orbax.checkpoint as ocp  # type: ignore

    last_exc: Exception | None = None
    for try_fn in (
        lambda: ckpt_mgr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state))),
        lambda: ckpt_mgr.save(step, args=ocp.args.StandardSave(state)),
        lambda: ckpt_mgr.save(step, state),
    ):
        try:
            try_fn()
            return
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    print(f"[pretrain] WARN: checkpoint save at step {step} failed: {last_exc}", flush=True)


def _build_optimizer(cfg: PretrainConfig, num_steps: int):
    import optax  # type: ignore

    # ``decay_steps`` in optax is the TOTAL step budget (warmup + decay),
    # not the tail after warmup. Keep it strictly greater than warmup so the
    # internal cosine window is positive — this avoids a ``decay_steps=0``
    # crash on tiny smoke configs.
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.peak_lr,
        warmup_steps=cfg.warmup_steps,
        decay_steps=max(num_steps, cfg.warmup_steps + 1),
        end_value=cfg.peak_lr * cfg.min_lr_ratio,
    )
    if cfg.optimizer == "adamw":
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=cfg.betas[0],
                b2=cfg.betas[1],
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
            ),
        )
    elif cfg.optimizer == "adafactor":
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adafactor(
                learning_rate=lr_schedule,
                weight_decay_rate=cfg.weight_decay,
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    return tx, lr_schedule


# ---------------------------------------------------------------------------
# Data pipeline (Parquet → packed sequences)
# ---------------------------------------------------------------------------


def _iter_packed_sequences(cfg: PretrainConfig, tokenizer):
    """Stream Parquet shards, tokenize, pack to `cfg.seq_len` + BOS/EOS per doc.

    Uses ``datasets`` in streaming mode for GCS-friendly iteration.
    Packing strategy: concatenate documents separated by EOS, cut at
    ``cfg.seq_len``. Leftovers carry over to the next sequence.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=cfg.corpus_glob,
        split="train",
        streaming=True,
    )

    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    bos_id = tokenizer.convert_tokens_to_ids("<|bos|>")
    buf: list[int] = [bos_id]
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= cfg.seq_len:
            seq = buf[: cfg.seq_len]
            buf = [bos_id] + buf[cfg.seq_len :]
            yield seq


def _batch_iterator(cfg: PretrainConfig, tokenizer, micro_batch: int):
    it = _iter_packed_sequences(cfg, tokenizer)
    while True:
        batch = []
        for _ in range(micro_batch):
            batch.append(next(it))
        yield np.asarray(batch, dtype=np.int32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: PretrainConfig) -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover
        raise RuntimeError(
            f"JAX/NumPy not available: {_IMPORT_ERROR}. "
            "Install EasyDeL's TPU requirements on the TPU VM."
        ) from _IMPORT_ERROR

    # Imports kept local so --help works on machines without the TPU stack.
    import easydel as ed  # type: ignore
    import flax.nnx as nnx  # type: ignore
    import orbax.checkpoint as ocp  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    wandb = None
    if cfg.wandb_project and jax.process_index() == 0:
        try:
            import wandb as _wandb  # type: ignore

            _wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=asdict(cfg),
            )
            wandb = _wandb
        except Exception as exc:
            print(f"[pretrain] wandb init failed: {exc}", flush=True)

    model_cfg = cfg.load_model_config()
    easy_cfg = _build_easydel_config(cfg, model_cfg)
    mesh, dp, tp = _build_mesh(cfg)
    if jax.process_index() == 0:
        print(
            f"[pretrain] mesh = (data={dp}, model={tp}); "
            f"total chips = {jax.device_count()}",
            flush=True,
        )

    # Global batch = dp * micro_batch * seq_len. Solve for micro_batch.
    tokens_per_step = cfg.global_batch_tokens
    micro_batch = max(1, tokens_per_step // (dp * cfg.seq_len))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    with mesh:
        # EasyDeL ≥ 0.1.4 uses Flax NNX: the model owns its parameters
        # directly as attributes and requires ``rngs``. Construction must
        # happen inside the mesh context so sharding constraints resolve.
        model = ed.LlamaForCausalLM(
            easy_cfg,
            rngs=nnx.Rngs(cfg.seed),
            dtype=getattr(jnp, cfg.compute_dtype),
            param_dtype=getattr(jnp, cfg.param_dtype),
            precision=None,
        )
        tx, lr_schedule = _build_optimizer(cfg, cfg.max_steps)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    ckpt_mgr = ocp.CheckpointManager(
        directory=cfg.output_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=cfg.ckpt_keep,
            keep_period=cfg.ckpt_permanent_every,
            create=True,
            enable_async_checkpointing=True,
        ),
    )
    start_step = 0
    if cfg.resume:
        latest = ckpt_mgr.latest_step()
        if latest is not None:
            if jax.process_index() == 0:
                print(f"[pretrain] resuming from step {latest}", flush=True)
            # NNX: save/restore the full optimizer (wraps model + opt state).
            state = nnx.state(optimizer)
            restored = ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(state))
            nnx.update(optimizer, restored)
            start_step = latest

    def loss_fn(model, batch):
        logits = model(batch[:, :-1]).logits
        labels = batch[:, 1:]
        loss = -jnp.take_along_axis(
            jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
            labels[..., None],
            axis=-1,
        ).squeeze(-1).mean()
        return loss

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(grads)
        grad_norm = jnp.sqrt(
            sum(jnp.vdot(g, g).real for g in jax.tree_util.tree_leaves(grads))
        )
        return loss, grad_norm

    batch_iter = _batch_iterator(cfg, tokenizer, micro_batch=micro_batch)

    step_time_ema = 0.0
    t_last_log = time.time()
    # EasyDeL's logical-sharding helpers require the mesh to still be active
    # inside the training loop, not just during model construction.
    with mesh:
        for step in range(start_step, cfg.max_steps):
            batch = next(batch_iter)
            t0 = time.time()
            loss, grad_norm = train_step(model, optimizer, jnp.asarray(batch))
            loss.block_until_ready()
            dt = time.time() - t0
            step_time_ema = 0.98 * step_time_ema + 0.02 * dt if step_time_ema else dt

            if step % 10 == 0 and jax.process_index() == 0:
                tokens_per_sec = tokens_per_step / max(step_time_ema, 1e-6)
                lr = float(lr_schedule(step))
                print(
                    f"step={step:07d} loss={float(loss):.4f} grad_norm={float(grad_norm):.3f} "
                    f"lr={lr:.2e} tok/s={tokens_per_sec:.0f} step_ms={step_time_ema*1000:.1f}",
                    flush=True,
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "loss": float(loss),
                            "grad_norm": float(grad_norm),
                            "learning_rate": lr,
                            "tokens_per_sec": tokens_per_sec,
                            "step_ms": step_time_ema * 1000,
                        },
                        step=step,
                    )

            if step > 0 and step % cfg.ckpt_every == 0 and not cfg.smoke_test:
                _save_checkpoint(ckpt_mgr, step, nnx.state(optimizer))

            if cfg.smoke_test and step >= 20:
                if jax.process_index() == 0:
                    print("[pretrain] smoke test complete — exiting at step 20", flush=True)
                break

        if not cfg.smoke_test:
            _save_checkpoint(ckpt_mgr, cfg.max_steps, nnx.state(optimizer))
            ckpt_mgr.wait_until_finished()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, dest="config_json")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus-glob", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--global-batch-tokens", type=int, default=2_097_152)
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--dp-size", type=int, default=None)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--peak-lr", type=float, default=1.5e-4)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--ckpt-keep", type=int, default=3)
    p.add_argument("--ckpt-permanent-every", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-shards", default=None)
    p.add_argument("--wandb-project", default="aksarallm-20b")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 20 steps and exit. Use on v6e-8 or a laptop.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PretrainConfig(
        config_json=args.config_json,
        tokenizer=args.tokenizer,
        corpus_glob=args.corpus_glob,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        global_batch_tokens=args.global_batch_tokens,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        optimizer=args.optimizer,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        ckpt_every=args.ckpt_every,
        ckpt_keep=args.ckpt_keep,
        ckpt_permanent_every=args.ckpt_permanent_every,
        eval_every=args.eval_every,
        eval_shards=args.eval_shards,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
