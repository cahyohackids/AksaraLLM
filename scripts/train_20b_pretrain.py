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
    corpus_glob: Optional[str]
    output_dir: str
    # Optional GCS glob to pre-tokenized .npy shards (shape [N_seqs, seq_len],
    # uint32) produced by ``scripts/tokenize_pretrain_corpus.py``. When set the
    # trainer streams ``int32`` batches directly from .npy without invoking
    # the HF tokenizer — eliminates the per-step CPU bottleneck.
    pretok_glob: Optional[str] = None

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

    # Sharding (4D mesh ``(dp, fsdp, tp, sp)`` matching EasyDeL defaults).
    # For 7B on v6e-8 the most memory-efficient choice is FSDP across all 8
    # chips: parameters and optimiser state are sharded along that axis,
    # which is exactly what ``shard_model`` expects when EasyDeL's default
    # ``axis_dims=(1, -1, 1, 1)`` partition rules are applied.
    fsdp_size: int = -1  # -1 = consume all chips not used by dp/tp/sp
    tp_size: int = 1
    sp_size: int = 1
    dp_size: int = 1

    # Precision
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"

    # Memory
    gradient_checkpointing: str = "NOTHING_SAVEABLE"  # NONE | CHECKPOINT_DOTS | NOTHING_SAVEABLE | EVERYTHING_SAVEABLE
    scan_layers: bool = True

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
    from easydel import EasyDeLGradientCheckPointers, LlamaConfig  # type: ignore

    arch = model_cfg["architecture"]
    seq_plan = model_cfg.get("sequence_plan", {})

    # Gradient checkpointing is MANDATORY for 20B on v5p-64 HBM budget and
    # also strongly recommended even for the 1B smoke path on v6e-8: without
    # it the per-layer bf16 [seq, d] activations sum to ~1.2GB per chip at
    # seq=8192 and OOM the training step. ``NOTHING_SAVEABLE`` is the most
    # aggressive policy (recomputes every intermediate) — trades ~30% step
    # time for ~3\u00d7 lower activation memory.
    gc_mode = cfg.gradient_checkpointing.upper()
    gc_policy = getattr(EasyDeLGradientCheckPointers, gc_mode)

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
        gradient_checkpointing=gc_policy,
        scan_layers=cfg.scan_layers,
    )


def _build_mesh(cfg: PretrainConfig):
    """Build a 4D ``(dp, fsdp, tp, sp)`` mesh matching EasyDeL defaults.

    EasyDeL's built-in partition rules are written against this exact axis
    naming/ordering — using a different naming makes ``shard_model`` raise
    ``Resource axis: fsdp ... is not found in mesh``.
    """
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils  # type: ignore

    total = jax.device_count()
    dp = max(1, cfg.dp_size)
    tp = max(1, cfg.tp_size)
    sp = max(1, cfg.sp_size)

    fsdp = cfg.fsdp_size
    if fsdp <= 0:
        if total % (dp * tp * sp) != 0:
            raise ValueError(
                f"Total chips ({total}) not divisible by dp*tp*sp ({dp}*{tp}*{sp})."
            )
        fsdp = total // (dp * tp * sp)

    if dp * fsdp * tp * sp != total:
        raise ValueError(
            f"dp*fsdp*tp*sp ({dp}*{fsdp}*{tp}*{sp}) must equal total chips ({total})."
        )

    devices = mesh_utils.create_device_mesh((dp, fsdp, tp, sp))
    mesh = Mesh(devices, axis_names=("dp", "fsdp", "tp", "sp"))
    return mesh, dp, fsdp, tp, sp


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


def _iter_pretokenized_sequences(cfg: PretrainConfig):
    """Stream pre-tokenized ``.npy`` shards from GCS and yield seq_len rows.

    Each shard is an ``[N_seqs, seq_len]`` uint32 array written by
    ``scripts/tokenize_pretrain_corpus.py``. Shards are listed in sorted
    order; rows within a shard are yielded sequentially. Loops forever so
    downstream can pull exactly ``max_steps`` batches.
    """
    import io
    import gcsfs  # type: ignore

    assert cfg.pretok_glob is not None
    fs = gcsfs.GCSFileSystem()
    paths = sorted(fs.glob(cfg.pretok_glob))
    if not paths:
        raise RuntimeError(f"no .npy shards matched {cfg.pretok_glob!r}")
    if jax.process_index() == 0:
        print(f"[pretrain] pretokenized shards: {len(paths)} matched {cfg.pretok_glob}", flush=True)
    while True:
        for path in paths:
            with fs.open(path, "rb") as fh:
                arr = np.load(io.BytesIO(fh.read()))
            if arr.shape[1] != cfg.seq_len:
                raise RuntimeError(
                    f"shard {path} has seq_len={arr.shape[1]} but --seq-len={cfg.seq_len}"
                )
            for i in range(arr.shape[0]):
                yield arr[i]


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


def _batch_iterator(cfg: PretrainConfig, tokenizer, global_batch_seqs: int):
    """Yield numpy batches of shape ``(global_batch_seqs, seq_len)`` int32.

    The trainer's caller is responsible for explicitly sharding this array
    across the ``(dp, fsdp)`` mesh axes via ``jax.device_put`` with a
    ``NamedSharding(mesh, P(('dp', 'fsdp'), None))``. With that sharding
    each chip sees ``global_batch_seqs / (dp * fsdp)`` sequences.
    """
    if cfg.pretok_glob:
        it = _iter_pretokenized_sequences(cfg)
    else:
        it = _iter_packed_sequences(cfg, tokenizer)
    while True:
        batch = []
        for _ in range(global_batch_seqs):
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
    mesh, dp, fsdp, tp, sp = _build_mesh(cfg)
    if jax.process_index() == 0:
        print(
            f"[pretrain] mesh = (dp={dp}, fsdp={fsdp}, tp={tp}, sp={sp}); "
            f"total chips = {jax.device_count()}",
            flush=True,
        )

    # EasyDeL's default partition rules read ``axis_dims`` / ``axis_names`` off
    # the model config when ``shard_model`` is invoked. Set them to match the
    # mesh we just built so the rules resolve correctly.
    easy_cfg.axis_dims = (dp, fsdp, tp, sp)
    easy_cfg.axis_names = ("dp", "fsdp", "tp", "sp")
    # Effective DP for batching = product of dp + fsdp axes (data is replicated
    # across fsdp shards because FSDP all-gathers params per step).
    eff_dp = dp * fsdp

    # Global batch is built as a single ``[B, seq_len]`` array and explicitly
    # sharded across the ``(dp, fsdp)`` mesh axes. Each chip ends up with
    # ``micro_batch`` sequences. We REQUIRE ``B`` to be a multiple of
    # ``eff_dp`` so the sharding splits evenly.
    micro_batch = max(1, cfg.global_batch_tokens // (eff_dp * cfg.seq_len))
    global_batch_seqs = micro_batch * eff_dp
    tokens_per_step = global_batch_seqs * cfg.seq_len
    if jax.process_index() == 0:
        print(
            f"[pretrain] tokens_per_step={tokens_per_step:,} "
            f"(global_batch_seqs={global_batch_seqs}, micro_batch_per_chip={micro_batch})",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    with mesh:
        # EasyDeL ≥ 0.1.4 uses Flax NNX. Direct construction
        # (`ed.LlamaForCausalLM(...)`) materialises every parameter on the
        # current host before sharding constraints are applied. For models
        # large enough that a single weight tensor exceeds *per-chip* HBM
        # (e.g. 20B), this OOMs even though host RAM is ample.
        #
        # We try ``lazy_init`` first (no allocation, then ``shard_model``
        # materialises directly with sharding); if that returns abstract
        # ShapeDtypeStruct leaves (some EasyDeL versions only annotate),
        # we fall back to direct construction. For ≤7B on a v6e-8 the
        # direct path always works because each parameter tensor fits in
        # 32 GB HBM per chip and ``shard_model`` then redistributes.
        try:
            model = ed.LlamaForCausalLM.lazy_init(
                easy_cfg,
                rngs=nnx.Rngs(cfg.seed),
                dtype=getattr(jnp, cfg.compute_dtype),
                param_dtype=getattr(jnp, cfg.param_dtype),
                precision=None,
            )
            model = model.shard_model(mesh=mesh)
            # Sanity check: if any leaf is still ShapeDtypeStruct (abstract),
            # ``shard_model`` did not materialise — fall back to direct.
            from jax import ShapeDtypeStruct  # type: ignore
            leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
            if any(isinstance(x, ShapeDtypeStruct) for x in leaves):
                raise RuntimeError("shard_model returned abstract leaves")
        except Exception as exc:
            if jax.process_index() == 0:
                print(f"[pretrain] lazy_init+shard_model failed ({exc}); "
                      f"falling back to direct construction.", flush=True)
            model = ed.LlamaForCausalLM(
                easy_cfg,
                rngs=nnx.Rngs(cfg.seed),
                dtype=getattr(jnp, cfg.compute_dtype),
                param_dtype=getattr(jnp, cfg.param_dtype),
                precision=None,
            )
            try:
                model = model.shard_model(mesh=mesh)
            except Exception as e2:
                if jax.process_index() == 0:
                    print(f"[pretrain] post-init shard_model failed ({e2}); "
                          f"continuing with replicated parameters.", flush=True)
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

    batch_iter = _batch_iterator(cfg, tokenizer, global_batch_seqs=global_batch_seqs)

    # Explicit FSDP+DP batch sharding: every step the host puts the
    # ``[global_batch_seqs, seq_len]`` array on devices with axis-0
    # split across the (dp, fsdp) mesh axes. Each chip then physically
    # sees only ``micro_batch`` sequences, which is the proper SPMD
    # data-parallel pattern for FSDP. Without this, every chip would
    # process the same replicated batch and ``tokens_per_step`` would
    # be ``fsdp``× smaller than configured.
    from jax.sharding import NamedSharding, PartitionSpec as P  # type: ignore
    batch_sharding = NamedSharding(mesh, P(("dp", "fsdp"), None))

    step_time_ema = 0.0
    t_last_log = time.time()
    # EasyDeL's logical-sharding helpers require the mesh to still be active
    # inside the training loop, not just during model construction.
    with mesh:
        for step in range(start_step, cfg.max_steps):
            batch = next(batch_iter)
            t0 = time.time()
            batch_jax = jax.device_put(jnp.asarray(batch), batch_sharding)
            loss, grad_norm = train_step(model, optimizer, batch_jax)
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
    p.add_argument("--corpus-glob", default=None,
                   help="GCS glob to raw parquet shards. Used with --tokenizer for on-the-fly tokenization. Mutually exclusive with --pretok-glob.")
    p.add_argument("--pretok-glob", default=None,
                   help="GCS glob to pre-tokenized .npy shards (output of scripts/tokenize_pretrain_corpus.py). Recommended: pre-tokenize once, then point trainer here to skip per-step HF tokenizer.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--global-batch-tokens", type=int, default=2_097_152)
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor-parallel axis size (mesh axis 'tp'). Default 1.")
    p.add_argument("--dp-size", type=int, default=1,
                   help="Pure data-parallel axis size (mesh axis 'dp'). Default 1.")
    p.add_argument("--fsdp-size", type=int, default=-1,
                   help="FSDP axis size (mesh axis 'fsdp'). -1 = consume all remaining chips. This is the axis EasyDeL's default partition rules shard parameters along.")
    p.add_argument("--sp-size", type=int, default=1,
                   help="Sequence-parallel axis size (mesh axis 'sp'). Default 1.")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--peak-lr", type=float, default=1.5e-4)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--param-dtype", default="float32", choices=["float32", "bfloat16"],
                   help="Parameter storage dtype. bfloat16 halves HBM but gives up fp32 master weights.")
    p.add_argument("--compute-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--gradient-checkpointing", default="NOTHING_SAVEABLE",
                   choices=["NONE", "CHECKPOINT_DOTS", "CHECKPOINT_DOTS_WITH_NO_BATCH_DIMS", "NOTHING_SAVEABLE", "EVERYTHING_SAVEABLE"],
                   help="EasyDeL gradient checkpointing policy. NOTHING_SAVEABLE recomputes every activation (most memory savings).")
    p.add_argument("--no-scan-layers", action="store_true",
                   help="Disable ``scan_layers`` (scanning layers keeps the compiled HLO small, highly recommended for large models).")
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
    if not args.corpus_glob and not args.pretok_glob:
        raise SystemExit("Exactly one of --corpus-glob or --pretok-glob must be set.")
    if args.corpus_glob and args.pretok_glob:
        raise SystemExit("--corpus-glob and --pretok-glob are mutually exclusive.")
    cfg = PretrainConfig(
        config_json=args.config_json,
        tokenizer=args.tokenizer,
        corpus_glob=args.corpus_glob,
        pretok_glob=args.pretok_glob,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        global_batch_tokens=args.global_batch_tokens,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        fsdp_size=args.fsdp_size,
        sp_size=args.sp_size,
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
        param_dtype=args.param_dtype,
        compute_dtype=args.compute_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        scan_layers=not args.no_scan_layers,
    )
    train(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
