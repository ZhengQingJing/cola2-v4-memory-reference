"""Cola2 v4: shared S, independent personal M, directional and codec losses.

Training uses arrived observations. Public-only readouts precede private scoring.
All neural imports and operations are within the approved server entry point.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase1_common import ModelBudget, RunLedger, approved, sha, write_json

REVISION = "db876b905cf719b60da33c12a75021ac111f7a67"
SOURCE_FILES = ("config.json", "configuration_cola2.py", "modeling_cola2.py", "modeling_layers.py")
ARMS = ("mse_shared", "direction_shared", "direction_relation_shared", "direction_relation_native")
PROFILES = ("G00", "G01", "G10", "G11")
SEED, MEMORY_SEED, EVAL_SEEDS, K, EOS, TOTAL_SECONDS = 311121, 311122, (311121, 311122), 24, 151643, 7500

class CapacityFailure(ValueError):
    pass

def cue_text(row):
    return f"Task: {row['task']}\nAction already carried out: {row['action']}\nUser's observed feedback:"

def noise_seed(cue_id, base_seed):
    return int.from_bytes(hashlib.sha256(f"{base_seed}\n{cue_id}".encode()).digest()[:8], "big") % (2**63 - 1)

def read_public(path, count):
    rows = json.loads(path.read_text("utf-8"))["cues"]
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError("Unexpected public cue count or duplicate id")
    if any(set(row) != {"id", "split", "task", "action"} for row in rows):
        raise ValueError("Public rows must contain only id/split/task/action")
    return rows

def read_states():
    return [("base_M0", None, None, 0, False), ("shared_M0", None, None, 0, True)] + [
        (f"{arm}_{stream}_{dose}", arm, stream, dose, arm != "direction_relation_native")
        for arm in ARMS for stream in ("A", "B") for dose in (24, 32)]

def memory_path(root, arm, stream, dose):
    return root / f"personal_{arm}_{stream}" / f"dose_{dose}" / "memory.pt"

def verify_public_read_grid(read_output):
    rows = [json.loads(line) for line in (read_output / "events.jsonl").read_text("utf-8").splitlines()]
    answers = [r for r in rows if r.get("event") == "answer" and r.get("mode") == "read"]
    keys = {(r["state"], r["cue_id"], r["eval_seed"]) for r in answers}
    if len(answers) != 520 or len(keys) != 520 or any(r.get("source_visible") is not False for r in answers):
        raise ValueError("Expected 520 unique source-free public read answers")
    all_ids = {r["cue_id"] for r in answers if r["state"] == "base_M0" and r["eval_seed"] == EVAL_SEEDS[0]}
    dose24_state = next(state for state, _, _, dose, _ in read_states() if dose == 24)
    dev_ids = {r["cue_id"] for r in answers if r["state"] == dose24_state and r["eval_seed"] == EVAL_SEEDS[0]}
    if len(all_ids) != 18 or len(dev_ids) != 10 or not dev_ids <= all_ids:
        raise ValueError("Public read must contain ten development plus eight confirmation cue IDs")
    expected = {(state, cue, seed) for state, _, _, dose, _ in read_states()
                for cue in (dev_ids if dose == 24 else all_ids) for seed in EVAL_SEEDS}
    if keys != expected:
        raise ValueError("Public state/seed/cue coverage differs from the complete preregistered grid")
    return dict(answers=520, unique_keys=520, development_cues=10, confirmation_cues=8)

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("probe", "shared", "personal", "shared_read", "read", "score"), required=True)
    for name in ("model", "output", "budget"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("shared-data", "experiences", "public-cues", "confirmation-public", "shared-public-cues",
                 "shared-checkpoint", "training-memories", "run-root", "evaluation", "confirmation-private", "read-output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--source-code", type=Path, default=ROOT / "research" / "cola2_20261111" / REVISION)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--stream", choices=("A", "B"))
    p.add_argument("--shared-updates", type=int, default=160)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()

def start(args):
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, approved(value))
    caps = dict(probe=300, shared=3000, personal=600, shared_read=240, read=1900, score=180)
    if not args.device.startswith("cuda") or not 0 < args.max_seconds <= caps[args.mode]:
        raise ValueError("Authorized CUDA and the preregistered mode cap are required")
    requirements = {
        "probe": {"shared_data"}, "shared": {"shared_data"},
        "personal": {"experiences", "arm", "stream", "shared_checkpoint"},
        "shared_read": {"shared_public_cues", "shared_checkpoint", "training_memories"},
        "read": {"public_cues", "confirmation_public", "shared_checkpoint", "run_root"},
        "score": {"evaluation", "confirmation_private", "read_output"}}
    possible = set().union(*requirements.values())
    present = {name for name in possible if getattr(args, name) is not None}
    if present != requirements[args.mode]:
        raise ValueError(f"{args.mode} requires exactly {sorted(requirements[args.mode])}; got {sorted(present)}")
    if args.shared_updates <= 0 or args.shared_updates % 40:
        raise ValueError("Shared dose must be fixed complete passes over all 40 observations")
    if args.mode == "score":
        completion = json.loads((args.read_output / "completion.json").read_text("utf-8"))
        if completion["status"] != "complete" or completion["actual"].get("answers") != 520:
            raise ValueError("Complete all 520 public readouts before opening private targets")
        public_grid = verify_public_read_grid(args.read_output)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
                 "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "TMP", "TEMP"):
        if name not in os.environ:
            raise ValueError("Missing root-scoped cache/temp variable: " + name)
        approved(os.environ[name])
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Local pinned loading requires HF_HUB_OFFLINE=1")
    source_sha = {name: sha(args.source_code / name) for name in SOURCE_FILES}
    if any(sha(args.model / name) != value for name, value in source_sha.items()):
        raise ValueError("Pinned source/config mismatch")
    inputs = dict(model=str(args.model), revision=REVISION, official_source_sha256=source_sha)
    for name in present - {"arm", "stream", "run_root", "training_memories", "read_output"}:
        path = getattr(args, name)
        inputs[name] = dict(path=str(path), sha256=sha(path))
    if args.mode == "read":
        inputs["M_checkpoints"] = {state: sha(approved(memory_path(args.run_root, arm, stream, dose)))
            for state, arm, stream, dose, _ in read_states()[2:]}
    if args.mode == "shared_read":
        inputs["training_M_checkpoints"] = {profile: sha(approved(args.training_memories / f"{profile}.pt")) for profile in PROFILES}
    if args.mode == "score":
        inputs["read_output"] = str(args.read_output)
        inputs["read_events_sha256"] = sha(args.read_output / "events.jsonl")
        inputs["public_grid_gate"] = public_grid
    budget = ModelBudget(args.budget, limit_seconds=TOTAL_SECONDS)
    if budget.expired(reserve=15):
        raise TimeoutError("Cumulative v4 budget exhausted")
    ledger = RunLedger(args.output, "user_memory_cola_v4_" + args.mode,
        dict(shared_budget=str(args.budget), cumulative_limit_seconds=TOTAL_SECONDS,
             process_max_seconds=args.max_seconds, remaining_seconds=budget.remaining()), inputs=inputs,
        code_paths=[Path(__file__), Path(__file__).with_name("phase1_common.py"), *(args.source_code / n for n in SOURCE_FILES)])
    return budget, ledger, source_sha

def main():
    args = arguments()
    budget, ledger, source_sha = start(args)
    begun = time.perf_counter()
    torch = model = tokenizer = None
    S, M, S0, M0, base, adapters = {}, {}, {}, {}, {}, {}
    handles, details = [], {}
    status, error, active = "failed", None, "initializing"
    peaks = dict(allocated=0, reserved=0)

    def check(*_):
        if time.perf_counter() - begun >= args.max_seconds or budget.expired(reserve=10):
            raise TimeoutError("Fixed process/cumulative v4 budget reached")
    def deadline(*_):
        raise TimeoutError("Fixed process/cumulative v4 budget reached")
    def record(row):
        row["elapsed_seconds"] = time.perf_counter() - begun
        if torch is not None and torch.cuda.is_available():
            peaks["allocated"] = max(peaks["allocated"], torch.cuda.max_memory_allocated(args.device))
            peaks["reserved"] = max(peaks["reserved"], torch.cuda.max_memory_reserved(args.device))
        ledger.record(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, min(args.max_seconds, budget.remaining() - 10))
        os.environ["USE_HUB_KERNELS"] = "NO"
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Neural operations require the authorized CUDA server")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint
        import transformers
        from transformers import AutoModel, AutoTokenizer
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
        active = "load_model"
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True, use_fast=True)
        model = AutoModel.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)
        if len(model.dit.blocks) != 34 or model.config.latent_dim != 64 or model.config.eos_token_id != EOS:
            raise ValueError("Unexpected Cola2 architecture")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("Base must remain BF16")
        base = {n: (p, id(p), p._version) for n, p in model.named_parameters()}

        def count_dit(*_):
            check()
            ledger.add(dit_nfe=1)
        handles.append(model.dit.register_forward_pre_hook(count_dit))

        def digest(values):
            h = hashlib.sha256()
            for name, value in sorted(values.items()):
                h.update(name.encode())
                h.update(value.detach().float().contiguous().cpu().numpy().tobytes())
            return h.hexdigest()

        def install():
            sg = torch.Generator(device=device).manual_seed(SEED)
            mg = torch.Generator(device=device).manual_seed(MEMORY_SEED)
            for i, block in enumerate(model.dit.blocks):
                for projection in ("proj_qkv", "proj_out"):
                    linear = getattr(block.msa, projection)
                    for tag, bank, rng in (("shared", S, sg), ("memory", M, mg)):
                        a = torch.nn.Parameter(torch.empty(8, linear.in_features, dtype=torch.float32, device=device))
                        b = torch.nn.Parameter(torch.zeros(linear.out_features, 8, dtype=torch.float32, device=device))
                        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=rng)
                        linear.register_parameter(tag + "_A", a)
                        linear.register_parameter(tag + "_B", b)
                        for part, value in (("A", a), ("B", b)):
                            key = f"dit.blocks.{i}.msa.{projection}.{part}"
                            bank[key] = value
                            adapters[tag + ":" + key] = value
                    def forward(this, hidden):
                        out = F.linear(hidden, this.weight, this.bias)
                        sh = F.linear(F.linear(hidden.float(), this.shared_A), this.shared_B)
                        me = F.linear(F.linear(hidden.float(), this.memory_A), this.memory_B)
                        return out + sh.to(out.dtype) + me.to(out.dtype)
                    linear.forward = types.MethodType(forward, linear)
            def wrap(block, label):
                original = block.forward
                def forward(this, *inputs):
                    calls = 0
                    def counted(*values):
                        nonlocal calls
                        check()
                        ledger.add(**{label + "_evaluation_entries": 1})
                        if calls:
                            ledger.add(**{label + "_checkpoint_recompute_entries": 1})
                        calls += 1
                        return original(*values)
                    needed = torch.is_grad_enabled() and (label == "dit_block" or any(
                        isinstance(x, torch.Tensor) and x.requires_grad for x in inputs))
                    return checkpoint(counted, *inputs, use_reentrant=False) if needed else counted(*inputs)
                block.forward = types.MethodType(forward, block)
            for block in model.dit.blocks:
                wrap(block, "dit_block")
            for block in list(model.vae.decoder_cond_stack) + list(model.vae.decoder["blocks"]):
                wrap(block, "codec_block")
        install()
        S0 = {n: p.detach().cpu().clone() for n, p in S.items()}
        M0 = {n: p.detach().cpu().clone() for n, p in M.items()}

        def copy_bank(bank, values):
            if set(bank) != set(values):
                raise ValueError("Adapter tensor names mismatch")
            with torch.no_grad():
                for name, value in values.items():
                    bank[name].copy_(value)

        def trainable(shared, memory):
            for p in S.values():
                p.requires_grad_(shared)
                p.grad = None
            for p in M.values():
                p.requires_grad_(memory)
                p.grad = None
            model.eval()  # Frozen codec remains eval even while input autograd is enabled.

        def reset():
            copy_bank(S, S0)
            copy_bank(M, M0)
            trainable(False, False)
            record(dict(event="reset_S0M0", S_sha256=digest(S), M_sha256=digest(M)))

        def load_shared():
            payload = torch.load(args.shared_checkpoint, map_location="cpu", weights_only=True)
            meta = payload["manifest"]
            if set(payload) != {"manifest", "S"} or meta["schema"] != "cola2_v4_shared_only" or meta["source_sha256"] != source_sha:
                raise ValueError("Expected S-only checkpoint with no M or optimizer")
            copy_bank(S, payload["S"])
            if digest(S) != meta["S_sha256"]:
                raise ValueError("Shared checksum mismatch")
            record(dict(event="shared_checkpoint_loaded",shared_training_updates=meta["updates"],S_sha256=meta["S_sha256"]))
            return meta

        def load_memory(path, expected=None):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            meta = payload["manifest"]
            if set(payload) != {"manifest", "M"} or meta["schema"] not in ("cola2_v4_personal", "cola2_v4_training_memory"):
                raise ValueError("Expected M-only checkpoint")
            if meta["source_sha256"] != source_sha or meta["S_sha256"] != digest(S):
                raise ValueError("Memory source/S identity mismatch")
            if expected and (meta["arm"], meta["stream"], meta["dose"]) != expected:
                raise ValueError("Personal checkpoint identity mismatch")
            copy_bank(M, payload["M"])
            if digest(M) != meta["M_sha256"]:
                raise ValueError("Memory checksum mismatch")

        reset()
        record(dict(event="loaded", pid=os.getpid(), torch_version=torch.__version__, transformers_version=transformers.__version__,
            gpu=torch.cuda.get_device_name(device), parameters_per_bank=sum(p.numel() for p in M.values()),
            S0_sha256=digest(S), M0_sha256=digest(M), weight_footprint_bytes=model.get_memory_footprint()))

        def fingerprint(tensor):
            return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), fp32_sha256=hashlib.sha256(
                tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest())

        def verify_base():
            current = dict(model.named_parameters())
            bad = [name for name, (p, identity, version) in base.items() if id(current.get(name)) != identity
                   or p._version != version or p.requires_grad or p.grad is not None]
            if bad:
                raise RuntimeError("Frozen base changed: " + ",".join(bad[:8]))
            return True

        def ids(text):
            return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

        def encode_dynamic(tokens):
            check()
            with torch.no_grad():
                out = model.vae.encode_dynamic(tokens[None], torch.ones_like(tokens[None], dtype=torch.bool))
            count = int(out.latent_mask[0].sum())
            ledger.add(codec_encodes=1, codec_input_tokens=tokens.numel())
            return out, count

        def encode_prefix(row):
            tokens = ids(cue_text(row))
            out, count = encode_dynamic(tokens)
            return dict(ids=tokens, lengths=out.span_lengths[0, :count].detach(),
                        scaled=model._scale_latents(out.latents[0, :count].detach().float()))

        def encode_target(row, feedback, allow_overflow=False):
            prefix = encode_prefix(row)
            body_ids = ids(" " + feedback)
            body_out, body_count = encode_dynamic(body_ids)
            body_lengths = body_out.span_lengths[0, :body_count].detach()
            overflow = body_count + 1 > K
            if overflow and not allow_overflow:
                raise CapacityFailure(f"{row['id']}: {body_count} body latents plus EOS exceed fixed {K}")
            padding = max(0, K - body_count - 1)
            eos_count = padding + 1
            target_ids = torch.cat((body_ids, torch.full((eos_count,), EOS, device=device, dtype=torch.long)))
            target_lengths = torch.cat((body_lengths, torch.ones(eos_count, device=device, dtype=torch.long)))
            with torch.no_grad():
                contextual = model.vae.encode_given_spans(torch.cat((prefix["ids"], target_ids)),
                                                          torch.cat((prefix["lengths"], target_lengths)))
            ledger.add(codec_encodes=1, codec_input_tokens=len(prefix["ids"]) + len(target_ids))
            target = model._scale_latents(contextual[len(prefix["scaled"]):].detach().float())
            return dict(prefix=prefix, target=target, target_lengths=target_lengths, target_ids=target_ids,
                body_latents=body_count, body_tokens=len(body_ids), real_eos_position=body_count,
                structural_padding_latents=padding, capacity_overflow=overflow, feedback=feedback)

        def sample_latents(public_prefix, generator, trace=False):
            """No target, target length, or token feedback can enter this function."""
            origin = len(public_prefix)
            history = public_prefix
            blocks, noise_hashes = 0, []
            schedule = torch.linspace(1., 0., 9, device=device, dtype=torch.float32).square()
            while len(history) - origin < K:
                check()
                remainder = len(history) % 8
                older = history[:-remainder] if remainder else history
                clean = history[-remainder:] if remainder else history[:0]
                noise = torch.randn((8 - remainder, 64), generator=generator, device=device, dtype=torch.float32)
                if trace:
                    noise_hashes.append(fingerprint(noise))
                state = torch.cat((clean, noise))
                for current, following in zip(schedule[:-1], schedule[1:], strict=True):
                    times = torch.cat((torch.zeros(len(older) + remainder, device=device),
                        torch.full((8 - remainder,), float(current) * model.config.diffusion_time, device=device)))
                    drift = model.dit(torch.cat((older, state)).to(torch.bfloat16), times.to(torch.bfloat16))[-8:].float()
                    updated = state - (current - following) * drift
                    # Functional clamp: preserve the complete across-step graph without in-place writes.
                    state = torch.cat((clean, updated[remainder:]))
                history = torch.cat((older, state))
                blocks += 1
            result = history[origin:origin + K]
            return result, dict(blocks=blocks, sampling_nfe=blocks * 8,
                sampled_active_latents=len(history) - origin, returned_active_latents=K,
                alignment_auxiliary_latents=len(history) - origin - K, initial_remainder=origin % 8,
                auxiliary_slots_enter_decoder=False, latent_history_only=True, token_feedback_reencoding=False,
                noise_fingerprints=noise_hashes)

        def decode_latents(prefix, recalled, true_lengths=None):
            """Decode all fixed recalled latents; only generated EOS and the public cap stop output."""
            with torch.no_grad():
                native = model._unscale_latents(torch.cat((prefix["scaled"], recalled))).to(model.vae.latent_in_proj.weight.dtype)
                _, duration_logits = model.vae.latent_decode(native)
                predicted = duration_logits[len(prefix["scaled"]):].argmax(-1).long() + 1
                lengths = predicted if true_lengths is None else true_lengths
                condition, _ = model.vae.expand_and_refine(native, torch.cat((prefix["lengths"], lengths)))
                target_condition = condition[len(prefix["ids"]):]
                committed = prefix["ids"]
                visible, offset, chunks, query_tokens = [], 0, 0, 0
                stop = "latent_budget"
                for length_tensor in lengths:
                    check()
                    width = int(length_tensor)
                    logits = model.vae.span_query_logits(committed, target_condition[offset:offset + width], conditional=True)
                    proposed = logits.argmax(-1).tolist()
                    chunks += 1
                    query_tokens += width
                    offset += width
                    used = []
                    for token in proposed:
                        if len(visible) >= 64:
                            stop = "cap"
                            break
                        visible.append(int(token))
                        used.append(int(token))
                        if int(token) == EOS:
                            stop = "eos"
                            break
                    committed = torch.cat((committed, torch.tensor(used, device=device, dtype=torch.long)))
                    if stop in ("cap", "eos") or len(visible) >= 64:
                        if stop != "eos":
                            stop = "cap"
                        break
                ledger.add(decoder_span_calls=chunks, decoder_query_tokens=query_tokens)
            return dict(raw_output=tokenizer.decode(visible, skip_special_tokens=False),
                output=tokenizer.decode(visible, skip_special_tokens=True), output_ids=visible,
                new_tokens=len(visible), stop_reason=stop, eos_seen=EOS in visible, cap_reached=len(visible) >= 64,
                predicted_duration_lengths=predicted.tolist(), used_duration_lengths=lengths.tolist(),
                duration_source="predicted" if true_lengths is None else "privileged_true_spans", decoder_cfg=0.,
                decoder_span_calls=chunks, decoder_query_tokens=query_tokens, true_feedback_tokens_in_prefix=False)

        def answer(row, state, base_seed, event="answer"):
            nonlocal active
            active = f"{event}:{state}:{row['id']}:{base_seed}"
            check()
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            begin, old_nfe = time.perf_counter(), ledger.counts.get("dit_nfe", 0)
            prefix = encode_prefix(row)
            derived_seed = noise_seed(row["id"], base_seed)
            with torch.no_grad():
                recalled, sampling = sample_latents(prefix["scaled"],
                    torch.Generator(device=device).manual_seed(derived_seed), trace=True)
            path = args.output / "generated_latents" / f"{state}_{row['id']}_{base_seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(schema="cola2_v4_generated_latents", state=state, cue_id=row["id"], eval_seed=base_seed,
                derived_seed=derived_seed, prefix=fingerprint(prefix["scaled"]), sampling=sampling,
                source_visible=False, target_padding_unknown=True, latents=recalled.detach().cpu()), path)
            decoded = decode_latents(prefix, recalled)
            record(dict(event=event, mode=args.mode, state=state, cue_id=row["id"], split=row.get("split"),
                public_prompt=cue_text(row), S_sha256=digest(S), M_sha256=digest(M), eval_seed=base_seed, derived_seed=derived_seed, source_visible=False,
                controlled_variant="CFG1 latent-history-only fixed24, decoderCFG0", sampling=sampling,
                sample_artifact=dict(path=str(path), sha256=sha(path)), seconds=time.perf_counter() - begin,
                dit_nfe=ledger.counts.get("dit_nfe", 0) - old_nfe,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device), **decoded))
            ledger.add(**({"answers": 1} if event == "answer" else {event + "_answers": 1}), generated_tokens=decoded["new_tokens"])

        def weights(target):
            positions = torch.arange(K, device=device)
            return torch.where(positions < target["body_latents"], 1., torch.where(
                positions == target["body_latents"], .2, .02))

        def endpoint(recalled, target):
            goal = target["target"]
            mse = (recalled - goal).square().mean(-1)
            cosine = F.cosine_similarity(recalled, goal, dim=-1, eps=1e-6)
            radius = recalled.square().mean(-1).add(1e-12).sqrt()
            goal_radius = goal.square().mean(-1).add(1e-12).sqrt()
            direction = 2 * (1 - cosine) + .1 * (radius - goal_radius).square()
            w = weights(target)
            regions = {}
            positions = torch.arange(K, device=device)
            for name, mask in (("body", positions < target["body_latents"]), ("real_eos", positions == target["body_latents"]),
                               ("padding", positions > target["body_latents"])):
                regions[name] = dict(positions=int(mask.sum()), mse=float(mse[mask].detach().mean()) if bool(mask.any()) else None,
                    cosine=float(cosine[mask].detach().mean()) if bool(mask.any()) else None,
                    generated_rms=float(radius[mask].detach().mean()) if bool(mask.any()) else None,
                    target_rms=float(goal_radius[mask].detach().mean()) if bool(mask.any()) else None)
            return (w * mse).sum() / w.sum(), (w * direction).sum() / w.sum(), regions

        def weighted_fm(target, rng):
            prefix = target["prefix"]["scaled"]
            sequence = torch.cat((prefix, target["target"]))
            starts = list(range(len(prefix) // 8 * 8, len(sequence), 8))
            start = starts[int(torch.randint(len(starts), (), device=device, generator=rng))]
            clean = sequence[start:start + 8]
            r = max(0, len(prefix) - start)
            drop = bool(torch.rand((), device=device, generator=rng) < .2)
            older = sequence[:0] if drop else sequence[:start]
            t = float(.02 + .96 * torch.rand((), device=device, generator=rng))
            noise = torch.randn(clean.shape, device=device, generator=rng, dtype=torch.float32)
            block = torch.cat((clean[:r], ((1-t) * clean + t * noise)[r:]))
            times = torch.cat((torch.zeros(len(older) + r, device=device),
                               torch.full((len(clean)-r,), t * model.config.diffusion_time, device=device)))
            drift = model.dit(torch.cat((older, block)).to(torch.bfloat16), times.to(torch.bfloat16))[-len(clean):].float()
            error = (drift[r:] - (noise-clean)[r:]).square().mean(-1)
            positions = torch.arange(start+r-len(prefix), start+len(clean)-len(prefix), device=device)
            w = weights(target)
            loss = len(starts) * (w[positions] * error).sum() / w.sum()
            return loss, dict(t=t, block_start=start, clean_remainder=r, earlier_prefix_dropped=drop,
                sampled_blocks=len(starts), global_weight_sum=float(w.sum()), teacher_previous_feedback_visible=True,
                block_weight_sum=float(w[positions].sum()), weighted_loss=float(loss.detach()))

        def relation_losses(recalled, target):
            # True lengths are training supervision coordinates, never public generation inputs.
            prefix = target["prefix"]
            native = model._unscale_latents(torch.cat((prefix["scaled"], recalled))).to(model.vae.latent_in_proj.weight.dtype)
            condition, duration_logits = model.vae.expand_and_refine(native, torch.cat((prefix["lengths"], target["target_lengths"])))
            query = condition[len(prefix["ids"]):]
            observed_spans = target["body_latents"] + 1
            observed_tokens = target["body_tokens"] + 1
            committed, offset, terms = prefix["ids"], 0, []
            for length in target["target_lengths"][:observed_spans]:
                check()
                width = int(length)
                logits = model.vae.span_query_logits(committed, query[offset:offset+width], conditional=True)
                terms.append(F.cross_entropy(logits.float(), target["target_ids"][offset:offset+width], reduction="sum"))
                # Only this invocation's generated words enter the next prefix; no true feedback words.
                committed = torch.cat((committed, logits.detach().argmax(-1)))
                offset += width
                ledger.add(training_codec_span_calls=1, training_codec_query_tokens=width)
            if offset != observed_tokens:
                raise RuntimeError("Observed body/EOS span coordinates mismatch")
            nll = torch.stack(terms).sum() / observed_tokens
            duration = F.cross_entropy(duration_logits[len(prefix["scaled"]):len(prefix["scaled"])+observed_spans].float(),
                                       target["target_lengths"][:observed_spans]-1)
            return nll, duration

        def compute_losses(target, objective, phase, presentation, episode_id):
            sample_seed = noise_seed(f"sample/{phase}/{presentation}/{episode_id}", SEED)
            fm_seed = noise_seed(f"fm/{phase}/{presentation}/{episode_id}", SEED)
            recalled, sampling = sample_latents(target["prefix"]["scaled"], torch.Generator(device=device).manual_seed(sample_seed))
            if not recalled.requires_grad:
                raise RuntimeError("Sampled latent graph disconnected")
            mse, direction, regions = endpoint(recalled, target)
            fm, fm_info = weighted_fm(target, torch.Generator(device=device).manual_seed(fm_seed))
            relation = objective == "direction_relation"
            nll, duration = relation_losses(recalled, target) if relation else (None, None)
            total = (mse if objective == "mse" else direction) + .1 * fm
            if relation:
                total = total + .2 * nll + .05 * duration
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("Nonfinite combined loss")
            info = dict(total_loss=float(total.detach()), weighted_mse=float(mse.detach()), weighted_direction=float(direction.detach()),
                weighted_fm=float(fm.detach()), observed_nll=float(nll.detach()) if relation else None,
                observed_duration_ce=float(duration.detach()) if relation else None, regions=regions, fm_details=fm_info,
                sample_seed=sample_seed, fm_seed=fm_seed, sampling=sampling,
                training_true_spans_visible=relation, true_feedback_words_in_prefix=False,
                discrete_generated_prefix_detached=relation, early_generated_eos_does_not_stop_supervision=relation,
                body_tokens=target["body_tokens"], body_latents=target["body_latents"], structural_padding=target["structural_padding_latents"])
            return total, direction, nll, duration, recalled, info

        def gradient_summary(grads):
            valid = [g for g in grads if g is not None]
            return dict(tensors=len(valid), nonzero_tensors=sum(bool(g.detach().count_nonzero()) for g in valid),
                norm=math.sqrt(sum(float(g.detach().float().square().sum()) for g in valid)),
                finite=all(bool(torch.isfinite(g).all()) for g in valid))

        def update(target, objective, phase, step, episode_id, optimizers, shared_active, context):
            nonlocal active
            active = f"update:{phase}:{step}:{episode_id}"
            check()
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            s_before, m_before = digest(S), digest(M)
            begin, nfe0 = time.perf_counter(), ledger.counts.get("dit_nfe", 0)
            loss, _, _, _, _, info = compute_losses(target, objective, phase, step, episode_id)
            loss.backward()
            active_parameters = list(S.values()) + list(M.values()) if shared_active else list(M.values())
            norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient")
            for opt in optimizers:
                opt.step()
            s_after, m_after = digest(S), digest(M)
            if not shared_active and s_before != s_after:
                raise RuntimeError("Personal update changed frozen S")
            verify_base()
            ledger.add(optimizer_updates=1, S_updates=int(shared_active), M_updates=1)
            record(dict(event="update", phase=phase, objective=objective, presentation=step, episode_id=episode_id,
                S_sha256_before=s_before, S_sha256_after=s_after, M_sha256_before=m_before, M_sha256_after=m_after,
                S_changed=s_before != s_after, M_changed=m_before != m_after, S_frozen=not shared_active,
                gradient_norm_before_clip=float(norm), seconds=time.perf_counter()-begin,
                dit_nfe=ledger.counts.get("dit_nfe", 0)-nfe0, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device), **context, **info))

        def observe(ep, label):
            answer(ep, label, EVAL_SEEDS[0], event="pre_observation")
            record(dict(event="arrived_observation", label=label, episode_id=ep["id"], task=ep["task"], action=ep["action"],
                feedback=ep["feedback"], turns=ep["turns"], provenance="Preregistered synthetic observation; no generated teacher"))
            target = encode_target(ep, ep["feedback"])
            record(dict(event="target_encoded", label=label, episode_id=ep["id"], body_tokens=target["body_tokens"],
                body_latents=target["body_latents"], structural_padding=target["structural_padding_latents"],
                prefix_tokens=len(target["prefix"]["ids"]), prefix_latents=len(target["prefix"]["scaled"])))
            return target

        def save_M(path, meta):
            meta.update(M_sha256=digest(M), S_sha256=digest(S), source_sha256=source_sha, K=K)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(manifest=meta, M={n: p.detach().cpu() for n,p in M.items()}), path)
            write_json(path.with_suffix(".json"), meta)

        def run_probe():
            data = json.loads(args.shared_data.read_text("utf-8"))["streams"]
            ep = data[0]["episodes"][0]
            reset()
            target = observe(ep, "probe_S0M0")
            trainable(True, True)
            opts = [torch.optim.AdamW(list(bank.values()), lr=2e-4, weight_decay=0.) for bank in (S,M)]
            before_s, before_m = digest(S), digest(M)
            loss, direction, nll, duration, recalled, info = compute_losses(target,"direction_relation","probe",1,ep["id"])
            gn = torch.autograd.grad(nll, list(M.values()), retain_graph=True, allow_unused=True)
            gd = torch.autograd.grad(direction, list(M.values()), retain_graph=True, allow_unused=True)
            gu = torch.autograd.grad(duration, recalled, retain_graph=True)[0]
            summary_n, summary_d = gradient_summary(gn), gradient_summary(gd)
            record(dict(event="probe_loss_gradients", NLL_to_M=summary_n, DIR_to_M=summary_d,
                duration_to_recalled=gradient_summary([gu]), **info))
            if not summary_n["finite"] or not summary_d["finite"] or not summary_n["nonzero_tensors"] or not summary_d["nonzero_tensors"]:
                raise RuntimeError("Required NLL/DIR to M gradient missing or nonfinite")
            del gn, gd, gu
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(S.values())+list(M.values()),1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Probe combined gradient nonfinite")
            for opt in opts:
                opt.step()
            ledger.add(optimizer_updates=1,S_updates=1,M_updates=1)
            changed_s,changed_m=before_s!=digest(S),before_m!=digest(M)
            record(dict(event="probe_joint_update", S_changed=changed_s, M_changed=changed_m,
                gradient_norm_before_clip=float(norm), base_frozen_verified=verify_base()))
            if not changed_s or not changed_m:
                raise RuntimeError("Probe joint optimizer did not change both S and M")
            del loss, direction, nll, duration, recalled
            trainable(False, True)
            fixed_s, fixed_m = digest(S), digest(M)
            torch.cuda.reset_peak_memory_stats(device)
            personal_begin,personal_nfe=time.perf_counter(),ledger.counts.get("dit_nfe",0)
            total, _, _, _, _, info = compute_losses(target,"direction_relation","probe_frozen_S",1,ep["id"])
            total.backward()
            record(dict(event="probe_personal_backward_only", optimizer_step=False, S_unchanged=fixed_s==digest(S),
                M_parameters_unchanged=fixed_m==digest(M), S_has_gradient=any(p.grad is not None for p in S.values()),
                M_gradient=gradient_summary([p.grad for p in M.values()]), base_frozen_verified=verify_base(),
                seconds=time.perf_counter()-personal_begin,dit_nfe=ledger.counts.get("dit_nfe",0)-personal_nfe,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),**info))
            if fixed_s!=digest(S) or any(p.grad is not None for p in S.values()):
                raise RuntimeError("Frozen S received a personal gradient/update")
            details["probe_parameters_discarded"] = True

        def run_shared():
            streams = json.loads(args.shared_data.read_text("utf-8"))["streams"]
            by_id = {s["id"]:s["episodes"] for s in streams}
            if set(by_id)!=set(PROFILES) or any(len(x)!=10 for x in by_id.values()):
                raise ValueError("Expected four shared profiles, ten observations each")
            if any([e["id"] for e in by_id[p]] != [e["id"] for e in by_id[PROFILES[0]]] for p in PROFILES):
                raise ValueError("Shared profile episode order mismatch")
            reset(); trainable(True,True)
            memories = {p:{n:t.clone() for n,t in M0.items()} for p in PROFILES}
            optS = torch.optim.AdamW(list(S.values()),lr=2e-4,weight_decay=0.)
            optM = {p:torch.optim.AdamW(list(M.values()),lr=2e-4,weight_decay=0.) for p in PROFILES}
            seen, visits, cache = set(), {p:0 for p in PROFILES}, {}
            try:
                for step in range(args.shared_updates):
                    profile = PROFILES[step % 4]
                    ep = by_id[profile][(step//4)%10]
                    copy_bank(M, memories[profile])
                    trainable(True,True)
                    pair=(profile,ep["id"])
                    if pair not in cache:
                        cache[pair]=observe(ep,f"shared_{profile}_before_{visits[profile]+1}")
                        seen.add(pair)
                    visits[profile]+=1
                    update(cache[pair],"direction_relation","shared",visits[profile],ep["id"],[optS,optM[profile]],True,
                        dict(global_update=step+1,profile=profile,unique_observations=len(seen),replays=step+1-len(seen)))
                    memories[profile]={n:p.detach().cpu().clone() for n,p in M.items()}
                meta=dict(schema="cola2_v4_shared_only",updates=args.shared_updates,source_sha256=source_sha,
                    S_sha256=digest(S),shared_data_sha256=sha(args.shared_data),profile_ids=list(PROFILES),
                    memory_included=False,optimizer_included=False,seed=SEED,K=K)
                torch.save(dict(manifest=meta,S={n:p.detach().cpu() for n,p in S.items()}),args.output/"shared_only.pt")
                write_json(args.output/"shared_manifest.json",meta)
                torch.save(optS.state_dict(),args.output/"shared_optimizer.pt")
                for profile in PROFILES:
                    copy_bank(M,memories[profile])
                    save_M(args.output/"training_memories"/f"{profile}.pt",dict(schema="cola2_v4_training_memory",profile=profile,
                        presentations=visits[profile],training_diagnostic_only=True))
                    torch.save(optM[profile].state_dict(),args.output/"training_memories"/f"{profile}.optimizer.pt")
            finally:
                if status != "complete" and not (args.output/"shared_only.pt").exists():
                    torch.save(dict(S={n:p.detach().cpu() for n,p in S.items()},memories=memories,
                        completed_updates=ledger.counts.get("optimizer_updates",0)),args.output/"partial_shared_training.pt")
                    torch.save(dict(S=optS.state_dict(),M={p:opt.state_dict() for p,opt in optM.items()}),
                               args.output/"partial_shared_optimizers.pt")

        def run_personal():
            data={s["id"]:s["episodes"] for s in json.loads(args.experiences.read_text("utf-8"))["streams"]}
            episodes=data[args.stream]
            if [e["stage"] for e in episodes] != ["acquisition"]*4+["new_domain"]:
                raise ValueError("Expected four personal acquisition observations and one new event")
            reset()
            shared_meta=load_shared()  # Always verify the common S checkpoint, including native control.
            if args.arm=="direction_relation_native":
                copy_bank(S,S0)
            trainable(False,True)
            fixed_s=digest(S)
            objective="mse" if args.arm=="mse_shared" else "direction" if args.arm=="direction_shared" else "direction_relation"
            opt=torch.optim.AdamW(list(M.values()),lr=2e-4,weight_decay=0.)
            cache,visits={},{}
            try:
                for step in range(1,33):
                    ep=episodes[(step-1)%4] if step<=24 else episodes[4]
                    if ep["id"] not in cache:
                        cache[ep["id"]]=observe(ep,f"{args.arm}_{args.stream}_before_{step}")
                    visits[ep["id"]]=visits.get(ep["id"],0)+1
                    update(cache[ep["id"]],objective,"personal",step,ep["id"],[opt],False,
                        dict(arm=args.arm,stream=args.stream,unique_observations=len(cache),replays=step-len(cache),visits=dict(visits)))
                    if digest(S)!=fixed_s:
                        raise RuntimeError("Personal S changed")
                    if step in (24,32):
                        path=args.output/f"dose_{step}"/"memory.pt"
                        save_M(path,dict(schema="cola2_v4_personal",arm=args.arm,stream=args.stream,dose=step,
                            shared_checkpoint_sha256=sha(args.shared_checkpoint),shared_trained_S_sha256=shared_meta["S_sha256"],
                            shared_training_updates=shared_meta["updates"],
                            uses_shared=args.arm!="direction_relation_native",experience_sha256=sha(args.experiences),
                            M0_sha256=digest(M0),unique_observations=len(cache),visits=dict(visits)))
                        torch.save(opt.state_dict(),path.parent/"optimizer.pt")
                        record(dict(event="personal_saved",arm=args.arm,stream=args.stream,dose=step,S_sha256=fixed_s,M_sha256=digest(M)))
            finally:
                if not (args.output/"dose_32"/"memory.pt").exists():
                    torch.save(dict(M={n:p.detach().cpu() for n,p in M.items()},S_sha256=fixed_s,
                        updates=ledger.counts.get("optimizer_updates",0)),args.output/"partial_personal.pt")
                    torch.save(opt.state_dict(),args.output/"partial_personal_optimizer.pt")

        def run_read(shared_training=False):
            if shared_training:
                dev=read_public(args.shared_public_cues,10)
                specs=[("shared_M0",None,None,0,True)]+[(f"shared_{p}",None,p,0,True) for p in PROFILES]
                confirmation=[]; seeds=EVAL_SEEDS[:1]
            else:
                dev=read_public(args.public_cues,10); confirmation=read_public(args.confirmation_public,8)
                if len({r["id"] for r in dev+confirmation})!=18:
                    raise ValueError("Public cue ids must be disjoint")
                specs=read_states(); seeds=EVAL_SEEDS
            for state,arm,stream,dose,use_shared in specs:
                reset()
                if use_shared: load_shared()
                if shared_training and stream is not None:
                    load_memory(args.training_memories/f"{stream}.pt")
                elif arm is not None:
                    load_memory(memory_path(args.run_root,arm,stream,dose),(arm,stream,dose))
                trainable(False,False)
                before_s,before_m=digest(S),digest(M)
                record(dict(event="read_state",state=state,S_sha256=before_s,M_sha256=before_m,uses_shared=use_shared,
                    shared_training_diagnostic=shared_training))
                cues=dev if shared_training or dose==24 else dev+confirmation
                for row in cues:
                    for seed in seeds: answer(row,state,seed)
                if before_s!=digest(S) or before_m!=digest(M):
                    raise RuntimeError("Public read changed S/M")
            expected=50 if shared_training else 520
            if ledger.counts.get("answers")!=expected:
                raise RuntimeError("Fixed read grid incomplete")

        def run_score():
            # start() already checked the complete 520-answer gate before any private file access.
            dev=json.loads(args.evaluation.read_text("utf-8"))["cases"]
            confirmation=json.loads(args.confirmation_private.read_text("utf-8"))["cases"]
            cases=dev+confirmation
            if len(dev)!=10 or len(confirmation)!=8 or len({c["id"] for c in cases})!=18:
                raise ValueError("Expected ten development and eight confirmation scoring cases")
            records=[json.loads(line) for line in (args.read_output/"events.jsonl").read_text("utf-8").splitlines()]
            reads={(r["state"],r["cue_id"],r["eval_seed"]):r for r in records if r.get("event")=="answer" and r.get("mode")=="read"}
            expected={(state,c["id"],seed) for state,arm,stream,dose,useS in read_states()
                      for c in (dev if dose==24 else cases) for seed in EVAL_SEEDS}
            if len(reads)!=520 or set(reads)!=expected:
                raise ValueError("Formal public read grid mismatch")
            trainable(False,False)
            cache,results,computed={},{},{}
            result_rows=[]
            try:
                for case in cases:
                    texts=[case["feedback_by_stream"]["A"],case["feedback_by_stream"]["B"]]
                    same=texts[0]==texts[1]
                    if same: texts[1]=case["foil_feedback"]
                    if texts[0]==texts[1]: raise ValueError("Counterfactual must differ")
                    cache[case["id"]]=dict(texts=texts,correct={"A":0,"B":0 if same else 1},
                        targets=[encode_target(case,t,allow_overflow=True) for t in texts])
                for state,arm,stream,dose,useS in read_states():
                    for case in (dev if dose==24 else cases):
                        entry=cache[case["id"]]
                        for seed in EVAL_SEEDS:
                            check()
                            read=reads[(state,case["id"],seed)]
                            if read["public_prompt"]!=cue_text(case) or read["derived_seed"]!=noise_seed(case["id"],seed):
                                raise ValueError("Scoring public cue/seed mismatch")
                            path=approved(read["sample_artifact"]["path"])
                            if not path.is_relative_to(args.read_output) or sha(path)!=read["sample_artifact"]["sha256"]:
                                raise ValueError("Saved sample identity mismatch")
                            payload=torch.load(path,map_location="cpu",weights_only=True)
                            if (payload["state"],payload["cue_id"],payload["eval_seed"])!=(state,case["id"],seed):
                                raise ValueError("Saved sample state/cue/seed mismatch")
                            generated=payload["latents"].to(device)
                            if tuple(generated.shape)!=(K,64): raise ValueError("Expected all fixed24 generated latents")
                            common=min(K,*(t["body_latents"] for t in entry["targets"]))
                            metrics=[]
                            for target in entry["targets"]:
                                if payload["prefix"]!=fingerprint(target["prefix"]["scaled"]):
                                    raise ValueError("Scoring changed public prefix")
                                n=min(K,len(target["target"]))
                                x,y=generated[:n],target["target"][:n]
                                sq=(x-y).square().mean(-1)
                                cosine=F.cosine_similarity(x,y,dim=-1,eps=1e-6)
                                ex,ey=x.square().mean(-1),y.square().mean(-1)
                                dot=(x*y).mean(-1)
                                pos=torch.arange(n,device=device)
                                regions={}
                                for label,mask in (("body",pos<target["body_latents"]),("real_eos",pos==target["body_latents"]),
                                                   ("padding",pos>target["body_latents"])):
                                    regions[label]=dict(positions=int(mask.sum()),mse=float(sq[mask].mean()) if bool(mask.any()) else None,
                                        cosine=float(cosine[mask].mean()) if bool(mask.any()) else None,
                                        generated_energy=float(ex[mask].mean()) if bool(mask.any()) else None,
                                        target_energy=float(ey[mask].mean()) if bool(mask.any()) else None,
                                        target_dot=float(dot[mask].mean()) if bool(mask.any()) else None)
                                metrics.append(dict(regions=regions,body_common_prefix_mse=float(sq[:common].mean()),
                                    body_common_prefix_cosine=float(cosine[:common].mean()),body_latents=target["body_latents"],
                                    capacity_overflow=target["capacity_overflow"],full_target_covered=not target["capacity_overflow"],
                                    structural_padding=target["structural_padding_latents"]))
                            computed[(state,case["id"],seed)]=metrics
                            baseline_state="base_M0" if state=="base_M0" or arm=="direction_relation_native" else "shared_M0"
                            reference=computed[(baseline_state,case["id"],seed)]
                            deltas={}
                            margins={}
                            for user,index in entry["correct"].items():
                                now,prior=metrics[index]["regions"]["body"],reference[index]["regions"]["body"]
                                improvement=prior["mse"]-now["mse"]
                                contraction=prior["generated_energy"]-now["generated_energy"]
                                dot_change=2*(now["target_dot"]-prior["target_dot"])
                                deltas[user]=dict(body_mse_improvement=improvement,body_cosine_change=now["cosine"]-prior["cosine"],
                                    energy_contraction=contraction,twice_target_dot_change=dot_change,
                                    identity_roundoff_residual=improvement-contraction-dot_change)
                                margins[user]=metrics[1-index]["body_common_prefix_mse"]-metrics[index]["body_common_prefix_mse"]
                            row=dict(event="privileged_score",state=state,arm=arm,stream=stream,dose=dose,baseline_state=baseline_state,
                                cue_id=case["id"],split=read["split"],eval_seed=seed,private_kind=case.get("kind"),
                                requires_stream_specific_memory=case.get("requires_stream_specific_memory"),source_visible=True,
                                candidate_feedback=entry["texts"],correct_candidate_by_stream=entry["correct"],candidate_metrics=metrics,
                                common_body_positions=common,common_body_margins=margins,delta_from_own_initialization=deltas,
                                raw_generated_feedback=read["raw_output"],output_ids=read["output_ids"],
                                padding_counts_as_memory_success=False,semantic_success_claimed=False,
                                interpretation="Post-read diagnostics only; complete observed relations require manual semantic assessment")
                            result_rows.append(row); record(row); ledger.add(scored_readouts=1)
            finally:
                write_json(args.output/"score_summary.json",dict(complete=len(result_rows)==520,scored_readouts=len(result_rows),
                    planned_readouts=520,answers_modified=False,selected_checkpoints=False,results=result_rows))

        protocol=dict(mode=args.mode,version="v4",K=K,arms=list(ARMS),shared_updates=args.shared_updates,
            planned_shared_passes=args.shared_updates//40,planned_shared_updates_in_this_mode=args.shared_updates if args.mode=="shared" else 0,
            personal_updates=32,personal_acquisition_updates=24,personal_new_event_updates=8,
            S_seed=SEED,M_seed=MEMORY_SEED,eval_seeds=list(EVAL_SEEDS),rank=8,alpha=8,
            optimizer="AdamW lr=2e-4 weight_decay=0 clip=1",endpoint_weights=dict(body=1.,real_eos=.2,padding=.02),
            direction="2*(1-cosine)+0.1*(RMS_pred-RMS_target)^2",FM_coefficient=.1,NLL_coefficient=.2,duration_CE_coefficient=.05,
            FM_normalization="Full24 weight sum times sampled block count; no per-block renormalization",
            training_rng="seed/phase/per-stream-presentation/episode-id; no arm, profile, stream or target; separate sample and FM generators",
            shared_schedule="Episode-major interleaving G00,G01,G10,G11; shared S evolves sequentially",
            codec_frozen=True,codec_eval=True,codec_checkpoint_on_input_gradient=True,
            actual_feedback_words_in_decoder_prefix=False,training_true_spans_visible=True,
            discrete_self_argmax_prefix_detached=True,early_EOS_does_not_stop_training_loss=True,
            padding_in_NLL=False,padding_in_duration_CE=False,public_predicted_spans=True,
            sampler="CFG1 8-step low-noise Euler fixed24 latent-history-only",decoder="CFG0 greedy EOS151643/cap64/fixed24",
            primary_read_answers=520,shared_training_diagnostic_answers=50,private_scoring_after_complete_read=True,
            best_of=False,answer_repair=False)
        write_json(args.output/"protocol.json",protocol)
        if args.mode=="probe": run_probe()
        elif args.mode=="shared": run_shared()
        elif args.mode=="personal": run_personal()
        elif args.mode=="shared_read": run_read(True)
        elif args.mode=="read": run_read(False)
        else: run_score()
        details.update(base_frozen_verified=verify_base(),final_S_sha256=digest(S),final_M_sha256=digest(M))
        status="complete"
    except BaseException as exc:
        status="capacity_failed" if isinstance(exc,CapacityFailure) else "budget_exhausted" if isinstance(exc,TimeoutError) else "failed"
        error=traceback.format_exc()
        record(dict(event="failure",active_call=active,error=error,exception_type=type(exc).__name__))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None and base:
            current=dict(model.named_parameters())
            changed=[n for n,(p,identity,version) in base.items() if id(current.get(n))!=identity or p._version!=version or p.requires_grad or p.grad is not None]
            details.update(base_frozen_verified=not changed,changed_base_parameter_names=changed)
            del current
        for handle in handles: handle.remove()
        for bank in (S,M,S0,M0,base,adapters): bank.clear()
        model=tokenizer=None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            peaks["allocated"]=max(peaks["allocated"],torch.cuda.max_memory_allocated(args.device))
            peaks["reserved"]=max(peaks["reserved"],torch.cuda.max_memory_reserved(args.device))
            torch.cuda.empty_cache()
            details["allocated_bytes_after_release"]=torch.cuda.memory_allocated(args.device)
        budget.close("user_memory_cola_v4_"+args.mode,args.output)
        details["completed_formal_shared_updates_this_run"]=ledger.counts.get("optimizer_updates",0) if args.mode=="shared" else 0
        ledger.finish(status,error=error,active_call=active,peak_allocated_bytes=peaks["allocated"],peak_reserved_bytes=peaks["reserved"],
            cumulative_budget_used_seconds=budget.state["used_seconds"],**details)
    if status!="complete": raise SystemExit(2)

if __name__=="__main__": main()
