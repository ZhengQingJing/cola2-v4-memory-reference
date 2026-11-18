"""Preregistered, source-visible span diagnostic after all v4 formal results.

Decode 16 learned latent sequences with observed spans, eight true-latent
references, and eight baseline-latent/span views from four fixed baseline
samples. No S/M checkpoint, DiT forward, optimizer or new latent sample is used.
Original formal answers remain unchanged beside the privileged diagnostics.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.phase1_common import ModelBudget, RunLedger, approved, sha, write_json

REVISION = "db876b905cf719b60da33c12a75021ac111f7a67"
SOURCE_FILES = ("config.json", "configuration_cola2.py", "modeling_cola2.py", "modeling_layers.py")
ARMS = ("mse_shared", "direction_shared", "direction_relation_shared", "direction_relation_native")
EVAL_SEEDS, K, EOS = (311121, 311122), 24, 151643

class CapacityFailure(ValueError):
    pass

def cue_text(row):
    return f"Task: {row['task']}\nAction already carried out: {row['action']}\nUser's observed feedback:"

def noise_seed(cue_id, base_seed):
    return int.from_bytes(hashlib.sha256(f"{base_seed}\n{cue_id}".encode()).digest()[:8], "big") % (2**63 - 1)

def read_states():
    return [("base_M0", None, None, 0, False), ("shared_M0", None, None, 0, True)] + [
        (f"{arm}_{stream}_{dose}", arm, stream, dose, arm != "direction_relation_native")
        for arm in ARMS for stream in ("A", "B") for dose in (24, 32)]

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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output", "budget", "experiences", "read-output", "score-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--limit-seconds", type=int, default=7500)
    parser.add_argument("--source-code", type=Path, default=ROOT / "research" / "cola2_20261111" / REVISION)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()

def formal_gate(args):
    for directory, phase, count_key in ((args.read_output, "read", "answers"),
                                        (args.score_output, "score", "scored_readouts")):
        done = json.loads((directory / "completion.json").read_text("utf-8"))
        if (done["phase"] != "user_memory_cola_v4_" + phase or done["status"] != "complete"
                or done["actual"].get(count_key) != 520):
            raise ValueError("Formal read and score must both complete all 520 cases first")
    verify_public_read_grid(args.read_output)
    read_rows = [json.loads(line) for line in (args.read_output / "events.jsonl").read_text("utf-8").splitlines()]
    reads = [r for r in read_rows if r.get("event") == "answer" and r.get("mode") == "read"]
    read_keys = {(r["state"], r["cue_id"], r["eval_seed"]) for r in reads}
    score_rows = [json.loads(line) for line in (args.score_output / "events.jsonl").read_text("utf-8").splitlines()]
    scores = [r for r in score_rows if r.get("event") == "privileged_score"]
    score_keys = {(r["state"], r["cue_id"], r["eval_seed"]) for r in scores}
    if len(scores) != 520 or len(score_keys) != 520 or score_keys != read_keys:
        raise ValueError("Unique formal read/score grids do not match")
    score_job = json.loads((args.score_output / "job.json").read_text("utf-8"))
    if score_job["inputs"]["read_events_sha256"] != sha(args.read_output / "events.jsonl"):
        raise ValueError("Formal read changed after scoring")
    return {(r["state"], r["cue_id"], r["eval_seed"]): r for r in reads}

def start(args):
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, approved(value))
    if not args.device.startswith("cuda") or not 0 < args.max_seconds <= 180 or not 0 < args.limit_seconds <= 7500:
        raise ValueError("Require authorized CUDA, cap <=180 seconds and cumulative limit <=7500")
    if not args.budget.exists():
        raise ValueError("Reuse the existing explicitly updated model budget; do not start a fresh budget")
    previous = json.loads(args.budget.read_text("utf-8"))
    if previous["limit_seconds"] != args.limit_seconds:
        raise ValueError("Primary agent must explicitly update the existing limit while preserving prior usage")
    # Source observations are not opened or hashed until both completed grids pass.
    reads = formal_gate(args)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "XDG_CACHE_HOME", "TORCH_HOME",
                 "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "TMP", "TEMP"):
        if name not in os.environ:
            raise ValueError("Missing root-scoped cache/temp variable: " + name)
        approved(os.environ[name])
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Pinned local loading requires HF_HUB_OFFLINE=1")
    source_sha = {name: sha(args.source_code / name) for name in SOURCE_FILES}
    if any(sha(args.model / name) != expected for name, expected in source_sha.items()):
        raise ValueError("Official model/source mismatch")
    read_job = json.loads((args.read_output / "job.json").read_text("utf-8"))
    if read_job["inputs"]["official_source_sha256"] != source_sha:
        raise ValueError("Diagnostic codec source differs from the formal read")
    budget = ModelBudget(args.budget, limit_seconds=args.limit_seconds)
    if budget.expired(reserve=15):
        raise TimeoutError("Existing cumulative budget exhausted")
    ledger = RunLedger(args.output, "user_memory_cola_v4_span_diagnostic",
        dict(shared_budget=str(args.budget), cumulative_limit_seconds=args.limit_seconds,
             process_max_seconds=args.max_seconds, prior_used_seconds=previous["used_seconds"],
             remaining_seconds=budget.remaining()),
        inputs=dict(model=str(args.model), official_source_sha256=source_sha,
            source_experiences_sha256=sha(args.experiences), read_events_sha256=sha(args.read_output / "events.jsonl"),
            score_events_sha256=sha(args.score_output / "events.jsonl"),
            complete_read_keys=520, complete_score_keys=520, original_outputs_unchanged=True),
        code_paths=[Path(__file__), Path(__file__).with_name("phase1_common.py"), *(args.source_code / n for n in SOURCE_FILES)])
    return budget, ledger, reads

def main():
    args = arguments()
    budget, ledger, reads = start(args)
    begun = time.perf_counter()
    torch = model = tokenizer = forbidden_handle = None
    base, targets, results = {}, {}, []
    baseline_samples_seen = set()
    status, error, active = "failed", None, "initializing"
    peaks = dict(allocated=0, reserved=0)
    details = {}

    def check(*_):
        if time.perf_counter() - begun >= args.max_seconds or budget.expired(reserve=10):
            raise TimeoutError("Fixed span-diagnostic or cumulative budget reached")
    def deadline(*_):
        raise TimeoutError("Fixed span-diagnostic or cumulative budget reached")
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
            raise RuntimeError("Codec inference requires the authorized CUDA server")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        from transformers import AutoModel, AutoTokenizer
        torch.cuda.reset_peak_memory_stats(device)
        active = "load_frozen_official_model"
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True, use_fast=True)
        model = AutoModel.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)
        if model.config.latent_dim != 64 or model.config.eos_token_id != EOS:
            raise ValueError("Unexpected official codec")
        base = {name: (p, id(p), p._version) for name, p in model.named_parameters()}
        def forbid_dit(*_):
            ledger.add(forbidden_DiT_calls=1)
            raise RuntimeError("This diagnostic must not call DiT or sample new memory")
        forbidden_handle = model.dit.register_forward_pre_hook(forbid_dit)
        ledger.add(optimizer_updates=0, new_latent_samples=0, loaded_S_M_checkpoints=0, forbidden_DiT_calls=0)
        record(dict(event="loaded_frozen_codec", pid=os.getpid(), weight_footprint_bytes=model.get_memory_footprint(),
                    source_visible_diagnostic=True, S_M_loaded=False, DiT_sampling=False))

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

        data = json.loads(args.experiences.read_text("utf-8"))["streams"]
        sources = {stream["id"]: {episode["id"]: episode for episode in stream["episodes"]} for stream in data}
        if set(sources) != {"A", "B"}:
            raise ValueError("Expected the two frozen personal experience streams")
        write_json(args.output / "protocol.json", dict(
            mode="source_visible_span_diagnostic", planned_outputs=32, original_24_outputs_retained=True,
            generated_latent_grid=dict(arms=list(ARMS), streams=["A", "B"], dose=32, cue_ids=["s1", "n1"], eval_seed=311121),
            source_reference_grid=dict(streams=["A", "B"], cue_ids=["s1", "n1"], durations=["predicted", "true"]),
            baseline_grid=dict(states=["base_M0", "shared_M0"], cue_ids=["s1", "n1"], eval_seed=311121,
                target_span_streams=["A", "B"], unique_saved_samples=4, auxiliary_outputs=8),
            no_new_sampling=True, no_training=True, no_S_M_loaded=True, source_visible=True,
            actual_feedback_words_in_decoder_prefix=False, full_fixed24_latents=True,
            decoder_cfg=0., eos=EOS, max_new_tokens=64, original_answers_modified=False,
            formal_success_claimed=False, selecting_examples_or_seeds=False,
            interpretation="True spans expose observation geometry; any gain is an auxiliary decoding upper bound, not independent recall"))
        for stream in ("A", "B"):
            for cue_id in ("s1", "n1"):
                active = f"encode_source:{stream}:{cue_id}"
                episode = sources[stream][cue_id]
                targets[(stream, cue_id)] = encode_target(episode, episode["feedback"])

        for arm in ARMS:
            for stream in ("A", "B"):
                state = f"{arm}_{stream}_32"
                for cue_id in ("s1", "n1"):
                    active = f"saved_latent_true_span:{state}:{cue_id}"
                    check()
                    episode, target = sources[stream][cue_id], targets[(stream, cue_id)]
                    original = reads[(state, cue_id, 311121)]
                    if original["public_prompt"] != cue_text(episode) or original["derived_seed"] != noise_seed(cue_id, 311121):
                        raise ValueError("Formal/source public cue or seed mismatch")
                    path = approved(original["sample_artifact"]["path"])
                    if not path.is_relative_to(args.read_output) or sha(path) != original["sample_artifact"]["sha256"]:
                        raise ValueError("Saved latent artifact path/hash mismatch")
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    if (payload["schema"] != "cola2_v4_generated_latents"
                            or (payload["state"], payload["cue_id"], payload["eval_seed"]) != (state, cue_id, 311121)
                            or payload["derived_seed"] != noise_seed(cue_id, 311121)
                            or payload["source_visible"] is not False
                            or payload["prefix"] != fingerprint(target["prefix"]["scaled"])):
                        raise ValueError("Saved latent identity or independent prefix mismatch")
                    recalled = payload["latents"].to(device)
                    if tuple(recalled.shape) != (K, 64):
                        raise ValueError("Expected all 24 original generated latents")
                    start_time = time.perf_counter()
                    decoded = decode_latents(target["prefix"], recalled, target["target_lengths"])
                    row = dict(event="saved_latent_true_span", arm=arm, stream=stream, dose=32, state=state, cue_id=cue_id,
                        eval_seed=311121, source_visible=True, true_spans_visible=True, true_target_latents_used=False,
                        original_formal_raw_output=original["raw_output"], original_formal_output_ids=original["output_ids"],
                        original_predicted_durations=original["predicted_duration_lengths"], original_formal_stop_reason=original["stop_reason"],
                        original_duration_predictions_reproduced=decoded["predicted_duration_lengths"] == original["predicted_duration_lengths"],
                        original_sample_path=str(path), original_sample_sha256=original["sample_artifact"]["sha256"],
                        public_prompt=cue_text(episode), observed_feedback=episode["feedback"], seconds=time.perf_counter()-start_time,
                        formal_answer_replaced=False, formal_success_claimed=False, **decoded)
                    results.append(row); record(row); ledger.add(saved_latent_span_answers=1, auxiliary_answers=1)
                    del payload, recalled

        for stream in ("A", "B"):
            for cue_id in ("s1", "n1"):
                episode, target = sources[stream][cue_id], targets[(stream, cue_id)]
                expected_ids = target["target_ids"][:target["body_tokens"]+1].tolist()
                for duration in ("predicted", "true"):
                    active = f"true_latent_reference:{stream}:{cue_id}:{duration}"
                    check()
                    start_time = time.perf_counter()
                    decoded = decode_latents(target["prefix"], target["target"],
                        target["target_lengths"] if duration == "true" else None)
                    row = dict(event="true_latent_reference", stream=stream, cue_id=cue_id, source_visible=True,
                        true_target_latents_used=True, true_spans_visible=duration == "true", public_prompt=cue_text(episode),
                        observed_feedback=episode["feedback"], expected_feedback_through_first_eos_ids=expected_ids,
                        feedback_through_first_eos_exact=decoded["output_ids"] == expected_ids,
                        seconds=time.perf_counter()-start_time, formal_success_claimed=False, **decoded)
                    results.append(row); record(row); ledger.add(true_latent_reference_answers=1, auxiliary_answers=1)

        # Append the null control after the original 24 outputs; no learned result selects these cases.
        for baseline in ("base_M0", "shared_M0"):
            for cue_id in ("s1", "n1"):
                active = f"load_baseline_latent:{baseline}:{cue_id}"
                check()
                original = reads[(baseline, cue_id, 311121)]
                path = approved(original["sample_artifact"]["path"])
                if not path.is_relative_to(args.read_output) or sha(path) != original["sample_artifact"]["sha256"]:
                    raise ValueError("Baseline latent artifact path/hash mismatch")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if (payload["schema"] != "cola2_v4_generated_latents"
                        or (payload["state"], payload["cue_id"], payload["eval_seed"]) != (baseline, cue_id, 311121)
                        or payload["derived_seed"] != noise_seed(cue_id, 311121) or payload["source_visible"] is not False):
                    raise ValueError("Baseline latent identity mismatch")
                recalled = payload["latents"].to(device)
                if tuple(recalled.shape) != (K, 64):
                    raise ValueError("Baseline must contain all 24 original generated latents")
                baseline_samples_seen.add((baseline, cue_id, 311121))
                ledger.add(baseline_unique_latents=1)
                for target_stream in ("A", "B"):
                    active = f"baseline_latent_true_span:{baseline}:{cue_id}:{target_stream}"
                    check()
                    episode, target = sources[target_stream][cue_id], targets[(target_stream, cue_id)]
                    if (original["public_prompt"] != cue_text(episode)
                            or original["derived_seed"] != noise_seed(cue_id, 311121)
                            or payload["prefix"] != fingerprint(target["prefix"]["scaled"])):
                        raise ValueError("Baseline/source public cue or independent prefix mismatch")
                    start_time = time.perf_counter()
                    decoded = decode_latents(target["prefix"], recalled, target["target_lengths"])
                    row = dict(event="baseline_latent_true_span", state=baseline, baseline=baseline,
                        target_stream=target_stream, cue_id=cue_id, eval_seed=311121, source_visible=True,
                        baseline_sample_key=[baseline, cue_id, 311121], baseline_has_personal_learning=False,
                        true_spans_visible=True, true_target_latents_used=False,
                        original_formal_raw_output=original["raw_output"], original_formal_output_ids=original["output_ids"],
                        original_predicted_durations=original["predicted_duration_lengths"], original_formal_stop_reason=original["stop_reason"],
                        original_duration_predictions_reproduced=decoded["predicted_duration_lengths"] == original["predicted_duration_lengths"],
                        original_sample_path=str(path), original_sample_sha256=original["sample_artifact"]["sha256"],
                        public_prompt=cue_text(episode), observed_feedback=episode["feedback"],
                        seconds=time.perf_counter()-start_time, formal_answer_replaced=False,
                        formal_success_claimed=False, personal_learning_success_claimed=False, **decoded)
                    results.append(row); record(row); ledger.add(baseline_latent_span_answers=1, auxiliary_answers=1)
                del payload, recalled
        if ledger.counts.get("auxiliary_answers") != 32 or len(baseline_samples_seen) != 4:
            raise RuntimeError("Fixed 32-answer auxiliary grid or four baseline samples incomplete")
        details["base_frozen_verified"] = verify_base()
        status = "complete"
    except BaseException as exc:
        status = "capacity_failed" if isinstance(exc, CapacityFailure) else "budget_exhausted" if isinstance(exc, TimeoutError) else "failed"
        error = traceback.format_exc()
        record(dict(event="failure", active_call=active, exception_type=type(exc).__name__, error=error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write_json(args.output / "span_diagnostic.json", dict(
            complete=len(results) == 32, planned_outputs=32, completed_outputs=len(results),
            original_24_outputs_retained=True, planned_baseline_unique_samples=4,
            baseline_unique_samples_loaded=len(baseline_samples_seen), baseline_sample_keys=sorted(baseline_samples_seen),
            source_visible=True, new_latent_samples=0, optimizer_updates=0, original_answers_modified=False,
            formal_success_claimed=False, results=results))
        if model is not None and base:
            current = dict(model.named_parameters())
            changed = [name for name, (p, identity, version) in base.items() if id(current.get(name)) != identity
                       or p._version != version or p.requires_grad or p.grad is not None]
            details.update(base_frozen_verified=not changed, changed_parameter_names=changed)
            del current
        if forbidden_handle is not None:
            forbidden_handle.remove()
        base.clear(); targets.clear()
        model = tokenizer = target = payload = recalled = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            peaks["allocated"] = max(peaks["allocated"], torch.cuda.max_memory_allocated(args.device))
            peaks["reserved"] = max(peaks["reserved"], torch.cuda.max_memory_reserved(args.device))
            torch.cuda.empty_cache()
            details["allocated_bytes_after_release"] = torch.cuda.memory_allocated(args.device)
        budget.close("user_memory_cola_v4_span_diagnostic", args.output)
        ledger.finish(status, error=error, active_call=active, peak_allocated_bytes=peaks["allocated"],
            peak_reserved_bytes=peaks["reserved"], cumulative_budget_used_seconds=budget.state["used_seconds"], **details)
    if status != "complete":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
