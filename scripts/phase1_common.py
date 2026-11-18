"""Standard-library provenance and explicit cost records for phase one.

No neural memory implementation is defined here. Model entry points supply
their own finite budget, counts, native backend counters and CUDA peak bytes.
All paths must be checked against the approved root by the remote entry point.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


RUN_SCHEMA = 'phase1_run_v1'
# Payload contents will be fixed with the Titans-inspired memory architecture.
SHARED_SCHEMA = 'phase1_shared_v1'
MEMORY_SCHEMA = 'phase1_neural_memory_v1'
APPROVED_ROOT = Path(__file__).resolve().parents[1]


def approved(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(APPROVED_ROOT.resolve()):
        raise ValueError(f'Path outside the project checkout: {resolved}')
    return resolved


class ModelBudget:
    """Accumulate sequential model-process time across prepare/train/read jobs."""
    def __init__(self, path, limit_seconds=10800):
        self.path = approved(path)
        self.started = time.perf_counter()
        self.state = json.loads(self.path.read_text('utf-8')) if self.path.exists() else dict(
            limit_seconds=limit_seconds, used_seconds=0., jobs=[])
        if self.state['limit_seconds'] != limit_seconds:
            raise ValueError('The shared model wall budget cannot silently change')
        self.closed = False

    def remaining(self):
        return self.state['limit_seconds'] - self.state['used_seconds'] - (time.perf_counter() - self.started)

    def expired(self, reserve=0.):
        return self.remaining() <= reserve

    def close(self, phase, output):
        if self.closed:
            return
        seconds = time.perf_counter() - self.started
        self.state['used_seconds'] += seconds
        self.state['jobs'].append(dict(pid=os.getpid(), phase=phase, seconds=seconds, output=str(output), ended_utc=utc_now()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.state)
        self.closed = True


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def append_jsonl(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def counter_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def implementation_files():
    root = Path(__file__).resolve().parents[1]
    paths = [*root.glob('src/phase1_*.py'), *root.glob('scripts/phase1_*.py'),
             *root.glob('scripts/phase1_*.sh'), *root.glob('configs/phase1_*.json')]
    paths += [root / name for name in ('src/lm_backend.py', 'src/soft_memory_backend_v6.py',
              'src/concept_reader_v13.py', 'src/layer_memory_reader_v11.py',
              'scripts/associative_v11_common.py', 'scripts/remote_env.sh')]
    return sorted(paths)


class RunLedger:
    """One process, one new output directory, explicit budget and actual costs.

    Instantiation is for model job entry points, after path-boundary validation.
    CUDA synchronization/peak measurement belongs to those remote entry points;
    this utility never imports torch or starts a process.
    """

    def __init__(self, output, phase, budget, inputs=None, code_paths=()):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.counts = {}
        self.job = dict(schema=RUN_SCHEMA, phase=phase, pid=os.getpid(),
                        started_utc=utc_now(), output=str(self.output), budget=budget,
                        inputs=inputs or {}, code_sha256={str(Path(p).resolve()): sha(p)
                                                          for p in code_paths})
        write_json(self.output / 'job.json', self.job)
        root = Path(__file__).resolve().parents[1]
        for source in code_paths:
            relative = Path(source).resolve().relative_to(root)
            destination = self.output / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    def add(self, **counts):
        for key, value in counts.items():
            self.counts[key] = self.counts.get(key, 0) + value

    def record(self, row):
        append_jsonl(self.output / 'events.jsonl', row)

    def finish(self, status, backend_counters=None, peak_allocated_bytes=None,
               peak_reserved_bytes=None, **details):
        result = dict(schema=RUN_SCHEMA, phase=self.job['phase'], pid=os.getpid(),
                      status=status, started_utc=self.job['started_utc'], ended_utc=utc_now(),
                      output=str(self.output), budget=self.job['budget'],
                      wall_seconds=time.perf_counter() - self.started,
                      actual=dict(self.counts), backend=backend_counters,
                      peak_allocated_bytes=peak_allocated_bytes,
                      peak_reserved_bytes=peak_reserved_bytes, details=details)
        write_json(self.output / 'completion.json', result)
        return result
