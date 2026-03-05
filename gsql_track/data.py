"""
Data preprocessing pipeline with caching and subprocess-parallel CLI.

CLI:
    python -m gsql_track.data -c config.yaml -y              # sequential
    python -m gsql_track.data -c config.yaml -y -g 0,1,2,3  # parallel across 4 GPUs
    python -m gsql_track.data -c config.yaml -y -n 4         # 4 parallel workers (no GPU pinning)
    python -m gsql_track.data -c config.yaml --status        # check progress
    python -m gsql_track.data -c config.yaml --reset         # reset stale jobs
"""
from __future__ import annotations
import argparse
import importlib
import os
import pickle
import sqlite3
import subprocess
import sys
import time
import yaml
from collections import deque
from pathlib import Path
from typing import Dict

from .types import (
    PreprocessStep,
    StepConfig,
    PREPROCESS_STEP_REGISTRY,
    register_pp,
    DATA_REGISTRY,
)


# === Core ===

class DataPreprocessor:
    """Preprocessing pipeline with optional pkl caching.

    >>> pp = DataPreprocessor(steps, cache_dir=Path('./data'))
    >>> train, valid, test = pp(train, valid, test)
    """
    def __init__(self, steps, cache_dir=None, cache_key=None):
        self.steps = steps
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._key = cache_key

    @property
    def key(self):
        if self._key is None:
            import hashlib
            self._key = hashlib.sha256(
                "_".join(s.cache_key for s in self.steps).encode()
            ).hexdigest()[:12]
        return self._key

    def _path(self, split):
        return self.cache_dir / f"{split}_{self.key}.pkl"

    def _load(self, data, split):
        if not self.cache_dir:
            return False
        p = self._path(split)
        if not p.exists():
            return False
        with open(p, 'rb') as f:
            data.features = pickle.load(f)
        return True

    def _save(self, data, split):
        if not self.cache_dir:
            return
        p = self._path(split)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(data.features, f, protocol=4)

    def fit(self, train):
        for s in self.steps:
            s._fitted_state = s.fit(train)
        return self

    def transform(self, data, split=None):
        if split and self._load(data, split):
            return data
        for s in self.steps:
            s.transform(data, s._fitted_state)
        if split:
            self._save(data, split)
        return data

    def __call__(self, train, valid=None, test=None, force=False):
        pairs = [(s, d) for s, d in [('train', train), ('valid', valid), ('test', test)] if d is not None]
        if not force and self.cache_dir and all(self._path(s).exists() for s, _ in pairs):
            for s, d in pairs:
                self._load(d, s)
        else:
            self.fit(train)
            for s, d in pairs:
                self.transform(d, s)
        return tuple(d for _, d in pairs) if len(pairs) > 1 else pairs[0][1]

    process = __call__  # backwards compat

    def cached(self, splits=('train', 'valid', 'test')):
        """Check if all splits are cached."""
        return self.cache_dir and all(self._path(s).exists() for s in splits)

    def __repr__(self):
        return f"DataPreprocessor(key={self.key!r}, steps={len(self.steps)}, cache={self.cache_dir})"


# === Factories ===

def create_pp(config: Dict) -> PreprocessStep:
    """Create preprocessing step from config dict."""
    c = config.copy()
    t = c.pop('type')
    if t not in PREPROCESS_STEP_REGISTRY:
        raise ValueError(f"Unknown step: {t}. Available: {list(PREPROCESS_STEP_REGISTRY)}")
    return PREPROCESS_STEP_REGISTRY[t](**c)


def create_preprocessor(preprocess_config, cache_dir, cache_name=None):
    """Create DataPreprocessor from config list."""
    return DataPreprocessor(
        [create_pp(c) for c in preprocess_config],
        cache_dir=cache_dir, cache_key=cache_name,
    )


def load_data(data_class="wrench_data", init_args=None, process_args=None):
    """Universal data loading from DATA_REGISTRY."""
    ia, pa = init_args or {}, process_args or {}
    if data_class not in DATA_REGISTRY:
        raise ValueError(f"Unknown: {data_class}. Available: {list(DATA_REGISTRY)}")
    return DATA_REGISTRY[data_class](**ia).process(**pa)


__all__ = ['DataPreprocessor', 'create_pp', 'create_preprocessor', 'load_data', 'JobTracker']


# === SQLite Job Tracker ===

class JobTracker:
    """SQLite-backed job tracker for parallel preprocessing.

    >>> tracker = JobTracker('/tmp/preprocess.db')
    >>> tracker.add(['yelp', 'imdb'])
    >>> tracker.set('yelp', 'running', gpu='0')
    >>> tracker.set('yelp', 'done', elapsed=12.3)
    """
    def __init__(self, db_path):
        self.db = str(db_path)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS jobs("
                "name TEXT PRIMARY KEY, status TEXT DEFAULT 'pending', "
                "error TEXT, elapsed REAL, gpu TEXT, ts REAL)"
            )

    def _conn(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def add(self, names):
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO jobs(name, ts) VALUES(?,?)",
                [(n, time.time()) for n in names],
            )

    def set(self, name, status, error=None, elapsed=None, gpu=None):
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, error=?, elapsed=COALESCE(?,elapsed), "
                "gpu=COALESCE(?,gpu), ts=? WHERE name=?",
                (status, error, elapsed, gpu, time.time(), name),
            )

    def get(self, name):
        with self._conn() as c:
            r = c.execute("SELECT status FROM jobs WHERE name=?", (name,)).fetchone()
            return r[0] if r else None

    def pending(self):
        with self._conn() as c:
            return [r[0] for r in c.execute(
                "SELECT name FROM jobs WHERE status IN ('pending','error') ORDER BY rowid"
            )]

    def summary(self):
        with self._conn() as c:
            return dict(c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())

    def reset(self):
        with self._conn() as c:
            c.execute("UPDATE jobs SET status='pending', error=NULL, elapsed=NULL")

    def report(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, status, error, elapsed, gpu FROM jobs ORDER BY rowid"
            ).fetchall()
        for name, status, error, elapsed, gpu in rows:
            sym = {'done': '+', 'error': 'x', 'running': '~'}.get(status, '.')
            parts = [f"[{sym}] {name}"]
            if gpu is not None:
                parts.append(f"gpu={gpu}")
            if elapsed:
                parts.append(f"{elapsed:.1f}s")
            if error:
                parts.append(f"ERR: {error}")
            print(f"  {' | '.join(parts)}")
        s = self.summary()
        total = sum(s.values())
        print(f"\n  {total} total | {s.get('done',0)} done | "
              f"{s.get('error',0)} error | {s.get('pending',0)} pending")


# === Dataset Processing ===

def _preprocess_one(spec, global_cache_name, force=False):
    """Run preprocessing for one dataset spec via load_data. Returns status string."""
    dc = spec.get('data_class', 'wrench_data')
    cn = spec.get('cache_name', global_cache_name)
    _skip = {'data_class', 'name', 'preprocess', 'cache_name', 'n_samples'}

    init_args = {'name': spec['name'], **{k: v for k, v in spec.items() if k not in _skip}}
    process_args = {k: spec[k] for k in ('preprocess', 'n_samples') if k in spec}
    process_args.update(cache_name=cn, preprocess_force=force)

    load_data(data_class=dc, init_args=init_args, process_args=process_args)
    return 'done'


# === CLI: Worker subprocess ===

def _worker_main(config_path, dataset_name, force=False):
    """Subprocess entry point: preprocess one dataset, update SQLite tracker."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for m in cfg.get('pre_import_modules', []):
        importlib.import_module(m)

    db_path = str(Path(config_path).with_suffix('.db'))
    tracker = JobTracker(db_path)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
    tracker.set(dataset_name, 'running', gpu=gpu)

    spec = next((d for d in cfg['datasets'] if d['name'] == dataset_name), None)
    if not spec:
        tracker.set(dataset_name, 'error', error=f"'{dataset_name}' not in config")
        raise SystemExit(1)

    t0 = time.time()
    try:
        result = _preprocess_one(spec, cfg['cache_name'], force=force)
        tracker.set(dataset_name, 'done', elapsed=time.time() - t0)
        print(f"[+] {dataset_name}: {result} ({time.time()-t0:.1f}s, GPU {gpu})")
    except Exception as e:
        tracker.set(dataset_name, 'error', error=str(e), elapsed=time.time() - t0)
        print(f"[x] {dataset_name}: {e}")
        raise SystemExit(1)


# === CLI: Main orchestrator ===

def main_cli(config_path, yes=False, force=False, gpus=None, n_workers=None, status=False, reset=False):
    """Orchestrate preprocessing: sequential, GPU-parallel, or worker-parallel."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    db_path = str(Path(config_path).with_suffix('.db'))
    db = JobTracker(db_path)

    if reset:
        db.reset()
        print("Reset all jobs to pending.")
        return
    if status:
        db.report()
        return

    datasets = cfg['datasets']
    cache_name = cfg['cache_name']
    for m in cfg.get('pre_import_modules', []):
        importlib.import_module(m)

    if force:
        db.reset()
    db.add([ds['name'] for ds in datasets])

    print(f"\n{'='*50}")
    print(f"Preprocessing {len(datasets)} dataset(s) | cache={cache_name}")
    if gpus:
        print(f"GPUs: {','.join(gpus)} ({len(gpus)} parallel)")
    elif n_workers:
        print(f"Workers: {n_workers} parallel")
    print(f"{'='*50}")
    for i, ds in enumerate(datasets, 1):
        steps = ', '.join(s.get('type', '?') for s in ds.get('preprocess', []))
        print(f"  {i}. {ds['name']} [{steps}] ({db.get(ds['name'])})")

    if not yes:
        try:
            if input("\nProceed? [Y/n] ").strip().lower() in ('n', 'no'):
                return
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

    pending_names = set(db.pending())
    pending = deque(ds for ds in datasets if ds['name'] in pending_names)
    if not pending:
        print("\nAll datasets already processed.")
        db.report()
        return

    print(f"\nProcessing {len(pending)} dataset(s)...\n")
    t0 = time.time()

    if gpus:
        slot_q = deque(gpus)
        active = {}

        while pending or active:
            while pending and slot_q:
                ds = pending.popleft()
                gpu = slot_q.popleft()
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
                cmd = [
                    sys.executable, '-m', 'gsql_track.data',
                    '--worker', '-c', config_path, '-d', ds['name'],
                ]
                if force:
                    cmd.append('-f')
                proc = subprocess.Popen(cmd, env=env)
                active[proc.pid] = (proc, gpu, ds['name'])
                print(f"  [~] {ds['name']} -> GPU {gpu} (pid={proc.pid})")

            for pid in list(active):
                proc, slot, name = active[pid]
                if proc.poll() is not None:
                    del active[pid]
                    slot_q.append(slot)
                    sym = '+' if proc.returncode == 0 else 'x'
                    print(f"  [{sym}] {name} (GPU {slot}, exit={proc.returncode})")
            if active:
                time.sleep(1)

    elif n_workers:
        slot_q = deque(range(n_workers))
        active = {}

        while pending or active:
            while pending and slot_q:
                ds = pending.popleft()
                slot = slot_q.popleft()
                cmd = [
                    sys.executable, '-m', 'gsql_track.data',
                    '--worker', '-c', config_path, '-d', ds['name'],
                ]
                if force:
                    cmd.append('-f')
                proc = subprocess.Popen(cmd)
                active[proc.pid] = (proc, slot, ds['name'])
                print(f"  [~] {ds['name']} -> worker {slot} (pid={proc.pid})")

            for pid in list(active):
                proc, slot, name = active[pid]
                if proc.poll() is not None:
                    del active[pid]
                    slot_q.append(slot)
                    sym = '+' if proc.returncode == 0 else 'x'
                    print(f"  [{sym}] {name} (worker {slot}, exit={proc.returncode})")
            if active:
                time.sleep(1)

    else:
        for ds in pending:
            name = ds['name']
            db.set(name, 'running')
            t1 = time.time()
            try:
                result = _preprocess_one(ds, cache_name, force=force)
                db.set(name, 'done', elapsed=time.time() - t1)
                print(f"  [+] {name}: {result} ({time.time()-t1:.1f}s)")
            except Exception as e:
                db.set(name, 'error', error=str(e), elapsed=time.time() - t1)
                print(f"  [x] {name}: {e}")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f}s")
    db.report()
    if db.summary().get('error', 0):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocessing CLI with subprocess parallelism and SQLite tracking",
        epilog="Examples:\n"
               "  python -m gsql_track.data -c config.yaml -y\n"
               "  python -m gsql_track.data -c config.yaml -y -g 0,1,2,3\n"
               "  python -m gsql_track.data -c config.yaml -y -n 4\n"
               "  python -m gsql_track.data -c config.yaml --status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-c', '--config', required=True, help='YAML config path')
    p.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    p.add_argument('-f', '--force', action='store_true', help='Force rebuild')
    p.add_argument('-g', '--gpus', help='Comma-separated GPU IDs for parallel (e.g. 0,1,2,3)')
    p.add_argument('-n', '--n-workers', type=int, help='Number of parallel workers (no GPU pinning)')
    p.add_argument('--status', action='store_true', help='Show job progress')
    p.add_argument('--reset', action='store_true', help='Reset all jobs to pending')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('-d', '--dataset', help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.worker:
        _worker_main(a.config, a.dataset, a.force)
    else:
        gpu_list = [g.strip() for g in a.gpus.split(',')] if a.gpus else None
        main_cli(a.config, a.yes, a.force, gpu_list, a.n_workers, a.status, a.reset)
