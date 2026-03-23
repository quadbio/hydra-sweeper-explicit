Launch all `_launcher_` groups concurrently using `ThreadPoolExecutor`.

**Problem**: `launcher.launch()` (submitit) blocks until SLURM jobs complete.
With sequential group dispatch, the sweeper waits for group 1 to finish
before even submitting group 2. If the submitter process dies during the
wait, remaining groups never get submitted.

**Fix**: Pre-build all launcher instances, then fan out all `launch()` calls
concurrently via threads. All `sbatch` calls fire within seconds of each
other. Result collection happens in parallel.

- Single-group case (no `_launcher_` or all same) stays single-threaded
- Each launcher instance is independent — no shared state, thread-safe
- `initial_job_idx` ensures distinct output subdirectories across groups
