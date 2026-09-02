# Contributing to FluxAlloc

Thank you for your interest in contributing to **FluxAlloc**! We welcome contributions, bug reports, feature suggestions, and pull requests from researchers and developers.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## Development Workflow

### 1. Environment Setup
Clone the repository and set up the development virtual environment:
```bash
git clone https://github.com/Imsachin010/flux-alloc.git
cd flux-alloc

# Create virtual environment
python -m venv fluxAlloc

# Activate environment (Windows)
.\fluxAlloc\Scripts\activate
# Activate environment (Linux/macOS)
# source fluxAlloc/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Verification & Tests
Ensure that your changes do not break trace integrity or existing benchmarks:
```bash
# Verify trace uniqueness and canary tests
python camera_ready/trace_audit.py

# Run unit tests
python -m unittest discover tests/
```

### 3. Reproducing Experiments
To verify that experimental outputs remain reproducible:
```bash
python camera_ready/run_experiments.py
python camera_ready/compute_stats.py
```

---

## Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names (e.g., `feat/lookahead-caching`, `fix/trace-seed-boundary`).
2. **Commit Messages**: Keep commit messages concise and descriptive.
3. **Deterministic Seeding**: Any new generator or evaluation function must accept explicit `seed` parameters and use independent `random.Random(seed)` instances to maintain reproducibility.
4. **Documentation**: Update relevant documentation in `docs/` or `README.md` if modifying CLI flags or core allocator interfaces.
5. **Code Style**: Follow PEP 8 guidelines for Python code.

---

## Reporting Issues

If you encounter a bug or unexpected behavior:
1. Open an issue on GitHub with a clear description.
2. Include the Python version, operating system, and reproduction command.
3. If related to benchmark outputs, provide the trace seed and workload type.
