"""Write isolated LIBERO configuration paths; does not import or run either simulator."""
from pathlib import Path

for label, repo in (
    ("libero", "/volt/code/frozen-flow-study/third_party/libero"),
    ("libero-plus", "/volt/benchmarks/libero-plus"),
):
    root = Path(repo) / "libero/libero"
    config = Path("/volt/benchmarks/config-" + label)
    config.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    (config / "config.yaml").write_text("\n".join(k + ": " + v for k, v in paths.items()) + "\n")
