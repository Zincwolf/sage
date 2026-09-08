"""Import isolation without aliases, including pending extension-backed imports."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("qtip_first", [False, True])
@pytest.mark.parametrize("core", [False, True])
def test_qtip_and_quip_sharp_import_independently(qtip_first: bool, core: bool) -> None:
    if core:
        missing = [
            name
            for name in ("quiptools_cuda", "primefac")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            pytest.skip(f"QuIP# upstream core imports still require: {', '.join(missing)}")
    qtip = "import lib"
    quip_sharp = "import quip_sharp"
    if core:
        qtip += "\nfrom lib.algo import finetune as qtip_finetune"
        quip_sharp += "\nfrom quip_sharp.algo import finetune as quip_sharp_finetune"
    imports = (qtip, quip_sharp) if qtip_first else (quip_sharp, qtip)
    root = Path(__file__).resolve().parents[1]
    script = (
        "\n".join(imports)
        + f"""
from pathlib import Path
assert Path(lib.__file__).resolve() == Path({str(root / "third_party/qtip/lib/__init__.py")!r})
assert Path(quip_sharp.__file__).resolve() == (
    Path({str(root / "third_party/quip-sharp/quip_sharp/__init__.py")!r})
)
assert lib is not quip_sharp
"""
    )
    if core:
        script += """
import sys
assert 'lm_eval' not in sys.modules
assert 'quip_sharp.utils.lm_eval_adaptor' not in sys.modules
assert qtip_finetune.utils is lib.utils
assert quip_sharp_finetune.utils is quip_sharp.utils
from lib import codebook as qtip_codebook
from quip_sharp import codebook as quip_codebook
assert qtip_codebook is not quip_codebook
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_upstream_sources_do_not_import_qtip_lib() -> None:
    upstream = Path(__file__).resolve().parents[1] / "third_party/quip-sharp"
    assert not (upstream / "lib").exists()
    assert not (upstream / "quip_sharp").is_symlink()
    for directory in ("quip_sharp", "model", "eval", "quantize_llama"):
        for path in (upstream / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert node.level or (node.module or "").split(".")[0] != "lib", path
                elif isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] != "lib" for alias in node.names), path
