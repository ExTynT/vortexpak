"""Build the Rust AVX2 native core (cdylib) via cargo and ship it inside the
vortexpak package. The Python side loads it through ctypes (vortexpak/native.py),
so no PyO3/pybind11 layer is involved."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_ROOT = Path(__file__).parent
_MANIFEST = _ROOT / "vortexpak_native" / "Cargo.toml"
_CDYLIB_NAMES = [
    "vortexpak_native.dll",
    "vortexpak_native.pyd",
    "vortexpak_native.so",
    "libvortexpak_native.so",
    "vortexpak_native.dylib",
    "libvortexpak_native.dylib",
]


class CargoExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class BuildExtCargo(build_ext):
    def build_extension(self, ext: Extension) -> None:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(_MANIFEST)],
            check=True,
        )
        target_dir = _ROOT / "vortexpak_native" / "target" / "release"
        dest = Path(self.get_ext_fullpath(ext.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        for name in _CDYLIB_NAMES:
            src = target_dir / name
            if src.is_file():
                self.copy_file(str(src), str(dest), level=self.verbose)
                return
        raise RuntimeError(
            f"cargo build succeeded but produced no cdylib in {target_dir}"
        )


setup(
    ext_modules=[CargoExtension("vortexpak.vortexpak_native")],
    cmdclass={"build_ext": BuildExtCargo},
)
