r"""
Precompile DreamPlace (CPU-only) and package it into a zip file.

The zip contains the ``dreamplace`` package files at the archive root (no
extra wrapper directory or metadata files). Extract into
``<site-packages>/dreamplace`` to install.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)
import site
import zipfile
import pathlib
from pathlib import Path
from packaging import version
from ml4co_kit import EnvInstallHelper


def _find_dreamplace_dir() -> Path:
    for base in map(Path, site.getsitepackages()):
        candidate = base / "dreamplace"
        if candidate.is_dir():
            return candidate.resolve()
    user = Path(site.getusersitepackages())
    candidate = user / "dreamplace"
    if candidate.is_dir():
        return candidate.resolve()
    sys.exit(
        "Could not find installed package directory 'dreamplace' under site-packages. "
        "Run DreamPlaceSolver().install() in this environment first."
    )


def _platform_tag() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform.replace("/", "-")


if __name__ == "__main__":
    # Get pytorch version
    python_version = sys.version.split()[0]
    
    # Get pytorch version
    if version.parse(python_version) < version.parse("3.12"):
        pytorch_version = "2.1.0"
    elif version.parse(python_version) < version.parse("3.13"):
        pytorch_version = "2.4.0"
    else:
        pytorch_version = "2.7.0"
    
    # Install basic environment
    env_install_helper = EnvInstallHelper(pytorch_version=pytorch_version)
    env_install_helper.install()
    
    # Install DreamPlace
    from ml4co_kit.solver.eda.dreamplace import DreamPlaceSolver
    dreamplace_solver = DreamPlaceSolver()
    dreamplace_solver.install(cpu_only=True)

    # Get output zip name
    os.makedirs("dist", exist_ok=True)
    dp = _find_dreamplace_dir()
    py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    out_zip = f"dist/dreamplace-{_platform_tag()}-py{py_tag}-cpu.zip"
    out_zip = pathlib.Path(out_zip)

    # Zip the dreamplace directory
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dp.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(dp).as_posix()
                zf.write(path, arcname)

    # Print the output zip name
    print(f"Wrote {out_zip.resolve()}")
