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


import site
import sys
import zipfile
import argparse
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dist"),
        help="Directory for the output zip (default: dist)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dp = _find_dreamplace_dir()

    py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    zip_name = f"dreamplace-{_platform_tag()}-py{py_tag}-cpu.zip"
    out_zip = args.out_dir / zip_name

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dp.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(dp).as_posix()
                zf.write(path, arcname)

    print(f"Wrote {out_zip.resolve()}")


if __name__ == "__main__":
    main()
