r"""
DreamPlace Install Helper.
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
import re
import sys
import stat
import site
import shlex
import shutil
import pathlib
import tempfile
import subprocess
import ctypes.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from ml4co_kit.utils.file_utils import download, extract_archive


###########################
#     Basic Constants     #
###########################

# Path for DreamPlace thirdparty
DREAMPLACE_THIRDPARTY_PATH = pathlib.Path(__file__).parent / "source/thirdparty"

# Base URL for DreamPlace prebuilt packages
DREAMPLACE_PREBUILT_BASE_URL = (
    "https://huggingface.co/datasets/ML4CO/ML4CO-Kit/resolve/main/dreamplace/"
)


# Run a command and return the output
def _run_output(command: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


# Get the prefix of a Homebrew tool
def _darwin_homebrew_tool_prefix(formula: str, marker: pathlib.Path) -> Optional[pathlib.Path]:
    if sys.platform != "darwin":
        return None
    out = _run_output(["brew", "--prefix", formula])
    if not out:
        return None
    prefix = pathlib.Path(out.strip())
    return prefix if (prefix / marker).is_file() else None


# Get the prefix of the Homebrew libomp
def _darwin_homebrew_libomp_prefix() -> Optional[pathlib.Path]:
    if sys.platform != "darwin":
        return None
    out = _run_output(["brew", "--prefix", "libomp"])
    if not out:
        return None
    prefix = pathlib.Path(out.strip())
    dylib = prefix / "lib" / "libomp.dylib"
    return prefix if dylib.is_file() else None


# Parse the version of Bison
def _parse_bison_version(output: Optional[str]) -> Optional[Tuple[int, int]]:
    if not output:
        return None
    match = re.search(r"bison\s*\(?GNU Bison\)?\s*(\d+)\.(\d+)", output, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


# Check if a header file is in the include directories
def _header_in_dirs(rel_path: str, include_dirs: List[pathlib.Path]) -> bool:
    return any((directory / rel_path).is_file() for directory in include_dirs)


# Get the include directories for Linux
def _linux_include_dirs() -> List[pathlib.Path]:
    dirs = [pathlib.Path("/usr/include"), pathlib.Path("/usr/local/include")]
    return [directory for directory in dirs if directory.is_dir()]


# Check if the Linux system has the libcairo library
def _linux_has_libcairo() -> bool:
    if ctypes.util.find_library("cairo") is not None:
        return True
    for lib_name in ("libcairo.so.2", "libcairo.so"):
        try:
            ctypes.CDLL(lib_name)
            return True
        except OSError:
            continue
    return False


# Get the default site-packages path for DreamPlace
def _default_dreamplace_site_packages_path() -> pathlib.Path:
    return pathlib.Path(site.getsitepackages()[0]) / "dreamplace"


###########################
#    Build Env Checker    #
###########################


class DreamPlaceBuildEnvChecker(object):
    """
    Verify native build dependencies before compiling DreamPlace from source.

    Package lists align with ``.github/workflows/dreamplace-prebuilt.yml``.
    """

    LINUX_APT_PACKAGES = (
        "make cmake flex bison zlib1g-dev libbz2-dev libfl-dev "
        "libboost-all-dev clang"
    )
    MACOS_BREW_PACKAGES = "cmake boost flex bison libomp"

    @classmethod
    def check(cls) -> None:
        missing = cls.collect_missing()
        if not missing:
            return
        cls.raise_for_missing(missing)

    @classmethod
    def collect_missing(cls) -> List[str]:
        missing: List[str] = []

        for command in ("cmake", "make"):
            if shutil.which(command) is None:
                missing.append(f"command not found: {command}")

        if not (
            shutil.which("clang")
            or shutil.which("clang++")
            or shutil.which("g++")
            or shutil.which("gcc")
        ):
            missing.append("C++ compiler not found (clang or g++)")

        if sys.platform.startswith("linux"):
            missing.extend(cls._collect_linux_missing())
        elif sys.platform == "darwin":
            missing.extend(cls._collect_macos_missing())
        else:
            missing.append(f"unsupported platform for DreamPlace build: {sys.platform}")

        return missing

    @classmethod
    def _collect_linux_missing(cls) -> List[str]:
        missing: List[str] = []
        for command in ("flex", "bison"):
            if shutil.which(command) is None:
                missing.append(f"command not found: {command}")

        include_dirs = _linux_include_dirs()
        # Cairo headers are not required: DreamPlace sets CMAKE_DISABLE_FIND_PACKAGE_Cairo=ON.
        linux_headers = (
            ("zlib.h", "zlib1g-dev"),
            ("bzlib.h", "libbz2-dev"),
            ("FlexLexer.h", "libfl-dev"),
            ("boost/version.hpp", "libboost-all-dev"),
        )
        for header, package in linux_headers:
            if not _header_in_dirs(header, include_dirs):
                missing.append(f"header not found: {header} (apt install {package})")
        return missing

    @classmethod
    def _collect_macos_missing(cls) -> List[str]:
        missing: List[str] = []
        if shutil.which("brew") is None:
            missing.append("Homebrew not found (https://brew.sh)")
            return missing

        flex_prefix = _darwin_homebrew_tool_prefix("flex", pathlib.Path("include/FlexLexer.h"))
        if flex_prefix is None:
            missing.append("Homebrew flex with FlexLexer.h missing (brew install flex)")

        bison_prefix = _darwin_homebrew_tool_prefix("bison", pathlib.Path("bin/bison"))
        if bison_prefix is None:
            missing.append("Homebrew bison missing (brew install bison)")
        else:
            bison_bin = bison_prefix / "bin" / "bison"
            bison_version = _parse_bison_version(_run_output([str(bison_bin), "--version"]))
            if bison_version is None or bison_version < (3, 3):
                missing.append(
                    "bison >= 3.3 required (system /usr/bin/bison is too old; "
                    "brew install bison)"
                )

        boost_prefix = _run_output(["brew", "--prefix", "boost"])
        if not boost_prefix or not (
            pathlib.Path(boost_prefix.strip()) / "include/boost/version.hpp"
        ).is_file():
            missing.append("Homebrew boost headers missing (brew install boost)")

        if _darwin_homebrew_libomp_prefix() is None:
            missing.append("Homebrew libomp missing (brew install libomp)")
        return missing

    @classmethod
    def raise_for_missing(cls, missing: List[str]) -> None:
        lines = ["DreamPlace native build dependencies are missing:"]
        lines.extend(f"  - {item}" for item in missing)
        if sys.platform.startswith("linux"):
            lines.append("")
            lines.append("On Ubuntu/Debian, install with:")
            lines.append(f"  sudo apt-get install {cls.LINUX_APT_PACKAGES}")
        elif sys.platform == "darwin":
            lines.append("")
            lines.append("On macOS, install with:")
            lines.append(f"  brew install {cls.MACOS_BREW_PACKAGES}")
        raise RuntimeError("\n".join(lines))


###########################
#        Pre-Built        #
###########################

class DreamPlacePrebuilt(object):
    """
    Download and install a CPU-only prebuilt ``dreamplace`` package from HuggingFace.

    The archive layout matches ``scripts/dreamplace_prebuild.py``: package files live
    at the zip root and are moved into ``final_path`` (default: site-packages/dreamplace).
    """

    BASE_URL = DREAMPLACE_PREBUILT_BASE_URL

    def __init__(self, final_path: Optional[Union[str, pathlib.Path]] = None) -> None:
        if final_path is None:
            self.final_path = _default_dreamplace_site_packages_path()
        else:
            self.final_path = pathlib.Path(final_path)

    @classmethod
    def platform_tag(cls) -> str:
        if sys.platform.startswith("linux"):
            return "linux"
        if sys.platform == "darwin":
            return "macos"
        raise RuntimeError(
            f"Prebuilt DreamPlace is not available on platform: {sys.platform}"
        )

    @classmethod
    def archive_name(cls) -> str:
        py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
        return f"dreamplace-{cls.platform_tag()}-py{py_tag}-cpu.zip"

    @property
    def archive_url(self) -> str:
        return f"{self.BASE_URL}{self.archive_name()}"

    @classmethod
    def collect_missing_runtime_deps(cls) -> List[str]:
        missing: List[str] = []
        if sys.platform.startswith("linux"):
            if not _linux_has_libcairo():
                missing.append(
                    "libcairo2 runtime library not found (apt install libcairo2)"
                )
        elif sys.platform != "darwin":
            missing.append(
                f"unsupported platform for prebuilt DreamPlace: {sys.platform}"
            )
        return missing

    @classmethod
    def check_environment(cls) -> None:
        missing = cls.collect_missing_runtime_deps()
        if not missing:
            return
        lines = ["DreamPlace prebuilt install requirements are missing:"]
        lines.extend(f"  - {item}" for item in missing)
        if sys.platform.startswith("linux"):
            lines.append("")
            lines.append("On Ubuntu/Debian, install with:")
            lines.append("  sudo apt-get install libcairo2")
        raise RuntimeError("\n".join(lines))

    def install(self) -> pathlib.Path:
        # Check if the prebuilt package exists
        self.check_environment()

        # Get the archive name and URL
        archive_name = self.archive_name()
        archive_url = self.archive_url
        final_path = self.final_path

        # Download and extract the prebuilt package
        with tempfile.TemporaryDirectory(prefix="dreamplace_prebuilt_") as tmp_dir:
            
            # Get the temporary directory
            tmp = pathlib.Path(tmp_dir)

            # Get the archive path and extract path
            archive_path = tmp / archive_name
            extract_path = tmp / "extracted"

            # Download the prebuilt package
            download(file_path=archive_path.as_posix(), url=archive_url)

            # Extract the prebuilt package
            extract_path.mkdir(parents=True, exist_ok=True)
            extract_archive(
                archive_path=archive_path.as_posix(),
                extract_path=extract_path.as_posix(),
            )

            # Check if the extracted package is empty
            if not any(extract_path.iterdir()):
                raise RuntimeError(
                    f"Prebuilt archive is empty: {archive_name}. "
                    f"Check that {archive_url} exists for this platform/Python version."
                )

            # Remove the final path if it exists
            if final_path.exists():
                shutil.rmtree(final_path)

            # Move the extracted package to the final path
            shutil.move(extract_path.as_posix(), final_path.as_posix())

        # Return the final path
        return final_path


###########################
#        GPU Rules        #
###########################

@dataclass(frozen=True)
class GpuRule:
    pattern: str
    arch: str
    min_cuda: str


GPU_RULES = (
    GpuRule(r"\b(geforce\s+)?rtx\s+50[0-9]{2}\b|\brtx\s+pro\b.*blackwell", "12.0", "12.8"),
    GpuRule(r"\bgb200\b|\bb200\b|\bb100\b", "10.0", "12.8"),
    GpuRule(r"\bh100\b|\bh200\b|\bgh200\b", "9.0", "11.8"),
    GpuRule(r"\b(geforce\s+)?rtx\s+40[0-9]{2}\b|\bl40s?\b|\bl4\b|\brtx\s+(6000|5000|4500|4000|3500|3000|2000)\s+ada\b", "8.9", "11.8"),
    GpuRule(r"\ba100\b|\ba30\b", "8.0", "11.0"),
    GpuRule(r"\b(geforce\s+)?rtx\s+30[0-9]{2}\b|\ba10g?\b|\ba16\b|\ba40\b|\ba2\b|\brtx\s+a(6000|5500|5000|4500|4000|3000|2000)\b", "8.6", "11.1"),
    GpuRule(r"\b(geforce\s+)?rtx\s+20[0-9]{2}\b|\bt4\b|\bquadro\s+rtx\b", "7.5", "10.0"),
    GpuRule(r"\bv100\b", "7.0", "9.0"),
    GpuRule(r"\bp100\b", "6.0", "8.0"),
    GpuRule(r"\bp4\b|\bp40\b|\bgtx\s+10[0-9]{2}\b", "6.1", "8.0"),
)


###########################
#     Install Helper      #
###########################

class DreamPlaceInstallHelper(object):
    """
    In order to integrate DreamPlace and achieve automated installation, 
    we used the codebase as of 2026-05-14 (with appropriate modifications/fixes).
    DreamPlace project: https://github.com/limbo018/DREAMPlace
    """

    def __init__(self, cpu_only: bool = False) -> None:
        self.src_path = pathlib.Path(__file__).parent / "source"
        self.install_path = pathlib.Path(__file__).parent / "install"
        self.cpu_only = cpu_only
        site_packages_dirs = site.getsitepackages()
        self.final_path = pathlib.Path(site_packages_dirs[0]) / "dreamplace"

    @staticmethod
    def _parse_version(value: Optional[str]) -> Optional[Tuple[int, int]]:
        if not value:
            return None
        match = re.search(r"(\d+)(?:\.(\d+))?", value)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2) or 0)

    @classmethod
    def _version_lt(cls, lhs: Optional[str], rhs: Optional[str]) -> bool:
        lhs_version = cls._parse_version(lhs)
        rhs_version = cls._parse_version(rhs)
        return bool(lhs_version and rhs_version and lhs_version < rhs_version)

    @classmethod
    def _rule_for_gpu_name(cls, name: str) -> Optional[GpuRule]:
        lowered = name.lower()
        for rule in GPU_RULES:
            if re.search(rule.pattern, lowered):
                return rule
        return None

    @classmethod
    def _min_cuda_for_arch(cls, arch: str) -> Optional[str]:
        for rule in GPU_RULES:
            if rule.arch == arch:
                return rule.min_cuda
        return None

    def _query_torch(self) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str]]:
        try:
            import torch
        except Exception as exc:
            print(f"PyTorch import failed: {exc}", file=sys.stderr)
            return [], None, None

        print(f"Python executable: {sys.executable}", file=sys.stderr)
        print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
        print(f"PyTorch CUDA version: {torch.version.cuda}", file=sys.stderr)

        gpus: List[Dict[str, str]] = []
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(device_id)
                gpus.append(
                    {
                        "name": torch.cuda.get_device_name(device_id),
                        "compute_cap": f"{major}.{minor}",
                        "source": "pytorch",
                    }
                )
        else:
            print("PyTorch reports CUDA is not available.", file=sys.stderr)
        return gpus, torch.__version__, torch.version.cuda

    def _query_nvidia_smi(self) -> List[Dict[str, str]]:
        output = _run_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
        if not output:
            return []

        gpus: List[Dict[str, str]] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            name, compute_cap = [field.strip() for field in line.rsplit(",", 1)]
            gpus.append({"name": name, "compute_cap": compute_cap, "source": "nvidia-smi"})
        return gpus

    def _query_nvcc_version(self) -> Optional[str]:
        output = _run_output(["nvcc", "--version"])
        if not output:
            return None
        match = re.search(r"release\s+(\d+\.\d+)", output)
        return match.group(1) if match else None

    def _query_torch_cxx_abi(self) -> int:
        try:
            import torch
            return int(torch._C._GLIBCXX_USE_CXX11_ABI)
        except Exception as exc:
            print(f"Failed to query PyTorch C++ ABI; falling back to ABI=0: {exc}", file=sys.stderr)
            return 0

    def resolve_cuda_architecture(self) -> Optional[str]:
        """Return one CUDA architecture like '12.0', or None for CPU-only builds."""
        if self.cpu_only:
            print("cpu_only=True; configuring DREAMPlace without CUDA support.", file=sys.stderr)
            return None

        torch_gpus, _, torch_cuda = self._query_torch()
        smi_gpus = self._query_nvidia_smi()
        nvcc_cuda = self._query_nvcc_version()
        if nvcc_cuda:
            print(f"NVCC CUDA version: {nvcc_cuda}", file=sys.stderr)

        override = os.environ.get("DREAMPLACE_CUDA_ARCH")
        gpus = torch_gpus or smi_gpus
        if not gpus and not override:
            print("No CUDA GPU detected; configuring a CPU-only build.", file=sys.stderr)
            return None

        selected = gpus[0] if gpus else {"name": "DREAMPLACE_CUDA_ARCH override", "compute_cap": override or "", "source": "env"}
        if len(gpus) > 1:
            names = ", ".join(gpu["name"] for gpu in gpus)
            print(f"Multiple GPUs detected ({names}); compiling only for the first visible GPU.", file=sys.stderr)

        if override:
            arch = override
            min_cuda = self._min_cuda_for_arch(arch)
            decision = "DREAMPLACE_CUDA_ARCH override"
        else:
            rule = self._rule_for_gpu_name(selected["name"])
            arch = rule.arch if rule else selected["compute_cap"]
            min_cuda = rule.min_cuda if rule else self._min_cuda_for_arch(arch)
            decision = "GPU model rule" if rule else f"{selected['source']} compute capability"

        sm_number = arch.replace(".", "")
        print(f"Selected GPU: {selected['name']} ({selected.get('compute_cap', 'unknown')}, from {selected['source']})", file=sys.stderr)
        print(f"Architecture decision: {decision} -> compute_{sm_number} / sm_{sm_number}", file=sys.stderr)

        for label, version in (("nvcc", nvcc_cuda), ("PyTorch CUDA", torch_cuda)):
            if min_cuda and version and self._version_lt(version, min_cuda):
                raise RuntimeError(f"{label} {version} is too old for sm_{sm_number}; CUDA {min_cuda}+ is required.")

        return arch

    def _darwin_openmp_cmake_cache(self) -> List[str]:
        """
        CMake cache entries so ``find_package(OpenMP)``
        succeeds with Apple Clang + Homebrew libomp.
        """
        base = _darwin_homebrew_libomp_prefix()
        if base is None:
            return []
        inc = base / "include"
        dylib = base / "lib" / "libomp.dylib"
        omp_flags = f"-Xpreprocessor -fopenmp -I{inc}"
        return [
            f"-DOpenMP_CXX_FLAGS={omp_flags}",
            "-DOpenMP_CXX_LIB_NAMES=omp",
            f"-DOpenMP_omp_LIBRARY={dylib}",
            f"-DOpenMP_C_FLAGS={omp_flags}",
            "-DOpenMP_C_LIB_NAMES=omp",
        ]

    def _darwin_flex_bison_cmake_cache(self) -> List[str]:
        """
        CMake cache entries for keg-only Homebrew flex/bison on macOS.

        Apple's ``/usr/bin/flex`` lacks ``FlexLexer.h``; ``/usr/bin/bison`` is
        too old for Limbo (needs bison >= 3.3).
        """
        args: List[str] = []
        flex_prefix = _darwin_homebrew_tool_prefix(
            "flex", pathlib.Path("include/FlexLexer.h")
        )
        if flex_prefix is not None:
            flex_lib = flex_prefix / "lib" / "libfl.dylib"
            if not flex_lib.is_file():
                flex_lib = flex_prefix / "lib" / "libfl.a"
            args.extend(
                [
                    f"-DFLEX_EXECUTABLE={flex_prefix / 'bin' / 'flex'}",
                    f"-DFLEX_INCLUDE_DIR={flex_prefix / 'include'}",
                    f"-DFLEX_INCLUDE_DIRS={flex_prefix / 'include'}",
                ]
            )
            if flex_lib.is_file():
                args.append(f"-DFL_LIBRARY={flex_lib}")
        bison_prefix = _darwin_homebrew_tool_prefix(
            "bison", pathlib.Path("bin/bison")
        )
        if bison_prefix is not None:
            args.append(f"-DBISON_EXECUTABLE={bison_prefix / 'bin' / 'bison'}")
        return args

    def cmake_arguments(self) -> List[str]:
        args = [
            "-U",
            "CMAKE_CUDA_FLAGS",
            f"-DCMAKE_INSTALL_PREFIX={self.install_path}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_CXX_ABI={self._query_torch_cxx_abi()}",
            f"-DCMAKE_DISABLE_FIND_PACKAGE_Cairo=ON",
            f"-DCMAKE_DISABLE_FIND_PACKAGE_GUROBI=ON",
            f"-DCMAKE_DISABLE_FIND_PACKAGE_CPLEX=ON",
            f"-DCMAKE_DISABLE_FIND_PACKAGE_LPSOLVE=ON",
        ]
        if sys.platform == "darwin":
            # HeteroSTA releases ship Linux x86-64 ELF .so files only.
            args.append("-DBUILD_HETEROSTA=OFF")
        if self.cpu_only:
            args.append("-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON")
        else:
            cuda_arch = self.resolve_cuda_architecture()
            if cuda_arch:
                args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        args.extend(self._darwin_openmp_cmake_cache())
        args.extend(self._darwin_flex_bison_cmake_cache())
        return args

    def build_script_text(self) -> str:
        quoted_args = " \\\n  ".join(shlex.quote(arg) for arg in self.cmake_arguments())
        return (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'if [[ -f "${SCRIPT_DIR}/CMakeLists.txt" ]]; then\n'
            '  SOURCE_DIR="${SCRIPT_DIR}"\n'
            '  BUILD_DIR="${SCRIPT_DIR}/build"\n'
            '  mkdir -p "${BUILD_DIR}"\n'
            '  cd "${BUILD_DIR}"\n'
            "else\n"
            '  SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"\n'
            '  cd "${SCRIPT_DIR}"\n'
            "fi\n\n"
            f"cmake {quoted_args} \\\n"
            '  "${SOURCE_DIR}"\n'
        )

    def write_build_script(
        self, path: Optional[Union[str, pathlib.Path]] = None
    ) -> pathlib.Path:
        output_path = pathlib.Path(path) if path else self.src_path / "build.sh"
        output_path.write_text(self.build_script_text(), encoding="utf-8")
        output_path.chmod(output_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return output_path

    def install(self):
        # Step1: Remove previous build/install outputs to avoid stale extensions.
        build_path = self.src_path / "build"
        if build_path.exists():
            shutil.rmtree(build_path)
        if self.install_path.exists():
            shutil.rmtree(self.install_path)
        if self.final_path.exists():
            shutil.rmtree(self.final_path)

        # Step2: Create the build directory
        build_path.mkdir(parents=True, exist_ok=True)

        # Step3: Write the build script
        self.write_build_script(build_path / "build.sh")

        # Step4: Configure the project
        original_path = os.getcwd()
        try:
            # Build and Install the project
            os.chdir(build_path)
            subprocess.run(["bash", "build.sh"], check=True)
            subprocess.run(["make"], check=True)
            subprocess.run(["make", "install"], check=True)
        except:
            raise ModuleNotFoundError(
                "CMake Error or build failure occurred. This may be due "
                "to missing dependencies. On Linux you may need: "
                "flex bison zlib1g-dev libbz2-dev libfl-dev libboost-all-dev libcairo2. "
                "On macOS with Apple Clang, install via Homebrew: "
                "`brew install libomp flex bison` (then re-run install). "
                "HeteroSTA timing is Linux-only; macOS builds use OpenTimer. "
                "If installation still fails, refer to the CMake log above."
            )
        finally:
            os.chdir(original_path)

        # Step6: Move the install directory to the final path
        shutil.move(self.install_path / "dreamplace", self.final_path)

        # Step7: Remove the build directory and install directory
        shutil.rmtree(build_path)
        shutil.rmtree(self.install_path)