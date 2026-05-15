import os
import re
import sys
import stat
import site
import shlex
import shutil
import pathlib
import subprocess
from dataclasses import dataclass


###########################
#     ThirdParty Path     #
###########################

DREAMPLACE_THIRDPARTY_PATH = pathlib.Path(__file__).parent / "source/thirdparty"


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
    def _parse_version(value: str | None) -> tuple[int, int] | None:
        if not value:
            return None
        match = re.search(r"(\d+)(?:\.(\d+))?", value)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2) or 0)

    @classmethod
    def _version_lt(cls, lhs: str | None, rhs: str | None) -> bool:
        lhs_version = cls._parse_version(lhs)
        rhs_version = cls._parse_version(rhs)
        return bool(lhs_version and rhs_version and lhs_version < rhs_version)

    @staticmethod
    def _run_output(command: list[str]) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
        except Exception:
            return None

    @classmethod
    def _rule_for_gpu_name(cls, name: str) -> GpuRule | None:
        lowered = name.lower()
        for rule in GPU_RULES:
            if re.search(rule.pattern, lowered):
                return rule
        return None

    @classmethod
    def _min_cuda_for_arch(cls, arch: str) -> str | None:
        for rule in GPU_RULES:
            if rule.arch == arch:
                return rule.min_cuda
        return None

    def _query_torch(self) -> tuple[list[dict[str, str]], str | None, str | None]:
        try:
            import torch
        except Exception as exc:
            print(f"PyTorch import failed: {exc}", file=sys.stderr)
            return [], None, None

        print(f"Python executable: {sys.executable}", file=sys.stderr)
        print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
        print(f"PyTorch CUDA version: {torch.version.cuda}", file=sys.stderr)

        gpus: list[dict[str, str]] = []
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

    def _query_nvidia_smi(self) -> list[dict[str, str]]:
        output = self._run_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
        if not output:
            return []

        gpus: list[dict[str, str]] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            name, compute_cap = [field.strip() for field in line.rsplit(",", 1)]
            gpus.append({"name": name, "compute_cap": compute_cap, "source": "nvidia-smi"})
        return gpus

    def _query_nvcc_version(self) -> str | None:
        output = self._run_output(["nvcc", "--version"])
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

    def resolve_cuda_architecture(self) -> str | None:
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

    def cmake_arguments(self) -> list[str]:
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
        if self.cpu_only:
            args.append("-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON")
        else:
            cuda_arch = self.resolve_cuda_architecture()
            if cuda_arch:
                args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
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

    def write_build_script(self, path: str | os.PathLike[str] | None = None) -> pathlib.Path:
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
                "to missing dependencies. You may need to install the packages: "
                "flex bison zlib1g-dev libbz2-dev libfl-dev libboost-all-dev libcairo2. "
                "If installation still fails, please refer to the error messages for "
                "any additional packages that may be required and install them accordingly."
            )
        finally:
            os.chdir(original_path)

        # Step6: Move the install directory to the final path
        shutil.move(self.install_path / "dreamplace", self.final_path)

        # Step7: Remove the build directory and install directory
        shutil.rmtree(build_path)
        shutil.rmtree(self.install_path)