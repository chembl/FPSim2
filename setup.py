from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import platform


class get_pybind_include:
    """Helper class to determine the pybind11 include path."""

    def __str__(self):
        import pybind11

        return pybind11.get_include()


# =============================================================================
# Platform detection (evaluated once at import time)
# =============================================================================
_MACHINE = platform.machine().lower()
IS_X86_64 = _MACHINE in ("x86_64", "amd64", "x64")
IS_ARM64 = _MACHINE in ("aarch64", "arm64")
IS_MACOS = sys.platform == "darwin"

# =============================================================================
# Build configuration from environment variables
# =============================================================================
# FPSIM2_ARCH: Architecture selection
#   - default: Scalar POPCNT (all x86-64), NEON (all ARM64)
#   - avx512:  AVX-512 VPOPCNTDQ (Ice Lake+, Zen4+)
#   - native:  Auto-detect via -march=native
#
# FPSIM2_FP_SIZE: Fixed fingerprint size in uint64s (4=256bit, 8=512bit, etc.)
#   - Enables compile-time optimizations, reduces flexibility
#
# For wheels: use defaults for maximum compatibility.
# Users can build from source with: FPSIM2_ARCH=avx512 FPSIM2_FP_SIZE=8

FPSIM2_ARCH = os.environ.get("FPSIM2_ARCH", "default").lower()
FPSIM2_FP_SIZE = os.environ.get("FPSIM2_FP_SIZE", "")


def _get_fp_size():
    """Parse and validate FPSIM2_FP_SIZE."""
    if FPSIM2_FP_SIZE:
        try:
            size = int(FPSIM2_FP_SIZE)
            if size > 0:
                return size
        except ValueError:
            pass
    return None


# =============================================================================
# Compiler flag configuration
# =============================================================================
# GCC/Clang x86-64 flags
_GCC_FLAGS = {
    "default": ["-mpopcnt"],
    "avx512": ["-mavx512f", "-mavx512vl", "-mavx512vpopcntdq"],
    "native": ["-march=native"],
}

# MSVC x86-64 flags (MSVC doesn't need explicit POPCNT)
_MSVC_FLAGS = {
    "default": [],
    "avx512": ["/arch:AVX512"],
    "native": [],
}

# MSVC defines most AVX-512 macros with /arch:AVX512 but not VPOPCNTDQ
_MSVC_AVX512_MACROS = [("__AVX512VPOPCNTDQ__", "1")]


ext_modules = [
    Extension(
        "FPSim2.FPSim2lib",
        sources=["FPSim2/src/sim.cpp", "FPSim2/src/utils.cpp", "FPSim2/src/wraps.cpp"],
        include_dirs=["FPSim2/src/include", get_pybind_include()],
        language="c++",
    ),
]


class BuildExt(build_ext):
    """Custom build extension for platform-specific compiler options."""

    def build_extensions(self):
        ct = self.compiler.compiler_type
        is_msvc = ct == "msvc"
        fp_size = _get_fp_size()

        # Determine compiler flags and macros
        if IS_X86_64:
            flags_dict = _MSVC_FLAGS if is_msvc else _GCC_FLAGS
            arch_flags = flags_dict.get(FPSIM2_ARCH, flags_dict["default"])
            arch_macros = _MSVC_AVX512_MACROS if (is_msvc and FPSIM2_ARCH == "avx512") else []
        else:
            # ARM64: NEON is mandatory, no special flags needed
            arch_flags = []
            arch_macros = []
            if FPSIM2_ARCH == "avx512":
                print(f"FPSim2: Warning: FPSIM2_ARCH=avx512 ignored on {_MACHINE} (using NEON)")

        # Build fingerprint size macro
        fp_macros = [("FPSIM2_FP_SIZE", str(fp_size))] if fp_size else []

        # Print build configuration
        self._print_config(arch_flags, arch_macros, fp_size)

        # Apply settings to all extensions
        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", f'"{self.distribution.get_version()}"')
            ] + arch_macros + fp_macros

            if is_msvc:
                ext.extra_compile_args = ["/EHsc", "/O2", "/std:c++17"] + arch_flags
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden"] + arch_flags
                if IS_MACOS:
                    ext.extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
                    ext.extra_link_args = ["-stdlib=libc++", "-mmacosx-version-min=10.14"]

        build_ext.build_extensions(self)

    def _print_config(self, arch_flags, arch_macros, fp_size):
        """Print build configuration if non-default settings are used."""
        if FPSIM2_ARCH == "default" and not fp_size:
            return

        if IS_X86_64:
            print(f"FPSim2: x86-64 build, FPSIM2_ARCH={FPSIM2_ARCH}")
            if arch_flags:
                print(f"  Compiler flags: {' '.join(arch_flags)}")
            if arch_macros:
                print(f"  Macros: {', '.join(f'{k}={v}' for k, v in arch_macros)}")
        elif IS_ARM64:
            print("FPSim2: ARM64 build (NEON always enabled)")

        if fp_size:
            print(f"  Fixed fingerprint size: {fp_size} uint64s ({fp_size * 64} bits)")


setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
