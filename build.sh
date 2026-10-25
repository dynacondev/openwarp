#!/usr/bin/env bash
# Idempotent dev build for openwarp.
#
#   ./build.sh                 # Release, auto-detect ORT/Torch, build all warp targets
#   ./build.sh --clean         # wipe build/ first (e.g. after toolchain changes)
#   ./build.sh --debug         # -DCMAKE_BUILD_TYPE=Debug (NN fill ~10x slower)
#   ./build.sh --ort-root DIR # explicit onnxruntime C++ release dir
#   ONNXRUNTIME_ROOT=DIR ./build.sh   # same via environment
#   ./build.sh --torch-root DIR # explicit unpacked libtorch dir
#   TORCH_ROOT=DIR ./build.sh   # same via environment
#
# Re-running is safe: configure carries the cached ONNXRUNTIME_ROOT /
# TORCH_ROOT, and a missing/invalid root simply falls back to the
# ORT/mock NN backend (never a broken build). Use --clean if CMake ever
# complains about stale cache.
#
# Hermetic CUDA runtime: when ORT C++ is enabled, build.sh symlinks the
# pip-pinned NVIDIA libs (from the mise venv) next to the provider .so
# and bakes $ORIGIN into its RUNPATH via patchelf — no LD_LIBRARY_PATH,
# no sudo/ldconfig, gamemode-proof. Needs: patchelf installed.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE="Release"
ORT_ROOT="${ONNXRUNTIME_ROOT:-}"
TORCH_ROOT="${TORCH_ROOT:-}"
WIPE=0
for arg in "$@"; do
	case "$arg" in
	--clean) WIPE=1 ;;
	--debug) BUILD_TYPE="Debug" ;;
	-h | --help) sed -n '2,14p' "$0"; exit 0 ;;
	--ort-root) echo "usage: --ort-root DIR" >&2; exit 2 ;;
	--ort-root=*) ORT_ROOT="${arg#--ort-root=}" ;;
	--torch-root) echo "usage: --torch-root DIR" >&2; exit 2 ;;
	--torch-root=*) TORCH_ROOT="${arg#--torch-root=}" ;;
	*) echo "unknown arg: $arg (see header)" >&2; exit 2 ;;
	esac
done

if [ "$WIPE" = 1 ]; then
	rm -rf build
fi
mkdir -p build

# Auto-detect a local onnxruntime C++ release (3060 host: tarball in ~/opt).
# Prefer gpu_cuda13 (matches the pinned onnxruntime-gpu 1.31 wheels), then
# any gpu build, then CPU. Explicit --ort-root / env always wins.
if [ -z "$ORT_ROOT" ]; then
	for cand in "$HOME"/opt/onnxruntime-linux-x64-gpu_cuda13-* \
		"$HOME"/opt/onnxruntime-linux-x64-gpu-* \
		"$HOME"/opt/onnxruntime-linux-x64-* \
		/opt/onnxruntime-linux-x64-gpu_cuda13-* \
		/opt/onnxruntime-linux-x64-gpu-* \
		/opt/onnxruntime-linux-x64-*; do
		if [ -f "$cand/include/onnxruntime_cxx_api.h" ] && ls "$cand"/lib/libonnxruntime.so* >/dev/null 2>&1; then
			ORT_ROOT="$cand"
			break
		fi
	done
fi

CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
if [ -n "$ORT_ROOT" ]; then
	if [ ! -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ]; then
		echo "error: ONNXRUNTIME_ROOT=$ORT_ROOT has no include/onnxruntime_cxx_api.h" >&2
		exit 1
	fi
	CMAKE_ARGS+=("-DONNXRUNTIME_ROOT=$ORT_ROOT")
fi

# LibTorch auto-detect (3060 host: unzip the libtorch zip into ~/opt).
# Explicit --torch-root / env always wins. Unlike ORT, torch needs no
# CUDA staging here: its bundled libs already carry $ORIGIN runpaths,
# so NEVER copy/symlink anything into the libtorch dir.
if [ -z "$TORCH_ROOT" ]; then
	for cand in "$HOME"/opt/libtorch* /opt/libtorch*; do
		if [ -f "$cand/share/cmake/Torch/TorchConfig.cmake" ]; then
			TORCH_ROOT="$cand"
			break
		fi
	done
fi
if [ -n "$TORCH_ROOT" ]; then
	if [ ! -f "$TORCH_ROOT/share/cmake/Torch/TorchConfig.cmake" ]; then
		echo "error: TORCH_ROOT=$TORCH_ROOT has no share/cmake/Torch/TorchConfig.cmake" >&2
		exit 1
	fi
	CMAKE_ARGS+=("-DTORCH_ROOT=$TORCH_ROOT")
fi

cmake -B build "${CMAKE_ARGS[@]}" > build/cmake-configure.log 2>&1 || {
	cat build/cmake-configure.log
	exit 1
}
grep -E "CMAKE_BUILD_TYPE|onnxruntime C\+\+|libtorch" build/cmake-configure.log || true
cmake --build build --target openwarp_vulkan_shaders openwarp-vulkan-test openwarp

# Hermetic CUDA runtime: the provider .so carries no RUNPATH, so its
# DT_NEEDED (cublas/cudart/curand) and its lazy dlopen()s (cublasLt,
# libcudnn.so by bare soname) ignore LD_LIBRARY_PATH games under
# sandboxed launchers. Fix it in the artifact, not the environment:
# symlink the pip-pinned NVIDIA libs next to it and bake $ORIGIN into
# its RUNPATH with patchelf. Zero env vars, zero sudo, survives
# gamemode. Re-runs are no-ops when links + RUNPATH are already right.
if grep -q "ONNXRUNTIME_LIB:FILEPATH=.*libonnxruntime" build/CMakeCache.txt; then
	ORT_LIB="$(dirname "$(grep ONNXRUNTIME_LIB:FILEPATH build/CMakeCache.txt | cut -d= -f2)")"
	PROVIDER="$ORT_LIB/libonnxruntime_providers_cuda.so"
	if [ -f "$PROVIDER" ]; then
		VENVDIR=""
		for v in "${VIRTUAL_ENV:-}" "$PWD/ml/.venv" "$PWD/../ml/.venv"; do
			[ -n "$v" ] || continue
			for d in "$v"/lib/python*/site-packages/nvidia; do
				[ -d "$d" ] && VENVDIR="$d" && break 2
			done
		done
		if [ -z "$VENVDIR" ]; then
			echo "note: no venv nvidia/ dir found; CUDA EP keeps system search order" >&2
		elif ! command -v patchelf >/dev/null 2>&1; then
			echo "note: patchelf missing (sudo apt install patchelf); skipping hermetic CUDA staging" >&2
		else
			need_libs="cublas/lib/libcublasLt.so.13 cublas/lib/libcublas.so.13 cuda_runtime/lib/libcudart.so.13 curand/lib/libcurand.so.10 cudnn/lib/libcudnn.so.9"
			staged=0
			for rel in $need_libs; do
				base="$(basename "$rel")"
				src="$VENVDIR/$rel"
				if [ -f "$src" ] && [ ! -e "$ORT_LIB/$base" ]; then
					ln -s "$src" "$ORT_LIB/$base"
					echo "staged $base"
					staged=1
				fi
			done
			if [ ! -e "$ORT_LIB/libcudnn.so" ]; then
				ln -s "$VENVDIR/cudnn/lib/libcudnn.so.9" "$ORT_LIB/libcudnn.so"
				echo "staged libcudnn.so"
				staged=1
			fi
			if ! readelf -d "$PROVIDER" | grep -qE "RUNPATH.*\\\$ORIGIN|RPATH.*\\\$ORIGIN"; then
				patchelf --set-rpath '$ORIGIN' "$PROVIDER"
				echo "patched RUNPATH=\$ORIGIN into $(basename "$PROVIDER")"
				staged=1
			fi
			[ "$staged" = 0 ] && echo "CUDA runtime staging: already hermetic"
		fi
	fi
fi

echo "---"
echo "build type : $(grep CMAKE_BUILD_TYPE:STRING build/CMakeCache.txt | cut -d= -f2)"
if grep -q "ONNXRUNTIME_LIB:FILEPATH=.*libonnxruntime" build/CMakeCache.txt; then
	echo "ORT C++    : $(grep ONNXRUNTIME_LIB:FILEPATH build/CMakeCache.txt | cut -d= -f2)"
else
	echo "ORT C++    : not found (NN fill uses Torch/mock backend)"
fi
TORCH_HIT="$(grep '^Torch_DIR:PATH=' build/CMakeCache.txt | cut -d= -f2 || true)"
case "$TORCH_HIT" in
"" | *NOTFOUND*) echo "Torch C++  : not found (NN fill uses ORT/mock backend)" ;;
*) echo "Torch C++  : $TORCH_HIT" ;;
esac
git log --oneline -1
