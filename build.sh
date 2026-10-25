#!/usr/bin/env bash
# Idempotent dev build for openwarp.
#
#   ./build.sh                 # Release, auto-detect ORT, build all warp targets
#   ./build.sh --clean         # wipe build/ first (e.g. after toolchain changes)
#   ./build.sh --debug         # -DCMAKE_BUILD_TYPE=Debug (NN fill ~10x slower)
#   ./build.sh --ort-root DIR # explicit onnxruntime C++ release dir
#   ONNXRUNTIME_ROOT=DIR ./build.sh   # same via environment
#
# Re-running is safe: configure carries the cached ONNXRUNTIME_ROOT, and a
# missing/invalid root simply falls back to the mock NN backend (never a
# broken build). Use --clean if CMake ever complains about stale cache.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE="Release"
ORT_ROOT="${ONNXRUNTIME_ROOT:-}"
WIPE=0
for arg in "$@"; do
	case "$arg" in
	--clean) WIPE=1 ;;
	--debug) BUILD_TYPE="Debug" ;;
	-h | --help) sed -n '2,12p' "$0"; exit 0 ;;
	--ort-root) echo "usage: --ort-root DIR" >&2; exit 2 ;;
	--ort-root=*) ORT_ROOT="${arg#--ort-root=}" ;;
	*) echo "unknown arg: $arg (see header)" >&2; exit 2 ;;
	esac
done

if [ "$WIPE" = 1 ]; then
	rm -rf build
fi

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

cmake -B build "${CMAKE_ARGS[@]}" > /tmp/opencode/cmake-configure.log 2>&1 || {
	cat /tmp/opencode/cmake-configure.log
	exit 1
}
grep -E "CMAKE_BUILD_TYPE|onnxruntime C\+\+" /tmp/opencode/cmake-configure.log || true
cmake --build build --target openwarp_vulkan_shaders openwarp-vulkan-test openwarp

echo "---"
echo "build type : $(grep CMAKE_BUILD_TYPE:STRING build/CMakeCache.txt | cut -d= -f2)"
if grep -q "ONNXRUNTIME_LIB:FILEPATH=.*libonnxruntime" build/CMakeCache.txt; then
	echo "ORT C++    : $(grep ONNXRUNTIME_LIB:FILEPATH build/CMakeCache.txt | cut -d= -f2)"
else
	echo "ORT C++    : not found (NN fill uses mock backend)"
fi
git log --oneline -1
