// Openwarp Vulkan NN hole fill implementation.
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan_nn_fill.hpp"

#ifdef OPENWARP_HAS_ORT
#include <onnxruntime_cxx_api.h>
#endif
#ifdef OPENWARP_HAS_TORCH
#include <torch/script.h>
#ifdef OPENWARP_HAS_TORCH
#include <ATen/Parallel.h> // at::set_num_threads / set_num_interop_threads
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#if defined(OPENWARP_HAS_ORT) && defined(__linux__)
// dlfcn (dladdr/RTLD_DEFAULT) + glob need _GNU_SOURCE (set in CMake for
// the ORT build); sys/stat filters glob hits to directories.
#include <dlfcn.h>
#include <glob.h>
#include <sys/stat.h>
#endif
#include <cassert>

namespace openwarp_vk {

namespace {

constexpr int kNet = VulkanNnFill::kNetSize;

float Clamp01(float v)
{
	if (v < 0.0f)
		return 0.0f;
	if (v > 1.0f)
		return 1.0f;
	return v;
}

double NowMs()
{
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// Single-pixel bilinear sample, edge-clamped. UpsampleBilinear and the
// fused Fill composite both go through here so bytes agree exactly.
float BilinearAt(const float* src, int sw, int sh, float fx, float fy)
{
	int x0 = (int)std::floor(fx);
	float tx = fx - x0;
	x0 = std::min(std::max(x0, 0), sw - 1);
	int x1 = std::min(x0 + 1, sw - 1);
	int y0 = (int)std::floor(fy);
	float ty = fy - y0;
	y0 = std::min(std::max(y0, 0), sh - 1);
	int y1 = std::min(y0 + 1, sh - 1);
	float v00 = src[(size_t)y0 * sw + x0];
	float v10 = src[(size_t)y0 * sw + x1];
	float v01 = src[(size_t)y1 * sw + x0];
	float v11 = src[(size_t)y1 * sw + x1];
	return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

} // namespace

#if defined(OPENWARP_HAS_ORT) && defined(__linux__)
namespace {

// CUDA/cuDNN user-space libs that libonnxruntime_providers_cuda.so needs
// at load time. Ground truth is the provider lib's own DT_NEEDED (CUDA 13
// era; readelf -d on the 1.31.0 gpu_cuda13 tarball):
//   libcublasLt.so.13, libcublas.so.13, libcurand.so.10, libcudart.so.13
// plus libcuda.so.1 (driver) and system libs (not our problem). Notably
// the provider lib carries NO RUNPATH/RPATH and does NOT DT_NEEDED
// libcudnn/libnvrtc, so same-dir symlinks alone never satisfy the loader
// — hence the absolute-path RTLD_GLOBAL preload below.
const char* kCudaDepLibs[] = {
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcurand.so.10",
    "libcudart.so.13",
};

// Directory holding the already-loaded libonnxruntime, with no env vars:
// resolve a known ORT export to its real definition, then dladdr it.
// (dlsym — NOT &OrtGetApiBase, which is this binary's PLT stub — so the
// address belongs to the provider library itself.)
std::string OrtLibDir()
{
	void* sym = dlsym(RTLD_DEFAULT, "OrtGetApiBase");
	if (!sym)
		return {};
	Dl_info info{};
	if (!dladdr(sym, &info) || !info.dli_fname)
		return {};
	std::string path = info.dli_fname;
	size_t slash = path.find_last_of('/');
	if (slash == std::string::npos)
		return {};
	return path.substr(0, slash);
}

void SplitDirs(const char* list, std::vector<std::string>& out)
{
	if (!list)
		return;
	std::string cur;
	for (const char* p = list;; ++p) {
		if (*p == ':' || *p == '\0') {
			if (!cur.empty())
				out.push_back(cur);
			cur.clear();
			if (*p == '\0')
				return;
		} else {
			cur += *p;
		}
	}
}

void GlobDirs(const char* pattern, std::vector<std::string>& out)
{
	glob_t found{};
	if (glob(pattern, 0, nullptr, &found) != 0)
		return;
	for (size_t i = 0; i < found.gl_pathc; ++i) {
		std::string d = found.gl_pathv[i];
		struct stat st{};
		if (stat(d.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
			while (!d.empty() && d.back() == '/')
				d.pop_back();
			if (!d.empty())
				out.push_back(d);
		}
	}
	globfree(&found);
}

// Where to look for the CUDA dep libs, in order: (1) the dir holding the
// loaded libonnxruntime (no env needed — covers the tarball lib dir and
// any symlinks placed there), (2) every LD_LIBRARY_PATH entry, (3) the
// active venv's nvidia/*/lib dirs, then ../ml/.venv-style relative
// fallbacks (globbed, last resort: depend on CWD + venv layout).
std::vector<std::string> CudaDepDirs()
{
	std::vector<std::string> dirs;
	std::string ortDir = OrtLibDir();
	if (!ortDir.empty())
		dirs.push_back(ortDir);
	SplitDirs(std::getenv("LD_LIBRARY_PATH"), dirs);
	if (const char* venv = std::getenv("VIRTUAL_ENV")) {
		std::string pat = std::string(venv) + "/lib/python*/site-packages/nvidia/*/lib";
		GlobDirs(pat.c_str(), dirs);
	}
	GlobDirs("ml/.venv/lib/python*/site-packages/nvidia/*/lib", dirs);
	GlobDirs("../ml/.venv/lib/python*/site-packages/nvidia/*/lib", dirs);
	std::vector<std::string> uniq;
	for (auto& d : dirs) {
		if (!d.empty() && std::find(uniq.begin(), uniq.end(), d) == uniq.end())
			uniq.push_back(d);
	}
	return uniq;
}

} // namespace
#endif

#ifdef OPENWARP_HAS_ORT
namespace {

// ORT lib dir for the "where to symlink" hint. Real implementation above
// (Linux); empty elsewhere (message names the dir generically instead).
#if !defined(__linux__)
std::string OrtLibDir()
{
	return {};
}
#endif

// dlopen() every CUDA dep basename from the first candidate dir that holds
// it, RTLD_NOW|RTLD_GLOBAL, BEFORE the CUDA EP session is created, so the
// provider .so's own DT_NEEDED resolves to the preloaded objects even when
// the loader search path is broken (gamemode/env-sanitize wiping
// LD_LIBRARY_PATH on the 3060 host). Handles stay open for process
// lifetime (standard preload practice, cf. Python ort.preload_dlls()).
// Returns basenames missing everywhere — reported only if the CUDA EP
// then actually fails, so a CPU-only box never warns about CUDA libs it
// never needed. Each attempt is logged iff OPENWARP_NN_LOG=1.
std::vector<std::string> PreloadCudaDeps()
{
#if defined(OPENWARP_HAS_ORT) && defined(__linux__)
	const bool verbose = VulkanNnFill::LogVerbose();
	std::vector<std::string> dirs = CudaDepDirs();
	if (verbose) {
		std::cout << "[nn-fill] cuda-dep search dirs (" << dirs.size() << "):";
		for (auto& d : dirs)
			std::cout << " " << d;
		std::cout << std::endl;
	}
	static std::vector<void*> held; // NOLINT: intentionally never dlclosed
	std::vector<std::string> missing;
	const size_t nLibs = sizeof(kCudaDepLibs) / sizeof(kCudaDepLibs[0]);
	for (size_t li = 0; li < nLibs; ++li) {
		const char* lib = kCudaDepLibs[li];
		bool have = false;
		for (auto& d : dirs) {
			std::string abs = d + "/" + lib;
			dlerror(); // reset: a stale message must not be misattributed
			void* h = dlopen(abs.c_str(), RTLD_NOW | RTLD_GLOBAL);
			if (verbose)
				std::cout << "[nn-fill] cuda-dep " << abs << (h ? ": ok" : ": miss")
				          << std::endl;
			if (h) {
				held.push_back(h);
				have = true;
				break;
			}
		}
		if (!have)
			missing.emplace_back(lib);
	}
	return missing;
#else
	return {};
#endif
}

} // namespace
#endif // OPENWARP_HAS_ORT

#ifdef OPENWARP_HAS_ORT
struct VulkanNnFill::OrtState {
	Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "openwarp-nn-fill"};
	Ort::Session session{nullptr};
	Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
	std::string inputName;
	std::string outputName;
};
#endif

#ifdef OPENWARP_HAS_TORCH
struct VulkanNnFill::TorchState {
	torch::jit::Module module;
	torch::Device device{torch::kCPU};
};
#endif

VulkanNnFill::VulkanNnFill() = default;

VulkanNnFill::~VulkanNnFill() = default;

bool VulkanNnFill::Fail(std::string* error, const std::string& msg)
{
	if (error)
		*error = msg;
	lastError_ = msg;
	return false;
}

void VulkanNnFill::DisableToBlack(const std::string& why)
{
	disabled_ = true;
	lastError_ = why;
	std::cerr << "[nn-fill] disabled to black: " << why << std::endl;
}

bool VulkanNnFill::Load(const std::string& modelPath, std::string* error)
{
	loaded_ = false;
	disabled_ = false;
	modelPath_.clear();
	lastError_.clear();
	providers_ = "mock";
	ortThreads_ = 0;
	torchThreads_ = 0;
#ifndef NDEBUG
	// Loud by design: the NN-fill CPU loops (downsample/upsample/
	// composite) run ~10x slower under -O0, which reads as "NN runs at
	// 8fps" on a perf host that silently stayed Debug. Once per process.
	{
		static bool warned = false;
		if (!warned) {
			warned = true;
			std::cerr << "[nn-fill] WARNING: unoptimized Debug build (NDEBUG off): "
			             "NN-fill CPU loops run ~10x slower. Reconfigure Release "
			             "on perf hosts (see docs/nn-fill-3060-runbook.md)."
			          << std::endl;
		}
	}
#endif
#ifdef OPENWARP_HAS_ORT
	ort_.reset();
#endif
#ifdef OPENWARP_HAS_TORCH
	torch_.reset();
#endif

	// Twin resolution: a ".onnx" path also probes its ".pt" sibling for
	// the Torch backend and vice versa, so every caller layout (env
	// OPENWARP_NN_MODEL, build default, relative ml/weights paths) works
	// with either backend compiled in. Unknown extensions try as-given.
	std::string onnxPath = modelPath;
	std::string ptPath = modelPath;
	if (modelPath.size() > 5 && modelPath.compare(modelPath.size() - 5, 5, ".onnx") == 0) {
		ptPath = modelPath.substr(0, modelPath.size() - 5) + ".pt";
	} else if (modelPath.size() > 3 && modelPath.compare(modelPath.size() - 3, 3, ".pt") == 0) {
		onnxPath = modelPath.substr(0, modelPath.size() - 3) + ".onnx";
	}
	auto fileBytes = [](const std::string& p) -> long {
		if (p.empty())
			return -1;
		std::ifstream f(p, std::ios::binary | std::ios::ate);
		if (!f)
			return -1;
		std::streamsize n = f.tellg();
		return n < 0 ? -1 : (long)n;
	};
	long onnxBytes = fileBytes(onnxPath);
	long ptBytes = fileBytes(ptPath);
	if (onnxBytes < 0 && ptBytes < 0)
		return Fail(error, "cannot open NN model: " + modelPath);
	std::string servedPath = modelPath;
	bool served = false;

#ifdef OPENWARP_HAS_TORCH
	// Torch first: its CUDA stacking works on the 3060 host where ORT's
	// prebuilt CUDA EP kept failing loader-side. fp32 for now (fp16 is a
	// follow-up). The .to(kCPU) after forward is a blocking copy, so no
	// explicit stream sync is needed before the memcpy below.
	if (ptBytes >= 0) {
		if (ptBytes == 0)
			return Fail(error, "NN model is empty: " + ptPath);
		try {
			auto st = std::make_unique<TorchState>();
			st->module = torch::jit::load(ptPath);
			st->module.eval();
			unsigned hw = std::thread::hardware_concurrency();
			if (hw == 0)
				hw = 4;
			unsigned threads = hw;
			if (const char* te = std::getenv("OPENWARP_NN_THREADS"))
				threads = (unsigned)std::max(1, std::atoi(te));
			torchThreads_ = (int)threads;
			at::set_num_threads((int)threads);
			at::set_num_interop_threads(1);
			// Prefer CUDA when the linked LibTorch can actually move
			// the module there: .to(cuda) throws cleanly (c10::Error)
			// in CPU-only builds or without a driver, so this probes
			// usability — no torch::cuda::* link symbols needed beyond
			// core ATen, and a partial move is rolled back to CPU.
			st->device = torch::Device(torch::kCPU);
			bool useCuda = false;
			try {
				int ordinal = 0;
				if (const char* de = std::getenv("OPENWARP_NN_CUDA_DEVICE"))
					ordinal = std::max(0, std::atoi(de));
				st->module.to(torch::Device(torch::kCUDA, ordinal));
				st->device = torch::Device(torch::kCUDA, ordinal);
				useCuda = true;
			} catch (const std::exception& e) {
				try {
					st->module.to(torch::kCPU);
				} catch (...) {
				}
				std::cerr << "[nn-fill] Torch CUDA unavailable (" << e.what()
				          << "); using CPU." << std::endl;
			}
			torch_ = std::move(st);
			served = true;
			servedPath = ptPath;
			useMock_ = false;
			providers_ = useCuda ? "TORCH[cuda,cpu]" : "TORCH[cpu]";
			std::cout << "[nn-fill] Torch backend: " << ptPath
			          << " (device=" << (useCuda ? "cuda" : "cpu")
			          << " threads=" << torchThreads_ << " io=1x6x" << kNet << "x"
			          << kNet << "->1x3x" << kNet << "x" << kNet << ")" << std::endl;
		} catch (const std::exception& e) {
			torch_.reset();
			return Fail(error, std::string("Torch session failed for ") + ptPath + ": " + e.what());
		}
	}
#endif

#ifdef OPENWARP_HAS_ORT
	if (!served && onnxBytes >= 0) {
		if (onnxBytes == 0)
			return Fail(error, "NN model is empty: " + onnxPath);
		try {
			ort_ = std::make_unique<OrtState>();
			Ort::SessionOptions opts;
			opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
			opts.EnableMemPattern(); // keep the arena on: per-run allocs hurt
			// Latency path: one inter-op stream, intra-op pool from env
			// OPENWARP_NN_THREADS (default = core count). The net is ~8k
			// params; oversubscription only adds scheduling jitter, so the
			// knob stays visible and the chosen value is logged below.
			unsigned hw = std::thread::hardware_concurrency();
			if (hw == 0)
				hw = 4;
			unsigned threads = hw;
			if (const char* te = std::getenv("OPENWARP_NN_THREADS"))
				threads = (unsigned)std::max(1, std::atoi(te));
			ortThreads_ = (int)threads;
			opts.SetIntraOpNumThreads((int)threads);
			opts.SetInterOpNumThreads(1);
			// Prefer the CUDA EP when the linked ORT ships it (3060 host).
			// CPU Ort::Values still work (ORT inserts the copies); when the
			// build is CPU-only this throws and we fall through to CPU EP.
			// Before that, preload the provider's DT_NEEDED CUDA libs by
			// absolute path (RTLD_GLOBAL): the provider .so carries no
			// RUNPATH, so a sanitized LD_LIBRARY_PATH breaks its load even
			// when the files sit next to it or in the venv.
			bool cudaEp = false;
			std::vector<std::string> cudaDepsMissing = PreloadCudaDeps();
			try {
				OrtCUDAProviderOptions cudaOpts{};
				if (const char* de = std::getenv("OPENWARP_NN_CUDA_DEVICE"))
					cudaOpts.device_id = std::atoi(de);
				opts.AppendExecutionProvider_CUDA(cudaOpts);
				cudaEp = true;
			} catch (const Ort::Exception& e) {
				std::cerr << "[nn-fill] CUDA EP unavailable (" << e.what()
				          << "); using CPU EP." << std::endl;
				for (const std::string& lib : cudaDepsMissing) {
					std::string ortDir = OrtLibDir();
					std::cerr << "[nn-fill] cuda-dep missing: " << lib
					          << " (place/symlink it into "
					          << (ortDir.empty() ? "<the dir holding libonnxruntime>"
					                             : ortDir)
					          << ", or export LD_LIBRARY_PATH=<venv>/nvidia/*/lib)"
					          << std::endl;
				}
			}
			providers_ = cudaEp ? "ORT[cuda,cpu]" : "ORT[cpu]";
			ort_->session = Ort::Session(ort_->env, onnxPath.c_str(), opts);
			Ort::AllocatorWithDefaultOptions alloc;
			if (ort_->session.GetInputCount() != 1 || ort_->session.GetOutputCount() != 1)
				throw std::runtime_error("expected 1 input / 1 output");
			ort_->inputName = ort_->session.GetInputNameAllocated(0, alloc).get();
			ort_->outputName = ort_->session.GetOutputNameAllocated(0, alloc).get();
			Ort::TypeInfo inType = ort_->session.GetInputTypeInfo(0);
			Ort::TypeInfo outType = ort_->session.GetOutputTypeInfo(0);
			auto inShape = inType.GetTensorTypeAndShapeInfo().GetShape();
			auto outShape = outType.GetTensorTypeAndShapeInfo().GetShape();
			ONNXTensorElementDataType inEl =
			    inType.GetTensorTypeAndShapeInfo().GetElementType();
			ONNXTensorElementDataType outEl =
			    outType.GetTensorTypeAndShapeInfo().GetElementType();
			if (inShape.size() != 4 || inShape[0] != 1 || inShape[1] != kInCh)
				throw std::runtime_error("input must be 1x6xHxW");
			if (outShape.size() != 4 || outShape[0] != 1 || outShape[1] != kOutCh)
				throw std::runtime_error("output must be 1x3xHxW");
			// Spatial dims must be dynamic (-1) or exactly the net size.
			// Rejecting a fixed-shape mismatch here (loud, at load) instead
			// of throwing per-warp after ORT already allocated: RunInfer
			// feeds exactly 1x6x512x512 and copies out exactly 1x3x512x512.
			for (int i = 2; i < 4; i++) {
				if ((inShape[i] != -1 && inShape[i] != kNet) ||
				    (outShape[i] != -1 && outShape[i] != kNet))
					throw std::runtime_error("model H/W must be dynamic or " +
					                         std::to_string(kNet));
			}
			if (inEl != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
			    outEl != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
				throw std::runtime_error("input/output must be float32");
			// H/W stay dynamic (export uses dynamic axes); only batch and
			// channel extents are pinned above.
		} catch (const std::exception& e) {
			ort_.reset();
			return Fail(error, std::string("ORT session failed for ") + onnxPath + ": " + e.what());
		}
		useMock_ = false;
		served = true;
		servedPath = onnxPath;
		std::cout << "[nn-fill] ORT backend: " << onnxPath << " (in=" << ort_->inputName
		          << " out=" << ort_->outputName << " providers=" << providers_
		          << " threads=" << ortThreads_ << " io=1x6x" << kNet << "x" << kNet << "->1x3x"
		          << kNet << "x" << kNet << ")" << std::endl;
	} // if (!served && onnxBytes >= 0)
#endif
	if (!served) {
		// Mock: no twin the compiled backends could serve (or no real
		// backend compiled at all). The as-given file still had to be
		// non-empty (a twin alone also satisfies the exists check
		// above), so a missing/empty-weights deploy fails loudly
		// instead of silently filling with the mock.
		if (fileBytes(modelPath) == 0)
			return Fail(error, "NN model is empty: " + modelPath);
		useMock_ = true;
#if defined(OPENWARP_HAS_TORCH) && defined(OPENWARP_HAS_ORT)
		std::cout << "[nn-fill] mock backend (no .pt/.onnx twin usable by Torch/ORT): ";
#elif defined(OPENWARP_HAS_TORCH)
		std::cout << "[nn-fill] mock backend (no .pt twin for Torch): ";
#elif defined(OPENWARP_HAS_ORT)
		std::cout << "[nn-fill] mock backend (no .onnx twin for ORT): ";
#else
		std::cout << "[nn-fill] mock backend (no Torch/ORT C++ at build time): ";
#endif
		std::cout << modelPath
		          << " (providers=mock threads=n/a io=1x6x" << kNet << "x" << kNet << "->1x3x"
		          << kNet << "x" << kNet
		          << "; 3x3 box blur of input RGB; plumbing + hard-mask composite are identical)"
		          << std::endl;
	}
	// Hot-path buffers are sized once here (not per warp): steady-state
	// Fill never reallocates, so no ORT Value can ever alias a buffer
	// across a reallocation. bad_alloc surfaces as a clean load error.
	try {
		scratchInput_.resize((size_t)kInCh * kNet * kNet);
		scratchPred_.resize((size_t)kOutCh * kNet * kNet);
	} catch (const std::exception& e) {
		return Fail(error, std::string("NN scratch alloc failed: ") + e.what());
	}
	loaded_ = true;
	modelPath_ = servedPath;
	return true;
}

std::string VulkanNnFill::BackendLabel() const
{
#ifdef OPENWARP_HAS_TORCH
	if (torch_)
		return providers_;
#endif
#ifdef OPENWARP_HAS_ORT
	if (!useMock_)
		return providers_;
#endif
	return "mock";
}

bool VulkanNnFill::LogVerbose()
{
	const char* e = std::getenv("OPENWARP_NN_LOG");
	return e && std::atoi(e) >= 1;
}

void VulkanNnFill::RollAccumulate(const FillStats& s)
{
	if (!rollInit_) {
		rollStart_ = std::chrono::steady_clock::now();
		rollInit_ = true;
	}
	rollFills_++;
	rollEarly_ += s.earlyOut ? 1 : 0;
	rollScanMs_ += s.scanMs;
	rollBuildMs_ += s.buildMs;
	rollInferMs_ += s.inferMs;
	rollCompMs_ += s.compositeMs;
	rollHole_ += s.holeFrac;
}

void VulkanNnFill::RollMaybeLog()
{
	using clock = std::chrono::steady_clock;
	double dt =
	    std::chrono::duration<double, std::milli>(clock::now() - rollStart_).count();
	if (dt < 1000.0)
		return;
	uint64_t n = rollFills_ ? rollFills_ : 1;
	std::cout << "[nn-fill] summary: warps=" << rollFills_ << " early=" << rollEarly_
	          << " hole=" << (100.0 * rollHole_ / n) << "%"
	          << " | ms/warp scan=" << rollScanMs_ / n << " build=" << rollBuildMs_ / n
	          << " infer=" << rollInferMs_ / n << " comp=" << rollCompMs_ / n << " total="
	          << (rollScanMs_ + rollBuildMs_ + rollInferMs_ + rollCompMs_) / n
	          << " | backend=" << BackendLabel() << " io=1x6x" << kNet << "x" << kNet
	          << "->1x3x" << kNet << "x" << kNet << std::endl;
	rollFills_ = rollEarly_ = 0;
	rollScanMs_ = rollBuildMs_ = rollInferMs_ = rollCompMs_ = rollHole_ = 0.0;
	rollStart_ = clock::now();
}

bool VulkanNnFill::RunInfer(const float* input6ch, float* pred3ch)
{
	if (!Available() || !input6ch || !pred3ch) {
		DisableToBlack("RunInfer: not loaded or null tensor");
		return false;
	}
	const size_t inCount = (size_t)kInCh * kNet * kNet;
	const size_t outCount = (size_t)kOutCh * kNet * kNet;
#ifdef OPENWARP_HAS_TORCH
	if (torch_) {
		try {
			// from_blob aliases input6ch in place (no copy on CPU; a
			// staging copy on CUDA via the .to below). The device
			// tensor and the outputs all die at scope end, before the
			// memcpy'd result is validated — the same aliasing contract
			// as the ORT Value path. The .to(kCPU) is a blocking copy,
			// so no explicit stream sync is needed before data_ptr().
			// InferenceMode (not just NoGrad): no autograd bookkeeping.
			c10::InferenceMode guard;
			torch::Tensor input =
			    torch::from_blob(const_cast<float*>(input6ch),
			                     {1, kInCh, kNet, kNet}, torch::kFloat32)
			        .to(torch_->device);
			torch::Tensor out = torch_->module.forward({input})
			                        .toTensor()
			                        .to(torch::kCPU, torch::kFloat32, false)
			                        .contiguous();
			if (out.numel() != (int64_t)outCount)
				throw std::runtime_error("output element count mismatch");
			const float* src = out.data_ptr<float>();
			if (!src)
				throw std::runtime_error("null output tensor data");
			std::memcpy(pred3ch, src, outCount * sizeof(float));
			for (size_t i = 0; i < outCount; i++) {
				if (!std::isfinite(pred3ch[i]))
					throw std::runtime_error("non-finite network output");
			}
		} catch (const std::exception& e) {
			DisableToBlack(std::string("torch infer failed: ") + e.what());
			return false;
		}
		return true;
	}
#endif
#ifdef OPENWARP_HAS_ORT
	if (ort_) {
		try {
			// The Values below alias the caller buffers in place (no copy):
			// inTensor references input6ch until the end of the scope, the
			// output Value owns its (arena) buffer until outs dies. Both die
			// before the memcpy'd result is validated, and neither outlives
			// this call — so Fill's scratch can never be reallocated from
			// under a live Value (it is resized only before BuildInput, when
			// no Value exists). The inner scope makes that lifetime explicit.
			std::array<int64_t, 4> inShape{1, kInCh, kNet, kNet};
			assert(inCount == (size_t)kInCh * kNet * kNet);
			assert(outCount == (size_t)kOutCh * kNet * kNet);
			{
				Ort::Value inTensor = Ort::Value::CreateTensor<float>(
				    ort_->mem, const_cast<float*>(input6ch), inCount, inShape.data(), 4);
				const char* inNames[] = {ort_->inputName.c_str()};
				const char* outNames[] = {ort_->outputName.c_str()};
				std::vector<Ort::Value> outs =
				    ort_->session.Run(Ort::RunOptions{nullptr}, inNames, &inTensor, 1,
				                      outNames, 1);
				if (outs.size() != 1)
					throw std::runtime_error("ORT returned != 1 output tensor");
				Ort::Value& o = outs[0];
				if (!o.IsTensor())
					throw std::runtime_error("output is not a tensor");
				auto info = o.GetTensorTypeAndShapeInfo();
				if (info.GetElementCount() != outCount)
					throw std::runtime_error("output element count mismatch");
				const float* src = o.GetTensorData<float>();
				if (!src)
					throw std::runtime_error("null output tensor data");
				std::memcpy(pred3ch, src, outCount * sizeof(float));
			} // inTensor + outs die here: no live alias remains past this point.
			for (size_t i = 0; i < outCount; i++) {
				if (!std::isfinite(pred3ch[i]))
					throw std::runtime_error("non-finite network output");
			}
		} catch (const std::exception& e) {
			DisableToBlack(std::string("infer failed: ") + e.what());
			return false;
		}
		return true;
	} // if (ort_)
#endif
	// Mock session: per-channel 3x3 box blur of input RGB (ch 0..2).
	// The blur-init weights compute gate(2) * box blur through identity
	// body/head stages, so this matches up to kGateScale; verification
	// gate-compensates exactly like ml/export_blur.py.
	BoxBlur3x3(input6ch, pred3ch, kNet, kNet, kOutCh);
	(void)inCount;
	(void)outCount;
	return true;
}

float VulkanNnFill::Luma(uint8_t r, uint8_t g, uint8_t b)
{
	return (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
}

void VulkanNnFill::BoxBlur3x3(const float* src, float* dst, int w, int h, int channels)
{
	for (int c = 0; c < channels; c++) {
		const float* s = src + (size_t)c * w * h;
		float* d = dst + (size_t)c * w * h;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				float acc = 0.0f;
				for (int dy = -1; dy <= 1; dy++) {
					int yy = std::min(std::max(y + dy, 0), h - 1);
					for (int dx = -1; dx <= 1; dx++) {
						int xx = std::min(std::max(x + dx, 0), w - 1);
						acc += s[(size_t)yy * w + xx];
					}
				}
				d[(size_t)y * w + x] = acc / 9.0f;
			}
		}
	}
}

void VulkanNnFill::UpsampleBilinear(const float* src, int sw, int sh, float* dst, int dw, int dh)
{
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / dh - 0.5f;
		for (int x = 0; x < dw; x++) {
			float fx = (x + 0.5f) * sw / dw - 0.5f;
			dst[(size_t)y * dw + x] = BilinearAt(src, sw, sh, fx, fy);
		}
	}
}

void VulkanNnFill::BuildInput(uint32_t w, uint32_t h, const uint8_t* rgba, const uint32_t* splat,
                              const uint8_t* prevRgbaOrNull, float* input6ch)
{
	const size_t plane = (size_t)kNet * kNet;
	float* ch[kInCh];
	for (int c = 0; c < kInCh; c++)
		ch[c] = input6ch + (size_t)c * plane;
	for (int hy = 0; hy < kNet; hy++) {
		uint32_t y0 = (uint32_t)((uint64_t)hy * h / kNet);
		uint32_t y1 = (uint32_t)((uint64_t)(hy + 1) * h / kNet);
		if (y1 <= y0)
			y1 = y0 + 1;
		if (y1 > h)
			y1 = h;
		for (int hx = 0; hx < kNet; hx++) {
			uint32_t x0 = (uint32_t)((uint64_t)hx * w / kNet);
			uint32_t x1 = (uint32_t)((uint64_t)(hx + 1) * w / kNet);
			if (x1 <= x0)
				x1 = x0 + 1;
			if (x1 > w)
				x1 = w;
			double r = 0, g = 0, b = 0, depth = 0, prev = 0;
			uint32_t total = 0, valid = 0;
			for (uint32_t y = y0; y < y1; y++) {
				for (uint32_t x = x0; x < x1; x++) {
					size_t i = (size_t)y * w + x;
					// Holes are (0,0,0,0) out of warp_fill.comp mode 0;
					// valid-but-black pixels keep alpha 255, and the
					// splat plane (when present) is exact either way.
					bool hole = splat ? (splat[i] == kSplatEmpty) : (rgba[4 * i + 3] == 0);
					r += rgba[4 * i + 0];
					g += rgba[4 * i + 1];
					b += rgba[4 * i + 2];
					if (!hole) {
						valid++;
						if (splat)
							depth += (double)splat[i] / kDepthScale;
					}
					if (prevRgbaOrNull)
						prev += Luma(prevRgbaOrNull[4 * i + 0], prevRgbaOrNull[4 * i + 1],
						             prevRgbaOrNull[4 * i + 2]);
					total++;
				}
			}
			size_t o = (size_t)hy * kNet + hx;
			double inv = total ? 1.0 / (total * 255.0) : 0.0;
			ch[0][o] = (float)(r * inv);
			ch[1][o] = (float)(g * inv);
			ch[2][o] = (float)(b * inv);
			ch[3][o] = valid ? (float)(depth / valid) : 0.0f;
			ch[4][o] = (valid < total) ? 1.0f : 0.0f;
			ch[5][o] = (prevRgbaOrNull && total) ? (float)(prev / total) : 0.0f;
		}
	}
}

bool VulkanNnFill::Fill(uint32_t w, uint32_t h, std::vector<uint8_t>& rgba, const uint32_t* splat)
{
	if (!Available())
		return false;
	if (w == 0 || h == 0 || rgba.size() != (size_t)w * h * 4) {
		DisableToBlack("Fill: bad frame size");
		return false;
	}
	const double t0 = NowMs();
	const size_t px = (size_t)w * h;

	// 1) Exact hole pre-scan: 1M cheap compares. Doubles as the hole-
	// fraction stat and the early-out gate. This stays exact (not
	// strided): an estimate could miss small holes and leave them black.
	uint64_t holes = 0;
	if (splat) {
		for (size_t i = 0; i < px; i++)
			holes += (splat[i] == kSplatEmpty) ? 1 : 0;
	} else {
		for (size_t i = 0; i < px; i++)
			holes += (rgba[4 * i + 3] == 0) ? 1 : 0;
	}
	const double tScan = NowMs();

	const uint8_t* prev = (prevFilled_.size() == rgba.size() && prevW_ == w && prevH_ == h)
	                          ? prevFilled_.data()
	                          : nullptr;
	// prev aliases prevFilled_ (read-only, consumed by BuildInput below)
	// and splat aliases the caller's plane (read-only, whole call): both
	// stay valid because neither vector is resized while in use — the
	// only writes (scratch resizes above, prevFilled_ copy at the end)
	// happen strictly outside the read windows.
	FillStats s;
	s.holes = holes;
	s.holeFrac = px ? (double)holes / (double)px : 0.0;
	s.scanMs = tScan - t0;

	if (holes == 0) {
		// Early-out: nothing to fill, so the frame passes through
		// bit-exact. The prev cache still advances (filled == input),
		// exactly as if the net had run on a hole-free frame. Skips
		// BuildInput + infer + composite entirely. The gate is exactly
		// zero, not "~0": a small nonzero threshold would leave visible
		// black specks and make output input-order-dependent.
		s.earlyOut = true;
		prevFilled_ = rgba;
		prevW_ = w;
		prevH_ = h;
		s.totalMs = NowMs() - t0;
		lastStats_ = s;
		RollAccumulate(s);
		RollMaybeLog();
		if (LogVerbose())
			std::cout << "[nn-fill] warp early-out (holes=0) scan=" << s.scanMs << "ms"
			          << std::endl;
		return true;
	}

	// Reused scratch: sized once at Load (hot path never reallocates).
	// The conditional resize stays as a no-op safety net. It runs here,
	// BEFORE BuildInput creates any ORT Value aliasing these buffers
	// (see RunInfer), so no realloc-while-aliased is possible.
	const size_t inCount = (size_t)kInCh * kNet * kNet;
	const size_t outCount = (size_t)kOutCh * kNet * kNet;
	if (scratchInput_.size() != inCount)
		scratchInput_.resize(inCount);
	if (scratchPred_.size() != outCount)
		scratchPred_.resize(outCount);
	assert(scratchInput_.size() == inCount && scratchPred_.size() == outCount);
	BuildInput(w, h, rgba.data(), splat, prev, scratchInput_.data());
	const double tBuild = NowMs();
	if (!RunInfer(scratchInput_.data(), scratchPred_.data()))
		return false; // already latched + logged; rgba untouched.
	const double tInfer = NowMs();

	// Fused upsample+composite, one pass: only hole pixels sample the 3
	// pred planes (same BilinearAt formula as UpsampleBilinear, so bytes
	// are identical to the old 3-plane path). Valid pixels pass through
	// bit-exact, never smeared by the net.
	for (uint32_t y = 0; y < h; y++) {
		float fy = (y + 0.5f) * kNet / h - 0.5f;
		for (uint32_t x = 0; x < w; x++) {
			size_t i = (size_t)y * w + x;
			bool hole = splat ? (splat[i] == kSplatEmpty) : (rgba[4 * i + 3] == 0);
			if (!hole)
				continue;
			float fx = (x + 0.5f) * kNet / w - 0.5f;
			for (int c = 0; c < kOutCh; c++) {
				const float* plane = scratchPred_.data() + (size_t)c * kNet * kNet;
				float v = BilinearAt(plane, kNet, kNet, fx, fy);
				rgba[4 * i + c] = (uint8_t)(Clamp01(v) * 255.0f + 0.5f);
			}
			rgba[4 * i + 3] = 255;
		}
	}
	const double tComp = NowMs();
	prevFilled_ = rgba;
	prevW_ = w;
	prevH_ = h;
	s.buildMs = tBuild - tScan;
	s.inferMs = tInfer - tBuild;
	s.compositeMs = tComp - tInfer;
	s.totalMs = tComp - t0;
	lastStats_ = s;
	RollAccumulate(s);
	RollMaybeLog();
	if (LogVerbose())
		std::cout << "[nn-fill] warp scan=" << s.scanMs << "ms build=" << s.buildMs << "ms infer="
		          << s.inferMs << "ms comp=" << s.compositeMs << "ms total=" << s.totalMs
		          << "ms holes=" << s.holes << " (" << (100.0 * s.holeFrac)
		          << "%) backend=" << BackendLabel() << std::endl;
	return true;
}

} // namespace openwarp_vk
