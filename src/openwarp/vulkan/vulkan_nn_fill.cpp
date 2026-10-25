// Openwarp Vulkan NN hole fill implementation.
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan_nn_fill.hpp"

#ifdef OPENWARP_HAS_ORT
#include <onnxruntime_cxx_api.h>
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

#ifdef OPENWARP_HAS_ORT
struct VulkanNnFill::OrtState {
	Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "openwarp-nn-fill"};
	Ort::Session session{nullptr};
	Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
	std::string inputName;
	std::string outputName;
};
#endif

VulkanNnFill::VulkanNnFill() = default;

VulkanNnFill::~VulkanNnFill()
{
#ifdef OPENWARP_HAS_ORT
	delete ort_;
	ort_ = nullptr;
#endif
}

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
	delete ort_;
	ort_ = nullptr;
#endif

	std::ifstream f(modelPath, std::ios::binary | std::ios::ate);
	if (!f)
		return Fail(error, "cannot open NN model: " + modelPath);
	std::streamsize n = f.tellg();
	if (n <= 0)
		return Fail(error, "NN model is empty: " + modelPath);

#ifdef OPENWARP_HAS_ORT
	try {
		ort_ = new OrtState();
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
		bool cudaEp = false;
		try {
			OrtCUDAProviderOptions cudaOpts{};
			if (const char* de = std::getenv("OPENWARP_NN_CUDA_DEVICE"))
				cudaOpts.device_id = std::atoi(de);
			opts.AppendExecutionProvider_CUDA(cudaOpts);
			cudaEp = true;
		} catch (const Ort::Exception& e) {
			std::cerr << "[nn-fill] CUDA EP unavailable (" << e.what()
			          << "); using CPU EP." << std::endl;
		}
		providers_ = cudaEp ? "ORT[cuda,cpu]" : "ORT[cpu]";
		ort_->session = Ort::Session(ort_->env, modelPath.c_str(), opts);
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
		if (inEl != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
		    outEl != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
			throw std::runtime_error("input/output must be float32");
		// H/W stay dynamic (export uses dynamic axes); only batch and
		// channel extents are pinned above.
	} catch (const std::exception& e) {
		delete ort_;
		ort_ = nullptr;
		return Fail(error, std::string("ORT session failed for ") + modelPath + ": " + e.what());
	}
	useMock_ = false;
	std::cout << "[nn-fill] ORT backend: " << modelPath << " (in=" << ort_->inputName
	          << " out=" << ort_->outputName << " providers=" << providers_
	          << " threads=" << ortThreads_ << " io=1x6x" << kNet << "x" << kNet << "->1x3x"
	          << kNet << "x" << kNet << ")" << std::endl;
#else
	useMock_ = true;
	std::cout << "[nn-fill] mock backend (no ORT C++ at build time): " << modelPath
	          << " (providers=mock threads=n/a io=1x6x" << kNet << "x" << kNet << "->1x3x"
	          << kNet << "x" << kNet
	          << "; 3x3 box blur of input RGB; plumbing + hard-mask composite are identical)"
	          << std::endl;
#endif
	loaded_ = true;
	modelPath_ = modelPath;
	return true;
}

std::string VulkanNnFill::BackendLabel() const
{
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
#ifdef OPENWARP_HAS_ORT
	try {
		std::array<int64_t, 4> inShape{1, kInCh, kNet, kNet};
		std::array<int64_t, 4> outShape{1, kOutCh, kNet, kNet};
		Ort::Value inTensor = Ort::Value::CreateTensor<float>(
		    ort_->mem, const_cast<float*>(input6ch), inCount, inShape.data(), 4);
		const char* inNames[] = {ort_->inputName.c_str()};
		const char* outNames[] = {ort_->outputName.c_str()};
		std::vector<Ort::Value> outs =
		    ort_->session.Run(Ort::RunOptions{nullptr}, inNames, &inTensor, 1, outNames, 1);
		Ort::Value& o = outs[0];
		if (!o.IsTensor())
			throw std::runtime_error("output is not a tensor");
		auto info = o.GetTensorTypeAndShapeInfo();
		if (info.GetElementCount() != (int64_t)outCount)
			throw std::runtime_error("output element count mismatch");
		const float* src = o.GetTensorData<float>();
		std::memcpy(pred3ch, src, outCount * sizeof(float));
		for (size_t i = 0; i < outCount; i++) {
			if (!std::isfinite(pred3ch[i]))
				throw std::runtime_error("non-finite network output");
		}
	} catch (const std::exception& e) {
		DisableToBlack(std::string("infer failed: ") + e.what());
		return false;
	}
#else
	// Mock session: per-channel 3x3 box blur of input RGB (ch 0..2).
	// The blur-init weights compute gate(2) * box blur through identity
	// body/head stages, so this matches up to kGateScale; verification
	// gate-compensates exactly like ml/export_blur.py.
	BoxBlur3x3(input6ch, pred3ch, kNet, kNet, kOutCh);
	(void)inCount;
	(void)outCount;
#endif
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

	// Reused scratch (sizes change only with net/frame size): Fill never
	// mallocs in the hot path anymore (the old per-warp input/pred/up
	// vectors did three big allocations per warp).
	const size_t inCount = (size_t)kInCh * kNet * kNet;
	const size_t outCount = (size_t)kOutCh * kNet * kNet;
	if (scratchInput_.size() != inCount)
		scratchInput_.resize(inCount);
	if (scratchPred_.size() != outCount)
		scratchPred_.resize(outCount);
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
