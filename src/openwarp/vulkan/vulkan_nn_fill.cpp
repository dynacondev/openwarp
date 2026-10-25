// Openwarp Vulkan NN hole fill implementation.
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan_nn_fill.hpp"

#ifdef OPENWARP_HAS_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

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
	          << " out=" << ort_->outputName << ")" << std::endl;
#else
	useMock_ = true;
	std::cout << "[nn-fill] mock backend (no ORT C++ at build time): " << modelPath
	          << " (3x3 box blur of input RGB; plumbing + hard-mask composite are identical)"
	          << std::endl;
#endif
	loaded_ = true;
	modelPath_ = modelPath;
	return true;
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
		int y0 = (int)std::floor(fy);
		float ty = fy - y0;
		y0 = std::min(std::max(y0, 0), sh - 1);
		int y1 = std::min(y0 + 1, sh - 1);
		for (int x = 0; x < dw; x++) {
			float fx = (x + 0.5f) * sw / dw - 0.5f;
			int x0 = (int)std::floor(fx);
			float tx = fx - x0;
			x0 = std::min(std::max(x0, 0), sw - 1);
			int x1 = std::min(x0 + 1, sw - 1);
			float v00 = src[(size_t)y0 * sw + x0];
			float v10 = src[(size_t)y0 * sw + x1];
			float v01 = src[(size_t)y1 * sw + x0];
			float v11 = src[(size_t)y1 * sw + x1];
			dst[(size_t)y * dw + x] =
			    (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
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
	const uint8_t* prev = (prevFilled_.size() == rgba.size() && prevW_ == w && prevH_ == h)
	                          ? prevFilled_.data()
	                          : nullptr;
	std::vector<float> input((size_t)kInCh * kNet * kNet);
	BuildInput(w, h, rgba.data(), splat, prev, input.data());
	std::vector<float> pred((size_t)kOutCh * kNet * kNet);
	if (!RunInfer(input.data(), pred.data()))
		return false; // already latched + logged; rgba untouched.

	// Upsample each predicted channel to full res, then hard-mask
	// composite: holes take the clamped prediction, valid pixels pass
	// through bit-exact (never smeared by the net).
	std::vector<float> up((size_t)w * h);
	for (int c = 0; c < kOutCh; c++) {
		UpsampleBilinear(pred.data() + (size_t)c * kNet * kNet, kNet, kNet, up.data(), w, h);
		for (uint32_t y = 0; y < h; y++) {
			for (uint32_t x = 0; x < w; x++) {
				size_t i = (size_t)y * w + x;
				bool hole = splat ? (splat[i] == kSplatEmpty) : (rgba[4 * i + 3] == 0);
				if (hole && (uint32_t)c == 0) {
					// First touch of this hole pixel: set alpha once.
					rgba[4 * i + 3] = 255;
				}
				if (hole) {
					float v = Clamp01(up[i]);
					rgba[4 * i + c] = (uint8_t)(v * 255.0f + 0.5f);
				}
			}
		}
	}
	prevFilled_ = rgba;
	prevW_ = w;
	prevH_ = h;
	return true;
}

} // namespace openwarp_vk
