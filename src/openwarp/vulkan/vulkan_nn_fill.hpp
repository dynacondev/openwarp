#pragma once
// Openwarp Vulkan NN hole fill (blur-init TinyFillNet inference path).
//
// Net contract (see ml/fill_net.py, READ ONLY): 6ch NCHW float32 0..1 at
// 512x512 in (warped RGB with holes=0, splat depth, hole mask, prev
// filled luma reprojected) -> RGB out. The caller composites with a hard
// mask (out = mask ? pred : warped) so valid pixels stay bit-exact.
// Frame is 1024x1024 (OpenwarpApplication WIDTH/HEIGHT); the prev frame
// is the last presented/filled frame (free-running, no teacher forcing).
//
// Threading: VulkanNnFill is NOT thread-safe — Fill/RunInfer run on the
// single warp thread. ORT intra-op threads (OPENWARP_NN_THREADS) are
// internal to one session Run and always joined before RunInfer returns,
// so reusing the scratch/prev buffers across warps is race-free by
// construction (sequential reuse, never concurrent).
//
// Dependency-light: the TorchScript C++ API (LibTorch) is used when
// OPENWARP_HAS_TORCH is set (CMake defines it when find_package(Torch)
// succeeds), the ONNX Runtime C++ API only when OPENWARP_HAS_ORT is set.
// Backend priority when several are compiled in: Torch > ORT > mock.
// Without either backend the session is a mock: a 3x3 box blur of
// the input RGB channels. That matches the committed blur-init weights
// up to the gate sigmoid(2)=0.881 scale factor, so verification
// gate-compensates exactly like ml/export_blur.py, and plumbing, shapes
// and the hard-mask composite are proven identically with or without
// a real backend.
//
// Failure safety: any load/infer error latches disabled (Available() ==
// false) with a stderr line; the caller keeps the black-hole frame, so a
// broken model can never paint garbage.
//
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace openwarp_vk {

class VulkanNnFill {
public:
	static constexpr int kNetSize = 512;
	static constexpr int kInCh = 6;
	static constexpr int kOutCh = 3;
	static constexpr uint32_t kSplatEmpty = 0xffffffffu;
	static constexpr float kDepthScale = 16777215.0f;
	// Gate bias +2.0 in fill_net.py attenuates the blur-init output by
	// sigmoid(2); the mock skips it, checks compensate (see above).
	static constexpr float kGateScale = 0.88079708f;

	VulkanNnFill();
	~VulkanNnFill();

	VulkanNnFill(const VulkanNnFill&) = delete;
	VulkanNnFill& operator=(const VulkanNnFill&) = delete;

	// Loads the model at modelPath. Accepts either twin extension: a
	// ".onnx" path also probes its ".pt" sibling for the Torch backend
	// and vice versa (Torch > ORT > mock when several could serve the
	// file). Even with no real backend this requires the file to exist
	// and be non-empty, so a missing-weights deploy fails loudly instead
	// of silently filling with the mock. Returns false with the reason
	// in *error (and LastError()) on failure.
	bool Load(const std::string& modelPath, std::string* error = nullptr);
	bool Available() const { return loaded_ && !disabled_; }
	// True when the mock backend serves: no real backend compiled in,
	// or no usable .pt/.onnx twin found at Load.
	bool UsingMock() const { return useMock_; }
	const std::string& ModelPath() const { return modelPath_; }
	const std::string& LastError() const { return lastError_; }

	// Per-warp timings from the last Fill() (ms, steady-clock deltas):
	// scan = hole-mask pre-scan, build = downsample to 1x6x512x512 input,
	// infer = RunInfer (Torch/ORT incl. any H2D/D2H, or mock box blur),
	// composite = bilinear upsample + hard-mask composite. total = Fill
	// wall time. holes/holeFrac = exact mask count. earlyOut = holes==0
	// so infer+composite were skipped (frame passes through bit-exact,
	// prev cache still advanced). Read via LastStats() for the app's
	// paced log; VulkanNnFill also prints its own 1 s rolling summary.
	struct FillStats {
		double scanMs = 0.0;
		double buildMs = 0.0;
		double inferMs = 0.0;
		double compositeMs = 0.0;
		double totalMs = 0.0;
		double holeFrac = 0.0;
		uint64_t holes = 0;
		bool earlyOut = false;
	};
	const FillStats& LastStats() const { return lastStats_; }
	// "mock", "ORT[cuda,cpu]" / "ORT[cpu]", or "TORCH[cuda,cpu]" /
	// "TORCH[cpu]" when the Torch backend served the .pt twin. Logged at
	// Load and on every rolling summary so the 3060 host can confirm
	// which backend actually ran there.
	std::string BackendLabel() const;
	// OPENWARP_NN_LOG=1 enables one line per warp (default 0 = rolling
	// 1 s summary only).
	static bool LogVerbose();

	// Latches the black fallback: Available() goes false, stderr line.
	// The caller keeps presenting the black-hole frame.
	void DisableToBlack(const std::string& why);

	// One inference: input 1x6x512x512 NCHW float -> pred 1x3x512x512.
	// Torch/ORT when built with them, else the mock box blur. False on
	// error (latches disabled via DisableToBlack).
	// Buffer contract: the ORT input Value / torch::from_blob tensor
	// reference input6ch in place (no copy) for the duration of the call,
	// so the caller must not resize, move or free either buffer until
	// RunInfer returns. (Fill resizes its scratch only before any Value
	// exists; RunInfer itself never reallocates.)
	bool RunInfer(const float* input6ch, float* pred3ch);

	// Full frame path: in-place fill of a w*h RGBA8 warp output whose
	// holes are (0,0,0,0) (what warp_fill.comp mode 0 produces). splat
	// is the w*h R32_UINT splat plane, or nullptr (hole = alpha==0
	// fallback, depth channel 0).
	// Splat lifetime: splat must point to at least w*h values that stay
	// valid for the whole (synchronous) call — e.g.
	// vulkanWarp->LastSplatPlane().data() from a coverage warp. No size
	// travels with a raw pointer, so the caller must check
	// plane.size() == w*h before passing splat (the app dispatch does)
	// and must not mutate the plane during the call.
	// Updates the cached prev-frame luma from the filled result. Any
	// failure returns false with rgba left untouched (caller presents
	// the black-hole frame).
	bool Fill(uint32_t w, uint32_t h, std::vector<uint8_t>& rgba, const uint32_t* splat);

	// Pure-CPU helpers shared by Fill and the headless harness.
	static float Luma(uint8_t r, uint8_t g, uint8_t b);
	// 3x3 box blur, edge-replicated, per channel (mock infer core; also
	// the reference for the blur-init check).
	static void BoxBlur3x3(const float* src, float* dst, int w, int h, int channels);
	// Bilinear upscale of one float plane.
	static void UpsampleBilinear(const float* src, int sw, int sh, float* dst, int dw, int dh);
	// 1024 (or any w*h) RGBA8 + splat -> 1x6x512x512 NCHW input tensor.
	// prevRgbaOrNull is the previous filled frame (same w*h) or nullptr
	// (first frame: prev-luma channel is 0).
	static void BuildInput(uint32_t w, uint32_t h, const uint8_t* rgba, const uint32_t* splat,
	                       const uint8_t* prevRgbaOrNull, float* input6ch);

private:
	bool Fail(std::string* error, const std::string& msg);

	bool loaded_ = false;
	bool disabled_ = false;
	bool useMock_ = true;
	std::string modelPath_;
	std::string lastError_;
	FillStats lastStats_;
	// Hot-path scratch (6.3 MiB + 3.1 MiB at kNet=512): sized once at
	// Load so steady-state Fill never mallocs. The conditional resize in
	// Fill is a no-op safety net after that. Values created in RunInfer
	// alias these buffers, so sizing them anywhere a Value is alive
	// would be a use-after-realloc — the order is: resize (Fill, no
	// Value exists) -> BuildInput -> RunInfer (Values live and die
	// inside) -> composite. Never resize between CreateTensor and the
	// end of RunInfer.
	std::vector<float> scratchInput_;
	std::vector<float> scratchPred_;
	// Rolling 1 s summary accumulators (reset each print).
	uint64_t rollFills_ = 0;
	uint64_t rollEarly_ = 0;
	double rollScanMs_ = 0.0;
	double rollBuildMs_ = 0.0;
	double rollInferMs_ = 0.0;
	double rollCompMs_ = 0.0;
	double rollHole_ = 0.0;
	std::chrono::steady_clock::time_point rollStart_{};
	bool rollInit_ = false;
	void RollAccumulate(const FillStats& s);
	void RollMaybeLog();
	std::string providers_ = "mock"; // active backend EP list (Torch/ORT)
	int ortThreads_ = 0; // intra-op pool size configured (ORT only)
	int torchThreads_ = 0; // CPU thread pool size configured (Torch only)
	// Last filled frame (full-res RGBA8) for the prev-luma channel.
	std::vector<uint8_t> prevFilled_;
	uint32_t prevW_ = 0;
	uint32_t prevH_ = 0;
#ifdef OPENWARP_HAS_ORT
	struct OrtState;
	std::unique_ptr<OrtState> ort_; // RAII: session dies before env (see .cpp order)
#endif
#ifdef OPENWARP_HAS_TORCH
	struct TorchState;
	std::unique_ptr<TorchState> torch_; // RAII: module dies with us
#endif
};

} // namespace openwarp_vk
