#pragma once
// Openwarp Vulkan compute warp (Option B).
//
// Headless forward-splat / backward-pull warp for debugging the
// dynacon-vr depth reprojection path outside the full compositor:
// upload a color + standard 0..1 nonlinear depth pair, warp from a
// source pose to a target pose, download the result.
//
// Only core Vulkan 1.0 + a compute queue is required, so this runs on
// lavapipe for CI. No window/swapchain/graphics queue needed.
//
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include <cstdint>
#include <string>
#include <vector>

namespace openwarp_vk {

// Pose in the same convention as the warp math: position + quaternion
// (x, y, z, w). View matrix = inverse(Translation * Rotation).
struct Pose {
	float pos[3] = {0.0f, 0.0f, 0.0f};
	float quat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

// Column-major 4x4, 16 floats.
using Mat4 = float[16];

// GL-style asymmetric frustum for standard 0..1 nonlinear depth
// (matches Vulkan-rendered depth buffers: Vulkan [0,1] NDC was defined
// to coincide with remapped GL [-1,1]*0.5+0.5). Returns false on
// degenerate input. Column-major, column-vector convention.
bool FrustumGL(float angleLeft, float angleRight, float angleDown, float angleUp,
               float nearZ, float farZ, Mat4 out);

// Symmetric perspective (same convention), angles in degrees.
bool PerspectiveGL(float fovyDeg, float aspect, float nearZ, float farZ, Mat4 out);

// out = Translation(pos) * Rotation(quat), i.e. the camera (inverse
// view) matrix.
void CameraFromPose(const Pose& pose, Mat4 out);

// out = a * b (column-vector convention).
void Multiply(const Mat4 a, const Mat4 b, Mat4 out);

// out = inverse(m). Returns false on singular input.
bool Inverse(const Mat4 m, Mat4 out);

// Headless warp context for one fixed image size. Not thread-safe.
class VulkanWarp {
public:
	VulkanWarp() = default;
	~VulkanWarp() { Shutdown(); }

	VulkanWarp(const VulkanWarp&) = delete;
	VulkanWarp& operator=(const VulkanWarp&) = delete;

	// Loads warp_splat.spv / warp_pull.spv from spvDir, creates the
	// device, images and pipelines. Returns false on failure (with the
	// reason in *error when non-null). glRenderer (from
	// glGetString(GL_RENDERER), may be null) steers device selection
	// toward the GPU the GL context runs on so both APIs share one
	// device and frames never cross PCIe.
	bool Init(uint32_t width, uint32_t height, const std::string& spvDir, std::string* error = nullptr,
	          const char* glRenderer = nullptr);
	bool IsInitialized() const { return initialized_; }
	void Shutdown();

	// Upload a frame. Row 0 is the first texel row for BOTH planes
	// (color and depth must agree); color is RGBA8, depth is standard
	// 0..1 nonlinear float. Returns false on failure.
	bool Upload(const uint8_t* colorRGBA, const float* depth);

	// Hole-fill background weighting exponent: pull stores
	// (rgb*w, w) with w = pow(depth, p), so the mip average is biased
	// toward background (deep) pixels and the occluding foreground
	// stops smearing into disocclusions. The fill pass (warp_fill.comp
	// weightPow push constant, same value) re-applies pow(depth, p) to
	// the mode 0/1 averaging taps in full precision, restoring the bias
	// the 8-bit mip chain dilutes at coarse LODs. p = 0 restores the
	// uniform average. Tunable at runtime; takes effect on the next Warp().
	void SetFillWeightPow(float p) { fillWeightPow_ = p; }

	// Hole-fill variant for the fill pass (warp_fill.comp push
	// constant): 0 = legacy mip-chain average, 1 = depth-guided local
	// average with mip fallback, 2 = directional BG march with mip
	// fallback, 3 = background extrapolate with mip fallback,
	// 4 = bilateral gather with mip fallback. Takes effect on the next
	// Warp(). Out-of-range values fall back to 0 in-shader (unknown
	// modes must never produce garbage).
	void SetFillMode(int m) { fillMode_ = m; }

	// Background stretch for the pull pass (warp_pull.comp tune.y): 0
	// disables (holes keep black for the blur fill); >0 searches a
	// (2r+1)^2 window for the deepest valid donor and copies it
	// (mesh-warp look). Holes wider than the window still fall through
	// to the blur fallback. Takes effect on the next Warp().
	void SetStretchRadius(int r) { stretchRadius_ = r; }

	// Temporal BG cache (persistent half-res background layer,
	// orthogonal to every fillMode): when enabled, Warp() ingests
	// valid pull pixels into bgColor/bgDepth on static frames and
	// projects cached texels into holes BEFORE the fill pass, so the
	// cache sits underneath any fill mode (0..4). Disabled by default
	// (zero behavior change). Takes effect on the next Warp(). A fresh
	// Upload() invalidates the cache (re-primed on the next static
	// Warp). Extra GPU memory at 1024x1024: 2x 512x512x4 B = 2 MiB.
	void SetTemporalCacheEnabled(bool e) { temporalCacheEnabled_ = e; }
	bool TemporalCacheEnabled() const { return temporalCacheEnabled_; }

	// Warp the uploaded frame from (srcPose, srcProj) to
	// (dstPose, dstProj) and download the RGB8 result into outColor
	// (width*height*4 bytes). holeRadius > 0 enables hole filling:
	// SetFillMode selects the fill variant (0 legacy mip-chain
	// average, 1 depth-guided, 2 directional march, 3 BG extrapolate,
	// 4 bilateral gather, each with mip fallback), and
	// SetStretchRadius>0 additionally copies the deepest nearby donor
	// into holes first (mesh-warp look) with blur as the fallback; 0
	// leaves disocclusions black. Filled regions are the
	// validity-weighted mean (blurred, never stretched) except for
	// stretched texels, which are exact background copies.
	// srcProj/dstProj are PerspectiveGL/FrustumGL matrices; near/far
	// must match between the two (same convention as dynacon-vr).
	// countCoverage also downloads the splat plane and counts filled
	// texels (extra copy in the SAME submit, plus a CPU scan): pass
	// true in tests, sparingly (e.g. 1/60 warps) in the live path.
	bool Warp(const Pose& srcPose, const Mat4 srcProj,
	          const Pose& dstPose, const Mat4 dstProj,
	          float holeRadius, std::vector<uint8_t>& outColor,
	          bool countCoverage = false);

	// Fraction of target pixels that received at least one splat in
	// the last Warp() with countCoverage=true (0..1, expensive: see
	// above). Only meaningful when the last Warp counted coverage.
	float LastCoverage() const { return lastCoverage_; }

	// Human-readable reason for the last Init/Upload/Warp failure.
	const std::string& LastError() const { return lastError_; }

	// Name of the selected physical device (empty until Init succeeds).
	const std::string& DeviceName() const { return deviceName_; }

	// GPU stage times from the last Warp (valid=false when timestamp
	// queries are unsupported). Splat/pull shader cost vs copies/fill.
	struct GpuTimes {
		bool valid = false;
		double clearMs = 0.0;
		double splatMs = 0.0;
		double pullMs = 0.0;
		double fillMs = 0.0;
		double copyMs = 0.0;
	};
	GpuTimes LastGpuTimes() const { return lastGpu_; }

	// Raw splat plane (R32_UINT encodings) from the last Warp with
	// countCoverage=true; empty otherwise. For validity-mask dumps.
	const std::vector<uint32_t>& LastSplatPlane() const { return lastSplat_; }

private:
	bool Fail(std::string* error, const std::string& msg);

	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool initialized_ = false;
	float fillWeightPow_ = 1.0f;
	int fillMode_ = 0;
	int stretchRadius_ = 0;
	bool temporalCacheEnabled_ = false;
	// Host-side temporal gating: the cache ingests only on static
	// frames (intra-warp src/dst delta < 2mm / 0.3deg). bgValid_ is
	// set by the first static update and cleared by Upload().
	bool bgValid_ = false;
	Pose prevSrcPose_{};
	Pose prevDstPose_{};
	bool havePrevPose_ = false;
	// World-space linear velocity stub ((P0-P1)*alpha, alpha = 1):
	// placeholder for motion-compensated projection. Full per-pixel
	// GFFE history is follow-up work.
	float lastVel_[3] = {0.0f, 0.0f, 0.0f};
	bool lastStatic_ = false;
	float lastCoverage_ = -1.0f;
	GpuTimes lastGpu_{};
	std::vector<uint32_t> lastSplat_;
	std::string lastError_;
	std::string deviceName_;

	// Opaque Vulkan state (PIMPL to keep <vulkan/vulkan.h> out of the
	// header). Public so the implementation's file-local helpers can
	// name it; still incomplete here, only the .cpp defines it.
public:
	struct State;
private:
	State* state_ = nullptr;
};

} // namespace openwarp_vk
