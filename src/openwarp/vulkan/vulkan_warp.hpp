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
	// reason in *error when non-null).
	bool Init(uint32_t width, uint32_t height, const std::string& spvDir, std::string* error = nullptr);
	bool IsInitialized() const { return initialized_; }
	void Shutdown();

	// Upload a frame. Row 0 is the first texel row for BOTH planes
	// (color and depth must agree); color is RGBA8, depth is standard
	// 0..1 nonlinear float. Returns false on failure.
	bool Upload(const uint8_t* colorRGBA, const float* depth);

	// Warp the uploaded frame from (srcPose, srcProj) to
	// (dstPose, dstProj) and download the RGBA8 result into outColor
	// (width*height*4 bytes). holeRadius 0 disables hole fill.
	// srcProj/dstProj are PerspectiveGL/FrustumGL matrices; near/far
	// must match between the two (same convention as dynacon-vr).
	bool Warp(const Pose& srcPose, const Mat4 srcProj,
	          const Pose& dstPose, const Mat4 dstProj,
	          float holeRadius, std::vector<uint8_t>& outColor);

	// Fraction of target pixels that received at least one splat in
	// the last Warp() (0..1). Useful for diagnosing empty warps.
	float LastCoverage() const { return lastCoverage_; }

private:
	bool Fail(std::string* error, const std::string& msg);

	uint32_t width_ = 0;
	uint32_t height_ = 0;
	bool initialized_ = false;
	float lastCoverage_ = 0.0f;

	// Opaque Vulkan state (PIMPL to keep <vulkan/vulkan.h> out of the
	// header). Public so the implementation's file-local helpers can
	// name it; still incomplete here, only the .cpp defines it.
public:
	struct State;
private:
	State* state_ = nullptr;
};

} // namespace openwarp_vk
