// Headless test for the Vulkan compute warp (Option B).
//
// Builds a synthetic color+depth pair, warps it with an identity pose
// (must reproduce the input) and with a translated pose (must shift),
// and writes PNGs for visual inspection.
//
// Usage: openwarp-vulkan-test [spvDir] [outDir]
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan/vulkan_warp.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/lib/stb_image_write.h"

namespace {

constexpr uint32_t kSize = 256;

void BuildSynthetic(std::vector<uint8_t>& color, std::vector<float>& depth)
{
	color.resize(kSize * kSize * 4);
	depth.resize(kSize * kSize);
	for (uint32_t y = 0; y < kSize; y++) {
		for (uint32_t x = 0; x < kSize; x++) {
			size_t i = y * kSize + x;
			// Gradient background, flat far depth.
			color[4 * i + 0] = static_cast<uint8_t>((x * 255) / (kSize - 1));
			color[4 * i + 1] = static_cast<uint8_t>((y * 255) / (kSize - 1));
			color[4 * i + 2] = 128;
			color[4 * i + 3] = 255;
			depth[i] = 0.8f;
			// White foreground square in the middle, near depth.
			if (x >= kSize / 2 - 32 && x < kSize / 2 + 32 && y >= kSize / 2 - 32 &&
			    y < kSize / 2 + 32) {
				color[4 * i + 0] = 255;
				color[4 * i + 1] = 255;
				color[4 * i + 2] = 255;
				depth[i] = 0.2f;
			}
		}
	}
}

double MeanAbsDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
	double acc = 0.0;
	size_t n = a.size();
	for (size_t i = 0; i < n; i++)
		acc += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
	return acc / static_cast<double>(n);
}

double BlackFraction(const std::vector<uint8_t>& img)
{
	size_t black = 0;
	size_t px = img.size() / 4;
	for (size_t i = 0; i < px; i++) {
		if (img[4 * i] == 0 && img[4 * i + 1] == 0 && img[4 * i + 2] == 0)
			black++;
	}
	return static_cast<double>(black) / static_cast<double>(px);
}

} // namespace

int main(int argc, char** argv)
{
	std::string spvDir = "../openwarp-shaders";
	if (const char* env = std::getenv("OPENWARP_SPV_DIR"))
		spvDir = env;
	if (argc > 1)
		spvDir = argv[1];
	std::string outDir = "./vulkan-test-out";
	if (argc > 2)
		outDir = argv[2];
	std::string mkdirCmd = std::string("mkdir -p ") + outDir;
	if (std::system(mkdirCmd.c_str()) != 0) {
		std::cerr << "cannot create " << outDir << std::endl;
		return 1;
	}

	openwarp_vk::VulkanWarp warp;
	std::string error;
	if (!warp.Init(kSize, kSize, spvDir, &error)) {
		std::cerr << "VulkanWarp::Init failed: " << error << std::endl;
		return 1;
	}

	std::vector<uint8_t> color;
	std::vector<float> depth;
	BuildSynthetic(color, depth);
	if (!warp.Upload(color.data(), depth.data())) {
		std::cerr << "Upload failed" << std::endl;
		return 1;
	}

	openwarp_vk::Pose identity{};
	openwarp_vk::Mat4 proj{};
	if (!openwarp_vk::PerspectiveGL(60.0f, 1.0f, 0.1f, 100.0f, proj)) {
		std::cerr << "PerspectiveGL failed" << std::endl;
		return 1;
	}

	bool ok = true;

	// Test 1: identity warp must reproduce the input.
	std::vector<uint8_t> idOut;
	if (!warp.Warp(identity, proj, identity, proj, 0.0f, idOut, true)) {
		std::cerr << "identity Warp() failed" << std::endl;
		return 1;
	}
	double idDiff = MeanAbsDiff(color, idOut);
	float idCov = warp.LastCoverage();
	std::cout << "[identity] meanAbsDiff=" << idDiff << " coverage=" << idCov << std::endl;
	if (!(idDiff < 3.0 && idCov > 0.9)) {
		std::cerr << "FAIL: identity warp diverged (diff<3, cov>0.9 required)" << std::endl;
		ok = false;
	}

	// Test 2: translated target pose must shift the image.
	openwarp_vk::Pose shifted = identity;
	shifted.pos[0] = 0.15f;
	std::vector<uint8_t> shiftOut;
	if (!warp.Warp(identity, proj, shifted, proj, 0.0f, shiftOut, true)) {
		std::cerr << "shifted Warp() failed" << std::endl;
		return 1;
	}
	double shiftDiff = MeanAbsDiff(color, shiftOut);
	float shiftCov = warp.LastCoverage();
	double blackFrac = BlackFraction(shiftOut);
	std::cout << "[shifted] meanAbsDiff=" << shiftDiff << " coverage=" << shiftCov
	          << " blackFrac=" << blackFrac << std::endl;
	if (!(shiftDiff > 1.0 && shiftCov > 0.5)) {
		std::cerr << "FAIL: translated warp did not move the image" << std::endl;
		ok = false;
	}

	// Test 4: pyramid fill. Same shifted warp with fill enabled: the
	// disoccluded square interior must be patched with a local average
	// (far fewer pure-black pixels than the unfilled run above), while
	// valid pixels stay bit-identical to the unfilled result.
	if (!warp.Upload(color.data(), depth.data())) {
		std::cerr << "refill Upload failed" << std::endl;
		return 1;
	}
	std::vector<uint8_t> shiftFilled;
	if (!warp.Warp(identity, proj, shifted, proj, 1.0f, shiftFilled, true)) {
		std::cerr << "filled Warp() failed" << std::endl;
		return 1;
	}
	double filledBlack = BlackFraction(shiftFilled);
	std::cout << "[filled] blackFrac=" << filledBlack << " (unfilled was " << blackFrac << ")"
	          << std::endl;
	if (!(filledBlack < 0.15 && filledBlack < blackFrac * 0.5)) {
		std::cerr << "FAIL: pyramid fill did not reduce black (filled<0.15, halved required)"
		          << std::endl;
		ok = false;
	}
	{
		// Valid pixels must be untouched by the fill pass.
		double validDiff = 0.0;
		size_t validCount = 0;
		for (size_t i = 0; i < shiftOut.size(); i++) {
			bool wasBlack = (shiftOut[i] == 0);
			if (!wasBlack) {
				validDiff += std::abs(static_cast<int>(shiftOut[i]) -
				                       static_cast<int>(shiftFilled[i]));
				validCount++;
			}
		}
		validDiff /= validCount ? static_cast<double>(validCount) : 1.0;
		std::cout << "[filled] valid-pixel meanAbsDiff=" << validDiff << std::endl;
		if (!(validDiff < 1e-6)) {
			std::cerr << "FAIL: fill pass altered valid pixels" << std::endl;
			ok = false;
		}
	}
	stbi_write_png((outDir + "/shifted_filled.png").c_str(), kSize, kSize, 4, shiftFilled.data(),
	               kSize * 4);

	stbi_write_png((outDir + "/input.png").c_str(), kSize, kSize, 4, color.data(), kSize * 4);
	stbi_write_png((outDir + "/identity.png").c_str(), kSize, kSize, 4, idOut.data(), kSize * 4);
	stbi_write_png((outDir + "/shifted.png").c_str(), kSize, kSize, 4, shiftOut.data(),
	               kSize * 4);

	// Test 3: far-plane regression. Flat depth-1.0 (sky) scene, target
	// backs away: without the z<=0.995 clamp the background's
	// reprojected depth exceeds 1.0 and the frame goes black.
	std::vector<uint8_t> flatColor(kSize * kSize * 4, 200);
	std::vector<float> flatDepth(kSize * kSize, 1.0f);
	if (!warp.Upload(flatColor.data(), flatDepth.data())) {
		std::cerr << "flat Upload failed" << std::endl;
		return 1;
	}
	openwarp_vk::Pose backed = identity;
	backed.pos[2] = -0.2f;
	std::vector<uint8_t> backedOut;
	if (!warp.Warp(identity, proj, backed, proj, 0.0f, backedOut, true)) {
		std::cerr << "backed Warp() failed" << std::endl;
		return 1;
	}
	float backedCov = warp.LastCoverage();
	double backedBlack = BlackFraction(backedOut);
	std::cout << "[backed] coverage=" << backedCov << " blackFrac=" << backedBlack << std::endl;
	if (!(backedCov > 0.9 && backedBlack < 0.1)) {
		std::cerr << "FAIL: far-plane background dropped (cov>0.9, black<0.1 required)"
		          << std::endl;
		ok = false;
	}
	stbi_write_png((outDir + "/backed.png").c_str(), kSize, kSize, 4, backedOut.data(),
	               kSize * 4);
	std::cout << (ok ? "VULKAN WARP TEST: PASS" : "VULKAN WARP TEST: FAIL") << std::endl;
	return ok ? 0 : 1;
}
