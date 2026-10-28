// Headless test for the Vulkan compute warp (modes 0-1).
//
// Builds a synthetic color+depth pair, warps it with an identity pose
// (must reproduce the input) and with a translated pose (must shift),
// and writes PNGs for visual inspection. Mode 0 (holes stay black) is
// the baseline; mode 1 (JFA background blur) must shrink holes vs
// mode 0 without disturbing the identity reproduction.
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

// Flat mid-gray background + white foreground square. The gradient
// scene above hides partial foreground mixing (128 gray + white fringe
// = ~180, below the >240 white gate). Flat 100 gray makes any bilinear
// pull fringe or gather reach show up as excess brightness in filled
// holes: ideal fill is exactly 100.
void BuildFlat(std::vector<uint8_t>& color, std::vector<float>& depth)
{
	color.resize(kSize * kSize * 4);
	depth.resize(kSize * kSize);
	for (uint32_t y = 0; y < kSize; y++) {
		for (uint32_t x = 0; x < kSize; x++) {
			size_t i = y * kSize + x;
			color[4 * i + 0] = 100;
			color[4 * i + 1] = 100;
			color[4 * i + 2] = 100;
			color[4 * i + 3] = 255;
			depth[i] = 0.8f;
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
	double acc = 0;
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

	// Test 1: identity warp must reproduce the input (black-hole fill:
	// holes stay black, so identity should have no holes at all).
	warp.SetFillMode(0);
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

	// Test 2: translated target pose must shift the image (disocclusions
	// stay black with the black-hole fill).
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

	// Test 4: JFA background blur (mode 1). Re-upload the synthetic
	// scene, warp the shifted pose with holeRadius>0: holes must shrink
	// vs mode 0 and the fill must come from background (no foreground
	// white smeared into the blur). Identity with mode 1 must still
	// reproduce the input (no holes to fill).
	if (!warp.Upload(color.data(), depth.data())) {
		std::cerr << "jfa Upload failed" << std::endl;
		return 1;
	}
	warp.SetFillMode(1);
	std::vector<uint8_t> jfaShiftOut;
	if (!warp.Warp(identity, proj, shifted, proj, 1.0f, jfaShiftOut, true)) {
		std::cerr << "jfa shifted Warp() failed" << std::endl;
		return 1;
	}
	double jfaBlack = BlackFraction(jfaShiftOut);
	std::cout << "[jfa-shifted] blackFrac=" << jfaBlack << " (mode0 blackFrac=" << blackFrac
	          << ")" << std::endl;
	if (!(jfaBlack < blackFrac)) {
		std::cerr << "FAIL: JFA fill did not reduce holes vs black" << std::endl;
		ok = false;
	}
	// Valid pixels must be bit-exact vs mode 0 (fill never touches them),
	// and filled holes must be background (never white foreground smear,
	// never left black).
	{
		size_t validMismatch = 0;
		size_t filledWhite = 0;
		size_t filledBlack = 0;
		size_t filled = 0;
		for (size_t i = 0; i < shiftOut.size() / 4; i++) {
			bool hole = (shiftOut[4 * i] == 0 && shiftOut[4 * i + 1] == 0 &&
			             shiftOut[4 * i + 2] == 0);
			if (!hole) {
				if (jfaShiftOut[4 * i] != shiftOut[4 * i] ||
				    jfaShiftOut[4 * i + 1] != shiftOut[4 * i + 1] ||
				    jfaShiftOut[4 * i + 2] != shiftOut[4 * i + 2])
					validMismatch++;
			} else {
				filled++;
				bool white = (jfaShiftOut[4 * i] > 240 && jfaShiftOut[4 * i + 1] > 240 &&
				              jfaShiftOut[4 * i + 2] > 240);
				bool black = (jfaShiftOut[4 * i] == 0 && jfaShiftOut[4 * i + 1] == 0 &&
				              jfaShiftOut[4 * i + 2] == 0);
				if (white)
					filledWhite++;
				if (black)
					filledBlack++;
			}
		}
		std::cout << "[jfa-shifted] validMismatch=" << validMismatch << " filledWhite=" << filledWhite
		          << " filledBlack=" << filledBlack << " filled=" << filled << std::endl;
		if (validMismatch != 0) {
			std::cerr << "FAIL: JFA touched valid pixels" << std::endl;
			ok = false;
		}
		if (filled > 0 && (filledWhite != 0 || filledBlack != 0)) {
			std::cerr << "FAIL: JFA filled holes with foreground/left black" << std::endl;
			ok = false;
		}
	}
	std::vector<uint8_t> jfaIdOut;
	if (!warp.Warp(identity, proj, identity, proj, 1.0f, jfaIdOut, true)) {
		std::cerr << "jfa identity Warp() failed" << std::endl;
		return 1;
	}
	double jfaIdDiff = MeanAbsDiff(color, jfaIdOut);
	std::cout << "[jfa-identity] meanAbsDiff=" << jfaIdDiff << std::endl;
	if (!(jfaIdDiff < 3.0)) {
		std::cerr << "FAIL: JFA identity diverged (diff<3 required)" << std::endl;
		ok = false;
	}
	stbi_write_png((outDir + "/jfa_shifted.png").c_str(), kSize, kSize, 4, jfaShiftOut.data(),
	               kSize * 4);
	stbi_write_png((outDir + "/jfa_identity.png").c_str(), kSize, kSize, 4, jfaIdOut.data(),
	               kSize * 4);

	// Test 5: foreground-fringe rig. Flat 100-gray background + white
	// square, shifted pose, JFA fill. Filled-hole brightness above flat
	// 100 is foreground dragged in via bilinear pull fringe or gather
	// reach across the edge: the eroded gather must keep it at zero.
	{
		std::vector<uint8_t> flatBgColor;
		std::vector<float> flatBgDepth;
		BuildFlat(flatBgColor, flatBgDepth);
		if (!warp.Upload(flatBgColor.data(), flatBgDepth.data())) {
			std::cerr << "fringe Upload failed" << std::endl;
			return 1;
		}
		warp.SetFillMode(0);
		std::vector<uint8_t> fringeBlack;
		if (!warp.Warp(identity, proj, shifted, proj, 0.0f, fringeBlack, false)) {
			std::cerr << "fringe black Warp() failed" << std::endl;
			return 1;
		}
		warp.SetFillMode(1);
		std::vector<uint8_t> fringeFill;
		if (!warp.Warp(identity, proj, shifted, proj, 1.0f, fringeFill, false)) {
			std::cerr << "fringe fill Warp() failed" << std::endl;
			return 1;
		}
		size_t holes = 0;
		double maxExcess = 0.0;
		double sumExcess = 0.0;
		size_t over10 = 0;
		size_t over30 = 0;
		for (size_t i = 0; i < fringeBlack.size() / 4; i++) {
			bool hole = (fringeBlack[4 * i] == 0 && fringeBlack[4 * i + 1] == 0 &&
			             fringeBlack[4 * i + 2] == 0);
			if (!hole)
				continue;
			holes++;
			double peak = (fringeFill[4 * i] + fringeFill[4 * i + 1] + fringeFill[4 * i + 2]) / 3.0;
			double excess = peak - 100.0;
			if (excess > maxExcess)
				maxExcess = excess;
			if (excess > 0)
				sumExcess += excess;
			if (excess > 10.0)
				over10++;
			if (excess > 30.0)
				over30++;
		}
		double meanExcess = holes ? sumExcess / (double)holes : 0.0;
		std::cout << "[fringe] holes=" << holes << " maxExcess=" << maxExcess
		          << " meanExcess=" << meanExcess << " over10=" << over10 << " over30=" << over30
		          << std::endl;
		if (!(maxExcess <= 1.0 && over10 == 0)) {
			std::cerr << "FAIL: foreground fringe in fill (maxExcess<=1, over10==0 required)"
			          << std::endl;
			ok = false;
		}
		stbi_write_png((outDir + "/fringe_fill.png").c_str(), kSize, kSize, 4, fringeFill.data(),
		               kSize * 4);
	}
	std::cout << (ok ? "VULKAN WARP TEST: PASS" : "VULKAN WARP TEST: FAIL") << std::endl;
	return ok ? 0 : 1;
}
