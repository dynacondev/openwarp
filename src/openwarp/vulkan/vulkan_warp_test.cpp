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

// Oracle for the checker scene under a given warp: expected output gray
// for target pixel (tx,ty), or -1 (bilinear boundary blend: skip strict
// check), or -2 (expect hole/black: EMPTY splat or suv out of bounds).
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

// Checkerboard background + white square: every background texel has a
// known procedural color, so a CPU mirror of the pull math predicts the
// exact expected output per pixel. Any spatial error (overshoot strip,
// bilinear fringe, fill smear) shows up as a wrong-cell mismatch.
void BuildChecker(std::vector<uint8_t>& color, std::vector<float>& depth)
{
	color.resize(kSize * kSize * 4);
	depth.resize(kSize * kSize);
	for (uint32_t y = 0; y < kSize; y++) {
		for (uint32_t x = 0; x < kSize; x++) {
			size_t i = y * kSize + x;
			bool cell = (((x / 16) + (y / 16)) % 2) == 0;
			uint8_t g = cell ? (uint8_t)60 : (uint8_t)200;
			color[4 * i + 0] = g;
			color[4 * i + 1] = g;
			color[4 * i + 2] = g;
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

uint8_t CheckerAt(int x, int y)
{
	if (x < 0)
		x = 0;
	if (y < 0)
		y = 0;
	if (x >= (int)kSize)
		x = (int)kSize - 1;
	if (y >= (int)kSize)
		y = (int)kSize - 1;
	bool cell = (((x / 16) + (y / 16)) % 2) == 0;
	return cell ? (uint8_t)60 : (uint8_t)200;
}

// CPU mirror of warp_pull.comp: unproject (tuv, target-view depth) with
// dstInv, reproject with srcVP. Returns false exactly when the shader
// marks a hole (EMPTY handled by caller, behind-camera, out-of-bounds).
// ez is the expected source depth for the consistency gate.
bool CpuPullSuv(float tuvX, float tuvY, float zt, const float* dstInv, const float* srcVP,
                float& suvX, float& suvY, float& ez)
{
	float v[4] = {tuvX * 2.0f - 1.0f, tuvY * 2.0f - 1.0f, zt * 2.0f - 1.0f, 1.0f};
	float w[4];
	for (int i = 0; i < 4; i++)
		w[i] = dstInv[i] * v[0] + dstInv[4 + i] * v[1] + dstInv[8 + i] * v[2] +
		       dstInv[12 + i] * v[3];
	float s[4];
	for (int i = 0; i < 4; i++)
		s[i] = srcVP[i] * w[0] + srcVP[4 + i] * w[1] + srcVP[8 + i] * w[2] + srcVP[12 + i] * w[3];
	if (s[3] <= 0.0f)
		return false;
	float x = s[0] / s[3] * 0.5f + 0.5f;
	float y = s[1] / s[3] * 0.5f + 0.5f;
	if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f)
		return false;
	suvX = x;
	suvY = y;
	ez = s[2] / s[3] * 0.5f + 0.5f;
	return true;
}

// CPU mirror of warp_splat.comp per-texel math: maps a source texel
// center + window depth to target float coords. Returns false when the
// texel splats nowhere (behind camera / outside frustum), exactly like
// the shader's early-outs.
bool CpuSplatTarget(float srcX, float srcY, float z, const float* renderInvVP,
                    const float* warpVP, uint32_t size, float& outTx, float& outTy)
{
	float guvX = (srcX + 0.5f) / (float)size;
	float guvY = (srcY + 0.5f) / (float)size;
	float v[4] = {guvX * 2.0f - 1.0f, guvY * 2.0f - 1.0f, z * 2.0f - 1.0f, 1.0f};
	float w[4];
	for (int i = 0; i < 4; i++)
		w[i] = renderInvVP[i] * v[0] + renderInvVP[4 + i] * v[1] + renderInvVP[8 + i] * v[2] +
		       renderInvVP[12 + i] * v[3];
	float t[4];
	for (int i = 0; i < 4; i++)
		t[i] = warpVP[i] * w[0] + warpVP[4 + i] * w[1] + warpVP[8 + i] * w[2] + warpVP[12 + i] * w[3];
	if (t[3] <= 0.0f)
		return false;
	float tz = t[2] / t[3];
	if (tz < -1.0f || tz > 1.0f)
		return false;
	float tx = t[0] / t[3] * 0.5f + 0.5f;
	float ty = t[1] / t[3] * 0.5f + 0.5f;
	if (tx < 0.0f || tx > 1.0f || ty < 0.0f || ty > 1.0f)
		return false;
	outTx = tx * (float)size;
	outTy = ty * (float)size;
	return true;
}

// Oracle for the checker scene under depth-aware pull: expected output
// gray for target pixel (tx,ty), or -2 (expect hole/black: EMPTY splat,
// suv out of bounds, or no depth-consistent tap). Mirrors the shader's
// 2x2 renormalized gather, so every pixel is strict (no boundary skip:
// consistent taps agree by construction).
int OracleChecker(int tx, int ty, uint32_t splatEnc, const float* dstInv, const float* srcVP)
{
	if (splatEnc == 0xffffffffu)
		return -2;
	float tuvX = ((float)tx + 0.5f) / (float)kSize;
	float tuvY = ((float)ty + 0.5f) / (float)kSize;
	float zt = (float)splatEnc / 16777215.0f;
	float suvX = 0.0f;
	float suvY = 0.0f;
	float ez = 0.0f;
	if (!CpuPullSuv(tuvX, tuvY, zt, dstInv, srcVP, suvX, suvY, ez))
		return -2;
	float fx = suvX * (float)kSize - 0.5f;
	float fy = suvY * (float)kSize - 0.5f;
	int x0 = (int)std::floor(fx);
	int y0 = (int)std::floor(fy);
	float ax = fx - (float)x0;
	float ay = fy - (float)y0;
	auto at = [](int sx, int sy) -> int {
		if (sx < 0)
			sx = 0;
		if (sy < 0)
			sy = 0;
		if (sx >= (int)kSize)
			sx = (int)kSize - 1;
		if (sy >= (int)kSize)
			sy = (int)kSize - 1;
		if (sx >= (int)kSize / 2 - 32 && sx < (int)kSize / 2 + 32 && sy >= (int)kSize / 2 - 32 &&
		    sy < (int)kSize / 2 + 32)
			return 255;
		return CheckerAt(sx, sy);
	};
	auto depthAt = [](int sx, int sy) -> float {
		if (sx < 0)
			sx = 0;
		if (sy < 0)
			sy = 0;
		if (sx >= (int)kSize)
			sx = (int)kSize - 1;
		if (sy >= (int)kSize)
			sy = (int)kSize - 1;
		if (sx >= (int)kSize / 2 - 32 && sx < (int)kSize / 2 + 32 && sy >= (int)kSize / 2 - 32 &&
		    sy < (int)kSize / 2 + 32)
			return 0.2f;
		return 0.8f;
	};
	int c[2][2] = {{at(x0, y0), at(x0 + 1, y0)}, {at(x0, y0 + 1), at(x0 + 1, y0 + 1)}};
	float w[2][2] = {{(1.0f - ax) * (1.0f - ay), ax * (1.0f - ay)},
	                 {(1.0f - ax) * ay, ax * ay}};
	float sum = 0.0f;
	float wsum = 0.0f;
	for (int oy = 0; oy <= 1; oy++) {
		for (int ox = 0; ox <= 1; ox++) {
			float d = depthAt(x0 + ox, y0 + oy);
			if (std::fabs(d - ez) < 0.02f) {
				sum += (float)c[oy][ox] * w[oy][ox];
				wsum += w[oy][ox];
			}
		}
	}
	if (wsum <= 1e-6f)
		return -2;
	return (int)(sum / wsum + 0.5f);
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
	// Valid pixels far from holes must be bit-exact vs mode 0 (fill
	// never touches them). Pixels within 1px of a mode-0 hole are the
	// dilation ring (refilled from interior background in mode 1):
	// allowed to change, but must land on background (never white
	// foreground smear, never black). Filled holes likewise.
	{
		size_t validMismatch = 0;
		size_t ringChanged = 0;
		size_t ringWhite = 0;
		size_t ringBlack = 0;
		size_t filledWhite = 0;
		size_t filledBlack = 0;
		size_t filled = 0;
		auto isBlack0 = [&](int x, int y) {
			size_t i = (size_t)y * kSize + (size_t)x;
			return shiftOut[4 * i] == 0 && shiftOut[4 * i + 1] == 0 &&
			       shiftOut[4 * i + 2] == 0;
		};
		for (int y = 0; y < (int)kSize; y++) {
			for (int x = 0; x < (int)kSize; x++) {
				size_t i = (size_t)y * kSize + (size_t)x;
				bool hole = isBlack0(x, y);
				if (!hole) {
					if (jfaShiftOut[4 * i] != shiftOut[4 * i] ||
					    jfaShiftOut[4 * i + 1] != shiftOut[4 * i + 1] ||
					    jfaShiftOut[4 * i + 2] != shiftOut[4 * i + 2]) {
						bool nearHole = false;
						for (int oy = -1; oy <= 1 && !nearHole; oy++) {
							for (int ox = -1; ox <= 1 && !nearHole; ox++) {
								int sx = x + ox;
								int sy = y + oy;
								if (sx < 0 || sy < 0 || sx >= (int)kSize || sy >= (int)kSize)
									continue;
								if (isBlack0(sx, sy))
									nearHole = true;
							}
						}
						if (nearHole) {
							ringChanged++;
							bool white = (jfaShiftOut[4 * i] > 240 &&
							              jfaShiftOut[4 * i + 1] > 240 &&
							              jfaShiftOut[4 * i + 2] > 240);
							bool black = (jfaShiftOut[4 * i] == 0 &&
							              jfaShiftOut[4 * i + 1] == 0 &&
							              jfaShiftOut[4 * i + 2] == 0);
							if (white)
								ringWhite++;
							if (black)
								ringBlack++;
						} else {
							validMismatch++;
						}
					}
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
		}
		std::cout << "[jfa-shifted] validMismatch=" << validMismatch << " ringChanged=" << ringChanged
		          << " ringWhite=" << ringWhite << " ringBlack=" << ringBlack
		          << " filledWhite=" << filledWhite << " filledBlack=" << filledBlack
		          << " filled=" << filled << std::endl;
		if (validMismatch != 0) {
			std::cerr << "FAIL: JFA touched valid pixels far from holes" << std::endl;
			ok = false;
		}
		if (filled > 0 && (filledWhite != 0 || filledBlack != 0)) {
			std::cerr << "FAIL: JFA filled holes with foreground/left black" << std::endl;
			ok = false;
		}
		if (ringChanged > 0 && (ringWhite != 0 || ringBlack != 0)) {
			std::cerr << "FAIL: dilation ring refilled with foreground/left black" << std::endl;
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
		// Full-image gate: the square moved fully off-screen, so every
		// output pixel must be flat 100 — including the 1px valid ring
		// the old code left fringe-tinted next to holes (hole-masked
		// metrics cannot see it; this one fails pre-fix with dev 34).
		{
			int fullMaxDev = 0;
			for (size_t i = 0; i < fringeFill.size() / 4; i++) {
				for (int c = 0; c < 3; c++) {
					int dev = std::abs((int)fringeFill[4 * i + c] - 100);
					if (dev > fullMaxDev)
						fullMaxDev = dev;
				}
			}
			std::cout << "[fringe] fullImageMaxDev=" << fullMaxDev << std::endl;
			if (!(fullMaxDev <= 1)) {
				std::cerr << "FAIL: full-image fringe (dev<=1 required)" << std::endl;
				ok = false;
			}
		}

		// Test 6: far-depth passthrough (dynacon 0.995-clip fix). Uniform
		// 0.997 depth under identity must splat back ~0.997, not the old
		// min(z, 0.995) collapse. Fails pre-fix (encodes 0.995).
		{
			std::vector<uint8_t> farColor(kSize * kSize * 4, 150);
			std::vector<float> farDepth(kSize * kSize, 0.997f);
			if (!warp.Upload(farColor.data(), farDepth.data())) {
				std::cerr << "far Upload failed" << std::endl;
				return 1;
			}
			warp.SetFillMode(0);
			std::vector<uint8_t> farOut;
			if (!warp.Warp(identity, proj, identity, proj, 0.0f, farOut, true)) {
				std::cerr << "far Warp() failed" << std::endl;
				return 1;
			}
			const std::vector<uint32_t>& splat = warp.LastSplatPlane();
			double meanEnc = 0.0;
			for (uint32_t w : splat)
				meanEnc += (double)w / 16777215.0;
			meanEnc /= (double)splat.size();
			std::cout << "[far] meanSplatDepth=" << meanEnc << std::endl;
			if (!(std::abs(meanEnc - 0.997) < 0.001)) {
				std::cerr << "FAIL: far depth collapsed (0.997 +/- 0.001 required)"
				          << std::endl;
				ok = false;
			}
		}

		// Test 7: small-shift drag rig (raw pull, mode 0). The big-shift
		// tests fling the square fully off-screen; here a 0.005 X step
		// keeps it on screen so overshoot is observable. The expected
		// foreground interval comes from warping the square edges on the
		// CPU with the same math as the splat shader; any non-background
		// output outside that interval (+1px quantization tolerance) is
		// dragged foreground. Flat 100 background makes any leak exact.
		{
			std::vector<uint8_t> flatBgColor;
			std::vector<float> flatBgDepth;
			BuildFlat(flatBgColor, flatBgDepth);
			if (!warp.Upload(flatBgColor.data(), flatBgDepth.data())) {
				std::cerr << "drag Upload failed" << std::endl;
				return 1;
			}
			openwarp_vk::Pose small = identity;
			small.pos[0] = 0.005f;
			// Fold matrices exactly like VulkanWarp::Warp.
			openwarp_vk::Mat4 camSrc, camDst, viewSrc, viewDst;
			openwarp_vk::Mat4 renderInvP, renderInvV, renderInvVP, warpVP;
			openwarp_vk::CameraFromPose(identity, camSrc);
			openwarp_vk::CameraFromPose(small, camDst);
			openwarp_vk::Inverse(camSrc, viewSrc);
			openwarp_vk::Inverse(camDst, viewDst);
			openwarp_vk::Inverse(proj, renderInvP);
			openwarp_vk::Inverse(viewSrc, renderInvV);
			openwarp_vk::Multiply(renderInvV, renderInvP, renderInvVP);
			openwarp_vk::Multiply(proj, viewDst, warpVP);
			float eL = 1e30f, eR = -1e30f;
			bool any = false;
			for (int x = kSize / 2 - 32; x < kSize / 2 + 32; x++) {
				float tx = 0.0f;
				float ty = 0.0f;
				if (CpuSplatTarget((float)x, 128.0f, 0.2f, renderInvVP, warpVP, kSize, tx,
				                   ty)) {
					if (tx < eL)
						eL = tx;
					if (tx > eR)
						eR = tx;
					any = true;
				}
			}
			if (!any) {
				std::cerr << "FAIL: drag rig square splats nowhere" << std::endl;
				ok = false;
			}
			warp.SetFillMode(0);
			std::vector<uint8_t> dragOut;
			if (!warp.Warp(identity, proj, small, proj, 0.0f, dragOut, false)) {
				std::cerr << "drag Warp() failed" << std::endl;
				return 1;
			}
			int expL = (int)std::floor(eL);
			int expR = (int)std::floor(eR);
			int maxDev = 0;
			size_t badPx = 0;
			size_t holePx = 0;
			size_t whiteOutside = 0;
			for (int y = 112; y < 144; y++) {
				for (int x = 0; x < (int)kSize; x++) {
					if (x >= expL - 1 && x <= expR + 1)
						continue;
					size_t i = (size_t)y * kSize + (size_t)x;
					bool hole = (dragOut[4 * i] == 0 && dragOut[4 * i + 1] == 0 &&
					             dragOut[4 * i + 2] == 0);
					if (hole) {
						holePx++;
						continue;
					}
					bool white = (dragOut[4 * i] > 240 && dragOut[4 * i + 1] > 240 &&
					              dragOut[4 * i + 2] > 240);
					if (white) {
						whiteOutside++;
						continue;
					}
					for (int c = 0; c < 3; c++) {
						int dev = std::abs((int)dragOut[4 * i + c] - 100);
						if (dev > maxDev)
							maxDev = dev;
						if (dev > 1)
							badPx++;
					}
				}
			}
			std::cout << "[drag] expFg=[" << expL << "," << expR << "] maxDevOutside=" << maxDev
			          << " badPx=" << badPx << " whiteOutside=" << whiteOutside << " holePx=" << holePx
			          << std::endl;
			if (!(whiteOutside == 0)) {
				std::cerr << "FAIL: full foreground splatted outside warped interval"
				          << std::endl;
				ok = false;
			}
			// Strict tint check (no quantization tolerance): point splats
			// are binary, so any tinted background strictly outside the
			// analytic interval is bilinear pull fringe. Fails pre-fix
			// (fringe texel reads ~235 on the occlusion edge).
			int strictDev = 0;
			for (int y = 112; y < 144; y++) {
				for (int x = 0; x < (int)kSize; x++) {
					if (x >= expL && x <= expR)
						continue;
					size_t i = (size_t)y * kSize + (size_t)x;
					bool hole = (dragOut[4 * i] == 0 && dragOut[4 * i + 1] == 0 &&
					             dragOut[4 * i + 2] == 0);
					if (hole)
						continue;
					bool white = (dragOut[4 * i] > 240 && dragOut[4 * i + 1] > 240 &&
					              dragOut[4 * i + 2] > 240);
					if (white)
						continue;
					for (int c = 0; c < 3; c++) {
						int dev = std::abs((int)dragOut[4 * i + c] - 100);
						if (dev > strictDev)
							strictDev = dev;
					}
				}
			}
			std::cout << "[drag] strictTintDev=" << strictDev << std::endl;
			if (!(strictDev <= 1)) {
				std::cerr << "FAIL: pull fringe tint outside warped interval (dev<=1 required)"
				          << std::endl;
				ok = false;
			}
			stbi_write_png((outDir + "/drag_raw.png").c_str(), kSize, kSize, 4, dragOut.data(),
			               kSize * 4);
		}

		// Test 8: checkerboard oracle (small shift, raw pull + fill).
		// Procedural background + CPU pull mirror predicts exact output
		// per pixel: any spatial error (overshoot, fringe, smear) is a
		// wrong-cell mismatch. Mode 1 must additionally leave no black
		// and keep the foreground core white.
		{
			std::vector<uint8_t> checkerColor;
			std::vector<float> checkerDepth;
			BuildChecker(checkerColor, checkerDepth);
			if (!warp.Upload(checkerColor.data(), checkerDepth.data())) {
				std::cerr << "checker Upload failed" << std::endl;
				return 1;
			}
			openwarp_vk::Pose small = identity;
			small.pos[0] = 0.005f;
			// Plus a 2-degree yaw (head-like motion: rotation about Y
			// plus translation). Pure yaw about the camera center
			// causes no disocclusion, so the translation stays.
			small.quat[1] = 0.0174524f;
			small.quat[3] = 0.9998477f;
			openwarp_vk::Mat4 camSrc, camDst, viewSrc, viewDst;
			openwarp_vk::Mat4 renderInvP, renderInvV, renderInvVP, warpVP;
			openwarp_vk::Mat4 dstInvP, dstInvV, dstInv, srcVP;
			openwarp_vk::CameraFromPose(identity, camSrc);
			openwarp_vk::CameraFromPose(small, camDst);
			openwarp_vk::Inverse(camSrc, viewSrc);
			openwarp_vk::Inverse(camDst, viewDst);
			openwarp_vk::Inverse(proj, dstInvP);
			openwarp_vk::Inverse(viewDst, dstInvV);
			openwarp_vk::Multiply(dstInvV, dstInvP, dstInv);
			openwarp_vk::Multiply(proj, viewSrc, srcVP);
			openwarp_vk::Inverse(proj, renderInvP);
			openwarp_vk::Inverse(viewSrc, renderInvV);
			openwarp_vk::Multiply(renderInvV, renderInvP, renderInvVP);
			openwarp_vk::Multiply(proj, viewDst, warpVP);
			// Analytic foreground interval at mid row (same geometry as
			// Test 7, now under yaw too).
			float eL = 1e30f, eR = -1e30f;
			for (int x = kSize / 2 - 32; x < kSize / 2 + 32; x++) {
				float tx = 0.0f;
				float ty = 0.0f;
				if (CpuSplatTarget((float)x, 128.0f, 0.2f, renderInvVP, warpVP, kSize, tx,
				                   ty)) {
					if (tx < eL)
						eL = tx;
					if (tx > eR)
						eR = tx;
				}
			}
			int expFgL = (int)std::floor(eL);
			int expFgR = (int)std::floor(eR);
			warp.SetFillMode(0);
			std::vector<uint8_t> checkOut0;
			if (!warp.Warp(identity, proj, small, proj, 0.0f, checkOut0, true)) {
				std::cerr << "checker Warp() failed" << std::endl;
				return 1;
			}
			const std::vector<uint32_t> splat0 = warp.LastSplatPlane();
			size_t mismatch = 0;
			size_t shown = 0;
			for (int y = 0; y < (int)kSize; y++) {
				for (int x = 0; x < (int)kSize; x++) {
					size_t i = (size_t)y * kSize + (size_t)x;
					int e = OracleChecker(x, y, splat0[i], dstInv, srcVP);
					bool black = (checkOut0[4 * i] == 0 && checkOut0[4 * i + 1] == 0 &&
					              checkOut0[4 * i + 2] == 0);
					if (e == -2) {
						if (!black) {
							if (shown < 8)
								std::cout << "[oracle] hole-mismatch at (" << x << "," << y
								          << ") got (" << (int)checkOut0[4 * i] << ","
								          << (int)checkOut0[4 * i + 1] << ","
								          << (int)checkOut0[4 * i + 2] << ")" << std::endl;
							shown++;
							mismatch++;
						}
					} else {
						for (int c = 0; c < 3; c++) {
							if (std::abs((int)checkOut0[4 * i + c] - e) > 1) {
								if (shown < 8)
									std::cout << "[oracle] mismatch at (" << x << "," << y
									          << ") exp " << e << " got ("
									          << (int)checkOut0[4 * i] << ","
									          << (int)checkOut0[4 * i + 1] << ","
									          << (int)checkOut0[4 * i + 2] << ")" << std::endl;
								shown++;
								mismatch++;
								break;
							}
						}
					}
				}
			}
			std::cout << "[oracle] mismatch=" << mismatch << std::endl;
			if (!(mismatch == 0)) {
				std::cerr << "FAIL: pull output deviates from analytic oracle" << std::endl;
				ok = false;
			}
			// Fringe gate: pixels the splat marks as foreground must be
			// foreground-colored. Bilinear pull blends across the source
			// silhouette, leaving a gray fringe column (e.g. 235 instead
			// of 255) on every edge — visible as foreground drag in
			// motion, yet boundary-skipped by the oracle above. Fails
			// pre-fix; depth-aware pull sampling cures it.
			{
				size_t fgTint = 0;
				for (int y = 112; y < 144; y++) {
					for (int x = 0; x < (int)kSize; x++) {
						size_t i = (size_t)y * kSize + (size_t)x;
						if (splat0[i] == 0xffffffffu)
							continue;
						float zt = (float)splat0[i] / 16777215.0f;
						if (zt > 0.5f)
							continue;
						bool white = (checkOut0[4 * i] > 240 && checkOut0[4 * i + 1] > 240 &&
						              checkOut0[4 * i + 2] > 240);
						if (!white)
							fgTint++;
					}
				}
				std::cout << "[oracle] fgTint=" << fgTint << std::endl;
				if (!(fgTint == 0)) {
					std::cerr << "FAIL: foreground-depth pixels not foreground-colored"
					          << std::endl;
					ok = false;
				}
			}
			stbi_write_png((outDir + "/oracle_raw.png").c_str(), kSize, kSize, 4, checkOut0.data(),
			               kSize * 4);

			// Mode 1 on the same scene: holes must vanish, foreground
			// core must stay white, and valid pixels away from holes
			// must still match the oracle (fill touches only holes +
			// the 1px dilation ring).
			warp.SetFillMode(1);
			std::vector<uint8_t> checkOut1;
			if (!warp.Warp(identity, proj, small, proj, 1.0f, checkOut1, true)) {
				std::cerr << "checker fill Warp() failed" << std::endl;
				return 1;
			}
			const std::vector<uint32_t> splat1 = warp.LastSplatPlane();
			size_t black1 = 0;
			size_t coreNonWhite = 0;
			size_t fillMismatch = 0;
			auto hole0 = [&](int x, int y) {
				size_t i = (size_t)y * kSize + (size_t)x;
				return checkOut0[4 * i] == 0 && checkOut0[4 * i + 1] == 0 &&
				       checkOut0[4 * i + 2] == 0;
			};
			for (int y = 0; y < (int)kSize; y++) {
				for (int x = 0; x < (int)kSize; x++) {
					size_t i = (size_t)y * kSize + (size_t)x;
					bool black = (checkOut1[4 * i] == 0 && checkOut1[4 * i + 1] == 0 &&
					              checkOut1[4 * i + 2] == 0);
					if (black)
						black1++;
					// Foreground core (well inside the warped square)
					// must stay white.
					if (y >= 112 && y < 144 && x >= expFgL + 13 && x <= expFgR - 13) {
						bool white = (checkOut1[4 * i] > 240 && checkOut1[4 * i + 1] > 240 &&
						              checkOut1[4 * i + 2] > 240);
						if (!white)
							coreNonWhite++;
					}
					// Strict oracle away from holes (Chebyshev >= 2):
					// the dilation ring (<= 1px) is exempt.
					bool nearHole = false;
					for (int oy = -1; oy <= 1 && !nearHole; oy++) {
						for (int ox = -1; ox <= 1 && !nearHole; ox++) {
							int sx = x + ox;
							int sy = y + oy;
							if (sx < 0 || sy < 0 || sx >= (int)kSize || sy >= (int)kSize)
								continue;
							if (hole0(sx, sy))
								nearHole = true;
						}
					}
					if (nearHole)
						continue;
					int e = OracleChecker(x, y, splat1[i], dstInv, srcVP);
					if (e < 0)
						continue;
					for (int c = 0; c < 3; c++) {
						if (std::abs((int)checkOut1[4 * i + c] - e) > 1) {
							fillMismatch++;
							break;
						}
					}
				}
			}
			std::cout << "[oracle-fill] black=" << black1 << " coreNonWhite=" << coreNonWhite
			          << " fillMismatch=" << fillMismatch << std::endl;
			if (!(black1 == 0 && coreNonWhite == 0 && fillMismatch == 0)) {
				std::cerr << "FAIL: fill output violates oracle (black/core/mismatch)"
				          << std::endl;
				ok = false;
			}
			stbi_write_png((outDir + "/oracle_fill.png").c_str(), kSize, kSize, 4, checkOut1.data(),
			               kSize * 4);
		}
		stbi_write_png((outDir + "/fringe_fill.png").c_str(), kSize, kSize, 4, fringeFill.data(),
		               kSize * 4);
	}
	std::cout << (ok ? "VULKAN WARP TEST: PASS" : "VULKAN WARP TEST: FAIL") << std::endl;
	return ok ? 0 : 1;
}
