// ml/datasetgen/capture.cpp — headless sequence-frame capture for hole-fill data.
//
// Renders the EXACT demo scene (ObjScene + resources/shaders/demo.* + the
// OpenwarpApplication projection) at caller-supplied poses with OpenGL in a
// hidden window, then runs the REAL VulkanWarp splat/pull/fill path
// (build/libopenwarp_vulkan.a, fillMode 0 = black holes).
//
// One process handles many shots: a single GL context and a single
// VulkanWarp init amortized over the whole batch. Row order everywhere is
// GL-native (row 0 = frustum bottom); the python packager flips once.
//
// Usage:
//   capture --shots shots.json --rawdir raw/ --spvdir <spv> --resources <res>
//           --hole-min 0.02 --hole-max 0.15 [--width 1024 --height 1024]
//
// shots.json: [{"cand": int, "p0": {"pos":[x,y,z],"quat":[x,y,z,w]},
//                           "ptm1": {...}, "pt": {...}}, ...]
//   Poses are camera-to-world (Translation * Rotation), i.e. the
//   OpenwarpApplication::renderedCameraMatrix convention, matching
//   VulkanWarp::Pose (view = inverse camera).
//
// Per shot: render P0 -> Upload -> Warp(P0 -> Pt, coverage) -> hole
// fraction. Gate on [hole-min, hole-max]; accepted shots additionally render
// Ptm1 + Pt, Warp(Ptm1 -> Pt) for the teacher-forced prev frame, and dump
// raw binaries raw/c<cand>_{src_rgba,src_depth,warp_rgba,splat,
// tm1_rgba,tm1_depth,t_rgba,prev_rgba}.bin. Rejected shots dump nothing.
//
// stdout: "CAND <idx> hole=<f> splat_empty=<f> alpha_extra=<f> accepted=<0/1>"
// plus "GL_RENDERER=..." / "VK_DEVICE=..." provenance lines.
//
// Fidelity notes (see ml/datasetgen/README.md):
// - Camera matrix, projection (perspective(45,1,0.1,100)), clear color,
//   cull/depth state, FBO formats and readback types are copied from
//   OpenwarpApplication::{initGL,renderScene,doVulkanMeshWarp}.
// - GL context version: requests 3.0 like the app; Mesa grants 4.5 compat,
//   which is what compiles the #version 450 demo shaders.
//
// Build (from openwarp/, no CMake changes):
//   g++ -std=c++17 -O2 -DGLEW_NO_GLU -DGLEW_STATIC \
//     -Iinclude/eigen -Iinclude/glfw/include -Iinclude/glew-cmake/include \
//     -Isrc/openwarp ml/datasetgen/capture.cpp src/openwarp/util/obj.cpp \
//     build/libopenwarp_vulkan.a build/lib/libglewd.a \
//     build/include/glfw/src/libglfw3.a \
//     -lvulkan -lGL -lX11 -lXext -ldl -lm -lpthread -o ml/datasetgen/capture
//
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Dense>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "util/obj.hpp"
#include "util/shader_util.hpp"
#include "vulkan/vulkan_warp.hpp"

namespace {

constexpr uint32_t kDefaultW = 1024;
constexpr uint32_t kDefaultH = 1024;
constexpr uint32_t kSplatEmpty = 0xffffffffu;

struct PoseQT {
	Eigen::Vector3f pos = Eigen::Vector3f::Zero();
	Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
};

struct Shot {
	int cand = -1;
	PoseQT p0, ptm1, pt;
};

// Exact copy of OpenwarpApplication::perspective() (OpenwarpApplication.hpp).
Eigen::Matrix4f AppPerspective(float fovy, float aspect, float zNear, float zFar)
{
	float radf = fovy * (float)(M_PI / 180.0);
	float tanHalfFovy = tan(radf / 2.0f);
	Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
	res(0, 0) = 1.0f / (aspect * tanHalfFovy);
	res(1, 1) = 1.0f / (tanHalfFovy);
	res(2, 2) = -(zFar + zNear) / (zFar - zNear);
	res(3, 2) = -1.0f;
	res(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
	return res;
}

// Exact copy of OpenwarpApplication::createCameraMatrix().
Eigen::Matrix4f CameraMatrix(const Eigen::Vector3f& position,
                             const Eigen::Quaternionf& orientation)
{
	Eigen::Matrix4f cameraMatrix = Eigen::Matrix4f::Identity();
	cameraMatrix.block<3, 1>(0, 3) = position;
	cameraMatrix.block<3, 3>(0, 0) = orientation.toRotationMatrix();
	return cameraMatrix;
}

openwarp_vk::Pose ToVkPose(const PoseQT& p)
{
	openwarp_vk::Pose v{};
	v.pos[0] = p.pos.x();
	v.pos[1] = p.pos.y();
	v.pos[2] = p.pos.z();
	v.quat[0] = p.quat.x();
	v.quat[1] = p.quat.y();
	v.quat[2] = p.quat.z();
	v.quat[3] = p.quat.w();
	return v;
}

// Minimal JSON parsing for the fixed shots schema (no dependency).
// Expects: [{"cand":1,"p0":{"pos":[...],"quat":[...]},"ptm1":{...},"pt":{...}}, ...]
bool ParseDoubles(const std::string& s, size_t& i, std::vector<double>& out, int n)
{
	out.clear();
	while ((int)out.size() < n) {
		while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' ||
		                        s[i] == '\r' || s[i] == ',' || s[i] == '['))
			i++;
		if (i >= s.size())
			return false;
		char* end = nullptr;
		double v = std::strtod(s.c_str() + i, &end);
		if (end == s.c_str() + i)
			return false;
		out.push_back(v);
		i = (size_t)(end - s.c_str());
	}
	return true;
}

bool ParseShots(const std::string& path, std::vector<Shot>& shots)
{
	std::ifstream f(path);
	if (!f) {
		std::cerr << "cannot open " << path << std::endl;
		return false;
	}
	std::stringstream ss;
	ss << f.rdbuf();
	std::string s = ss.str();
	size_t i = 0;
	while (true) {
		size_t k = s.find("\"cand\"", i);
		if (k == std::string::npos)
			break;
		i = k + 6;
		while (i < s.size() && (s[i] < '0' || s[i] > '9') && s[i] != '-')
			i++;
		char* end = nullptr;
		long cand = std::strtol(s.c_str() + i, &end, 10);
		i = (size_t)(end - s.c_str());
		Shot sh;
		sh.cand = (int)cand;
		PoseQT* targets[3] = {&sh.p0, &sh.ptm1, &sh.pt};
		const char* names[3] = {"\"p0\"", "\"ptm1\"", "\"pt\""};
		for (int t = 0; t < 3; t++) {
			size_t kp = s.find(names[t], i);
			if (kp == std::string::npos) {
				std::cerr << "missing " << names[t] << std::endl;
				return false;
			}
			i = kp;
			size_t kpos = s.find("\"pos\"", i);
			size_t kquat = s.find("\"quat\"", i);
			if (kpos == std::string::npos || kquat == std::string::npos) {
				std::cerr << "missing pos/quat" << std::endl;
				return false;
			}
			i = s.find('[', kpos);
			if (i == std::string::npos)
				return false;
			std::vector<double> v;
			if (!ParseDoubles(s, i, v, 3))
				return false;
			targets[t]->pos = Eigen::Vector3f((float)v[0], (float)v[1], (float)v[2]);
			i = s.find('[', kquat);
			if (i == std::string::npos)
				return false;
			if (!ParseDoubles(s, i, v, 4))
				return false;
			// JSON stores xyzw; Eigen ctor takes (w,x,y,z).
			targets[t]->quat = Eigen::Quaternionf((float)v[3], (float)v[0], (float)v[1],
			                                       (float)v[2]);
			targets[t]->quat.normalize();
		}
		shots.push_back(sh);
	}
	return true;
}

bool WriteBin(const std::string& path, const void* data, size_t bytes)
{
	std::ofstream f(path, std::ios::binary);
	if (!f)
		return false;
	f.write((const char*)data, (std::streamsize)bytes);
	return (bool)f;
}

struct GL {
	GLFWwindow* window = nullptr;
	GLuint renderTexture = 0;
	GLuint depthTexture = 0;
	GLuint renderFBO = 0;
	GLint demoProgram = 0;
	GLint uModelview = -1;
	GLint uProjection = -1;
	GLuint vao = 0;
	Openwarp::ObjScene scene;
	Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
	uint32_t W = kDefaultW, H = kDefaultH;
	std::vector<uint8_t> rgba;
	std::vector<float> depth;
};

bool InitGL(GL& g, uint32_t W, uint32_t H, const std::string& resources)
{
	g.W = W;
	g.H = H;
	if (!glfwInit()) {
		std::cerr << "glfwInit failed" << std::endl;
		return false;
	}
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	// Same request as OpenwarpApplication (Mesa grants 4.5 compat, which
	// compiles the #version 450 demo shaders).
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	g.window = glfwCreateWindow((int)W, (int)H, "capture", nullptr, nullptr);
	if (!g.window) {
		std::cerr << "glfwCreateWindow failed" << std::endl;
		return false;
	}
	glfwMakeContextCurrent(g.window);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "glewInit failed" << std::endl;
		return false;
	}

	// Color + depth render textures (mirrors createRenderTexture()).
	glGenTextures(1, &g.renderTexture);
	glBindTexture(GL_TEXTURE_2D, g.renderTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, (GLsizei)W, (GLsizei)H, 0, GL_RGB,
	             GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &g.depthTexture);
	glBindTexture(GL_TEXTURE_2D, g.depthTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, (GLsizei)W, (GLsizei)H, 0,
	             GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenFramebuffers(1, &g.renderFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, g.renderFBO);
	glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g.renderTexture, 0);
	glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, g.depthTexture, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cerr << "FBO incomplete" << std::endl;
		return false;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::string dir = resources;
	if (!dir.empty() && dir.back() != '/')
		dir += '/';
	g.scene = Openwarp::ObjScene(dir, "scene.obj");
	if (!g.scene.successfully_loaded_model) {
		std::cerr << "scene load failed" << std::endl;
		return false;
	}

	g.projection = AppPerspective(45.0f, (float)W / (float)H, 0.1f, 100.0f);

	g.demoProgram = init_and_link((dir + "shaders/demo.vert").c_str(),
	                              (dir + "shaders/demo.frag").c_str());
	g.uModelview = glGetUniformLocation(g.demoProgram, "u_modelview");
	g.uProjection = glGetUniformLocation(g.demoProgram, "u_projection");
	glGenVertexArrays(1, &g.vao);

	g.rgba.resize((size_t)W * H * 4);
	g.depth.resize((size_t)W * H);
	return true;
}

// Render the demo scene at `pose` (mirrors renderScene()).
void RenderAt(GL& g, const PoseQT& pose)
{
	glBindVertexArray(g.vao);
	glUseProgram(g.demoProgram);
	glBindFramebuffer(GL_FRAMEBUFFER, g.renderFBO);
	glViewport(0, 0, (GLsizei)g.W, (GLsizei)g.H);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glClearDepth(1);
	Eigen::Matrix4f cam = CameraMatrix(pose.pos, pose.quat);
	Eigen::Matrix4f view = cam.inverse();
	glUniformMatrix4fv(g.uModelview, 1, GL_FALSE, view.data());
	glUniformMatrix4fv(g.uProjection, 1, GL_FALSE, g.projection.data());
	glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	g.scene.Draw();
	glFlush();
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, g.renderFBO);
	glReadPixels(0, 0, (GLsizei)g.W, (GLsizei)g.H, GL_RGBA, GL_UNSIGNED_BYTE,
	             g.rgba.data());
	glReadPixels(0, 0, (GLsizei)g.W, (GLsizei)g.H, GL_DEPTH_COMPONENT, GL_FLOAT,
	             g.depth.data());
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

} // namespace

int main(int argc, char** argv)
{
	std::string shotsPath, rawDir, spvDir, resDir;
	float holeMin = 0.02f, holeMax = 0.15f;
	uint32_t W = kDefaultW, H = kDefaultH;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto need = [&](std::string& out) {
			if (i + 1 >= argc) {
				std::cerr << "missing value for " << a << std::endl;
				return false;
			}
			out = argv[++i];
			return true;
		};
		if (a == "--shots") {
			if (!need(shotsPath))
				return 1;
		} else if (a == "--rawdir") {
			if (!need(rawDir))
				return 1;
		} else if (a == "--spvdir") {
			if (!need(spvDir))
				return 1;
		} else if (a == "--resources") {
			if (!need(resDir))
				return 1;
		} else if (a == "--hole-min") {
			std::string v;
			if (!need(v))
				return 1;
			holeMin = std::stof(v);
		} else if (a == "--hole-max") {
			std::string v;
			if (!need(v))
				return 1;
			holeMax = std::stof(v);
		} else if (a == "--width") {
			std::string v;
			if (!need(v))
				return 1;
			W = (uint32_t)std::stoul(v);
		} else if (a == "--height") {
			std::string v;
			if (!need(v))
				return 1;
			H = (uint32_t)std::stoul(v);
		} else {
			std::cerr << "unknown arg " << a << std::endl;
			return 1;
		}
	}
	if (shotsPath.empty() || rawDir.empty() || spvDir.empty() || resDir.empty()) {
		std::cerr << "need --shots --rawdir --spvdir --resources" << std::endl;
		return 1;
	}

	std::vector<Shot> shots;
	if (!ParseShots(shotsPath, shots) || shots.empty()) {
		std::cerr << "no shots parsed from " << shotsPath << std::endl;
		return 1;
	}

	GL g;
	if (!InitGL(g, W, H, resDir))
		return 1;
	std::cout << "GL_RENDERER=" << (const char*)glGetString(GL_RENDERER) << std::endl;

	openwarp_vk::VulkanWarp warp;
	{
		std::string error;
		if (!warp.Init(W, H, spvDir, &error)) {
			std::cerr << "VulkanWarp::Init failed: " << error << std::endl;
			return 1;
		}
	}
	std::cout << "VK_DEVICE=" << warp.DeviceName() << std::endl;
	warp.SetFillMode(0);

	openwarp_vk::Mat4 proj{};
	std::memcpy(proj, g.projection.data(), sizeof(proj));

	std::vector<uint8_t> out;
	size_t px = (size_t)W * H;
	char base[64];

	for (const Shot& sh : shots) {
		// Main pull pass: src P0 -> target Pt.
		RenderAt(g, sh.p0);
		std::vector<uint8_t> srcRgba = g.rgba;
		std::vector<float> srcDepth = g.depth;
		if (!warp.Upload(srcRgba.data(), srcDepth.data())) {
			std::cerr << "Upload failed: " << warp.LastError() << std::endl;
			return 1;
		}
		if (!warp.Warp(ToVkPose(sh.p0), proj, ToVkPose(sh.pt), proj, 0.0f, out, true)) {
			std::cerr << "Warp failed: " << warp.LastError() << std::endl;
			return 1;
		}
		const std::vector<uint32_t>& splat = warp.LastSplatPlane();
		size_t nEmpty = 0, nAlphaHole = 0, nHole = 0;
		for (size_t i = 0; i < px; i++) {
			bool e = (splat[i] == kSplatEmpty);
			bool a = (out[4 * i + 3] < 128);
			if (e)
				nEmpty++;
			if (a && !e)
				nAlphaHole++;
			if (e || a)
				nHole++;
		}
		float holeFrac = (float)nHole / (float)px;
		float splatEmptyFrac = (float)nEmpty / (float)px;
		float alphaExtraFrac = (float)nAlphaHole / (float)px;
		bool accepted = (holeFrac >= holeMin && holeFrac <= holeMax);
		if (accepted) {
			// Teacher-forced prev: true Ptm1 -> Pt through the same warp.
			RenderAt(g, sh.ptm1);
			std::vector<uint8_t> tm1Rgba = g.rgba;
			std::vector<float> tm1Depth = g.depth;
			RenderAt(g, sh.pt);
			std::vector<uint8_t> tRgba = g.rgba;
			if (!warp.Upload(tm1Rgba.data(), tm1Depth.data())) {
				std::cerr << "prev Upload failed" << std::endl;
				return 1;
			}
			std::vector<uint8_t> prev;
			if (!warp.Warp(ToVkPose(sh.ptm1), proj, ToVkPose(sh.pt), proj, 0.0f,
			               prev, false)) {
				std::cerr << "prev Warp failed" << std::endl;
				return 1;
			}
			std::snprintf(base, sizeof(base), "%s/c%06d", rawDir.c_str(), sh.cand);
			std::string b = base;
			if (!WriteBin(b + "_src_rgba.bin", srcRgba.data(), px * 4) ||
			    !WriteBin(b + "_src_depth.bin", srcDepth.data(), px * 4) ||
			    !WriteBin(b + "_warp_rgba.bin", out.data(), px * 4) ||
			    !WriteBin(b + "_splat.bin", splat.data(), px * 4) ||
			    !WriteBin(b + "_tm1_rgba.bin", tm1Rgba.data(), px * 4) ||
			    !WriteBin(b + "_tm1_depth.bin", tm1Depth.data(), px * 4) ||
			    !WriteBin(b + "_t_rgba.bin", tRgba.data(), px * 4) ||
			    !WriteBin(b + "_prev_rgba.bin", prev.data(), px * 4)) {
				std::cerr << "write failed for cand " << sh.cand << std::endl;
				return 1;
			}
		}
		std::cout << "CAND " << sh.cand << " hole=" << holeFrac
		          << " splat_empty=" << splatEmptyFrac << " alpha_extra=" << alphaExtraFrac
		          << " accepted=" << (accepted ? 1 : 0) << std::endl;
	}
	return 0;
}
