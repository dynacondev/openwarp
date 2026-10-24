// Openwarp Vulkan compute warp implementation.
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan_warp.hpp"

#include <vulkan/vulkan.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

namespace openwarp_vk {
namespace {

constexpr float kDepthScale = 16777215.0f;
constexpr uint32_t kSplatEmpty = 0xffffffffu;

struct SplatUBO {
	float renderInvVP[16];
	float warpVP[16];
};

struct PullUBO {
	float dstInv[16];
	float srcVP[16];
	// x: fill weight exponent p (w = pow(depth, p)); yzw: reserved.
	float tune[4];
};

} // namespace

bool FrustumGL(float angleLeft, float angleRight, float angleDown, float angleUp,
               float nearZ, float farZ, Mat4 out)
{
	if (!(std::isfinite(nearZ) && std::isfinite(farZ)) || nearZ <= 0.0f || farZ <= nearZ)
		return false;
	float l = nearZ * tanf(angleLeft);
	float r = nearZ * tanf(angleRight);
	float b = nearZ * tanf(angleDown);
	float t = nearZ * tanf(angleUp);
	if (!(std::isfinite(l) && std::isfinite(r) && std::isfinite(b) && std::isfinite(t)) || !(r > l && t > b))
		return false;
	float rl = 2.0f * nearZ / (r - l);
	float tb = 2.0f * nearZ / (t - b);
	float rlMid = (r + l) / (r - l);
	float tbMid = (t + b) / (t - b);
	float fn = -(farZ + nearZ) / (farZ - nearZ);
	float fn2 = -2.0f * farZ * nearZ / (farZ - nearZ);
	// Column 0
	out[0] = rl;
	out[1] = 0;
	out[2] = 0;
	out[3] = 0;
	// Column 1
	out[4] = 0;
	out[5] = tb;
	out[6] = 0;
	out[7] = 0;
	// Column 2
	out[8] = rlMid;
	out[9] = tbMid;
	out[10] = fn;
	out[11] = -1;
	// Column 3
	out[12] = 0;
	out[13] = 0;
	out[14] = fn2;
	out[15] = 0;
	return true;
}

bool PerspectiveGL(float fovyDeg, float aspect, float nearZ, float farZ, Mat4 out)
{
	if (!(aspect > 0.0f) || !(farZ > nearZ))
		return false;
	float rad = fovyDeg * static_cast<float>(M_PI) / 180.0f;
	float tanHalf = tanf(rad / 2.0f);
	float l = -tanHalf * aspect * nearZ;
	float r = tanHalf * aspect * nearZ;
	float b = -tanHalf * nearZ;
	float t = tanHalf * nearZ;
	// Reuse the asymmetric constructor with slope angles.
	return FrustumGL(atanf(l / nearZ), atanf(r / nearZ), atanf(b / nearZ), atanf(t / nearZ),
	                 nearZ, farZ, out);
}

void CameraFromPose(const Pose& pose, Mat4 out)
{
	Eigen::Quaternionf q(pose.quat[3], pose.quat[0], pose.quat[1], pose.quat[2]);
	Eigen::Vector3f p(pose.pos[0], pose.pos[1], pose.pos[2]);
	Eigen::Matrix4f m = (Eigen::Translation3f(p) * q).matrix();
	std::memcpy(out, m.data(), sizeof(float) * 16);
}

void Multiply(const Mat4 a, const Mat4 b, Mat4 out)
{
	Eigen::Matrix4f r = Eigen::Map<const Eigen::Matrix4f>(a) * Eigen::Map<const Eigen::Matrix4f>(b);
	std::memcpy(out, r.data(), sizeof(float) * 16);
}

bool Inverse(const Mat4 m, Mat4 out)
{
	Eigen::Map<const Eigen::Matrix4f> src(m);
	if (std::abs(src.determinant()) < 1e-12f)
		return false;
	Eigen::Matrix4f r = src.inverse();
	std::memcpy(out, r.data(), sizeof(float) * 16);
	return true;
}

// --- Vulkan state ---

struct VulkanWarp::State {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;
	uint32_t queueFamily = 0;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;

	struct Image {
		VkImage handle = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
		// Per-level layouts (the warp target carries a mip chain whose
		// levels diverge during blits). Mirrors layout for 1-level imgs.
		std::vector<VkImageLayout> levels;
	} srcColor, srcDepth, splat, dst;

	// Sampled view of dst mip levels 1..N for the fill pass.
	VkImageView dstMipView = VK_NULL_HANDLE;
	uint32_t mipLevels = 1;

	struct Buffer {
		VkBuffer handle = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		void* mapped = nullptr;
		VkDeviceSize size = 0;
	} uboSplat, uboPull, staging;

	VkSampler colorSampler = VK_NULL_HANDLE;
	VkSampler depthSampler = VK_NULL_HANDLE;
	VkDescriptorSetLayout splatLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout pullLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout fillLayout = VK_NULL_HANDLE;
	VkDescriptorPool descPool = VK_NULL_HANDLE;
	VkDescriptorSet splatSet = VK_NULL_HANDLE;
	VkDescriptorSet pullSet = VK_NULL_HANDLE;
	VkDescriptorSet fillSet = VK_NULL_HANDLE;
	VkPipelineLayout splatPipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout pullPipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout fillPipeLayout = VK_NULL_HANDLE;
	VkPipeline splatPipe = VK_NULL_HANDLE;
	VkPipeline pullPipe = VK_NULL_HANDLE;
	VkPipeline fillPipe = VK_NULL_HANDLE;

	// Optional GPU timestamp queries (stage timing). Created when the
	// queue family supports them; all use is guarded by hasTimestamps.
	VkQueryPool queryPool = VK_NULL_HANDLE;
	bool hasTimestamps = false;
	double timestampPeriodMs = 0.0;

	bool srcUploaded = false;
};

namespace {

bool Check(VkResult r, const char* what, std::string* error)
{
	if (r == VK_SUCCESS)
		return true;
	if (error)
		*error = std::string(what) + " (VkResult " + std::to_string(r) + ")";
	return false;
}

bool LoadSpv(const std::string& path, std::vector<uint32_t>& words, std::string* error)
{
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		if (error)
			*error = "cannot open SPIR-V: " + path;
		return false;
	}
	std::streamsize n = f.tellg();
	f.seekg(0, std::ios::beg);
	if (n <= 0 || (n % 4) != 0) {
		if (error)
			*error = "bad SPIR-V size: " + path;
		return false;
	}
	words.resize(static_cast<size_t>(n) / 4);
	return static_cast<bool>(f.read(reinterpret_cast<char*>(words.data()), n));
}

uint32_t FindMemory(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want)
{
	VkPhysicalDeviceMemoryProperties props{};
	vkGetPhysicalDeviceMemoryProperties(phys, &props);
	for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & want) == want)
			return i;
	}
	return UINT32_MAX;
}

bool CreateBuffer(VulkanWarp::State* s, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memFlags, VulkanWarp::State::Buffer& b, std::string* error)
{
	VkBufferCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	ci.size = size;
	ci.usage = usage;
	ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (!Check(vkCreateBuffer(s->device, &ci, nullptr, &b.handle), "vkCreateBuffer", error))
		return false;
	VkMemoryRequirements req{};
	vkGetBufferMemoryRequirements(s->device, b.handle, &req);
	uint32_t memIdx = FindMemory(s->phys, req.memoryTypeBits, memFlags);
	if (memIdx == UINT32_MAX) {
		if (error)
			*error = "no suitable buffer memory type";
		return false;
	}
	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = req.size;
	ai.memoryTypeIndex = memIdx;
	if (!Check(vkAllocateMemory(s->device, &ai, nullptr, &b.memory), "vkAllocateMemory(buffer)", error))
		return false;
	if (!Check(vkBindBufferMemory(s->device, b.handle, b.memory, 0), "vkBindBufferMemory", error))
		return false;
	b.size = size;
	if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
		if (!Check(vkMapMemory(s->device, b.memory, 0, size, 0, &b.mapped), "vkMapMemory", error))
			return false;
	}
	return true;
}

bool CreateImage(VulkanWarp::State* s, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
                 VulkanWarp::State::Image& img, std::string* error, uint32_t mipLevels = 1)
{
	VkImageCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ci.imageType = VK_IMAGE_TYPE_2D;
	ci.format = format;
	ci.extent = {w, h, 1};
	ci.mipLevels = mipLevels;
	ci.arrayLayers = 1;
	ci.samples = VK_SAMPLE_COUNT_1_BIT;
	ci.tiling = VK_IMAGE_TILING_OPTIMAL;
	ci.usage = usage;
	ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (!Check(vkCreateImage(s->device, &ci, nullptr, &img.handle), "vkCreateImage", error))
		return false;
	VkMemoryRequirements req{};
	vkGetImageMemoryRequirements(s->device, img.handle, &req);
	uint32_t memIdx = FindMemory(s->phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (memIdx == UINT32_MAX)
		memIdx = FindMemory(s->phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	if (memIdx == UINT32_MAX) {
		if (error)
			*error = "no suitable image memory type";
		return false;
	}
	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = req.size;
	ai.memoryTypeIndex = memIdx;
	if (!Check(vkAllocateMemory(s->device, &ai, nullptr, &img.memory), "vkAllocateMemory(image)", error))
		return false;
	if (!Check(vkBindImageMemory(s->device, img.handle, img.memory, 0), "vkBindImageMemory", error))
		return false;
	VkImageViewCreateInfo vi{};
	vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vi.image = img.handle;
	vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vi.format = format;
	vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	if (!Check(vkCreateImageView(s->device, &vi, nullptr, &img.view), "vkCreateImageView", error))
		return false;
	img.layout = VK_IMAGE_LAYOUT_UNDEFINED;
	img.levels.assign(mipLevels, VK_IMAGE_LAYOUT_UNDEFINED);
	return true;
}

// Whole-image Transition: kept for 1-level images. Mipped images use
// TransitionLevels below.
void Transition(VulkanWarp::State* s, VulkanWarp::State::Image& img, VkImageLayout next,
                VkAccessFlags dstAccess, VkPipelineStageFlags dstStage, uint32_t baseLevel = 0,
                uint32_t levelCount = 1)
{
	// All tracked levels in range must already agree (true by
	// construction: ranges are only ever transitioned uniformly).
	// Skip when the whole range is already there.
	bool already = true;
	for (uint32_t i = baseLevel; i < baseLevel + levelCount && i < img.levels.size(); i++) {
		if (img.levels[i] != next) {
			already = false;
			break;
		}
	}
	if (already)
		return;
	VkImageMemoryBarrier bar{};
	bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	bar.oldLayout = img.levels[baseLevel];
	bar.newLayout = next;
	bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.image = img.handle;
	bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseLevel, levelCount, 0, 1};
	bar.dstAccessMask = dstAccess;
	// Source masks derived from the tracked current layout.
	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	switch (img.levels[baseLevel]) {
	case VK_IMAGE_LAYOUT_UNDEFINED:
		bar.srcAccessMask = 0;
		srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		break;
	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
		bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		break;
	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
		bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		break;
	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
		bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		break;
	default:
		bar.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		break;
	}
	vkCmdPipelineBarrier(s->cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &bar);
	for (uint32_t i = baseLevel; i < baseLevel + levelCount && i < img.levels.size(); i++)
		img.levels[i] = next;
	img.layout = next;
}

bool BeginOnce(VulkanWarp::State* s, std::string* error)
{
	if (!Check(vkResetFences(s->device, 1, &s->fence), "vkResetFences", error))
		return false;
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	return Check(vkBeginCommandBuffer(s->cmd, &bi), "vkBeginCommandBuffer", error);
}

bool SubmitWait(VulkanWarp::State* s, std::string* error)
{
	if (!Check(vkEndCommandBuffer(s->cmd), "vkEndCommandBuffer", error))
		return false;
	VkSubmitInfo si{};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &s->cmd;
	if (!Check(vkQueueSubmit(s->queue, 1, &si, s->fence), "vkQueueSubmit", error))
		return false;
	return Check(vkWaitForFences(s->device, 1, &s->fence, VK_TRUE, UINT64_MAX), "vkWaitForFences",
	             error);
}

} // namespace

bool VulkanWarp::Fail(std::string* error, const std::string& msg)
{
	if (error)
		*error = msg;
	lastError_ = msg;
	return false;
}

bool VulkanWarp::Init(uint32_t width, uint32_t height, const std::string& spvDir, std::string* error,
                    const char* glRenderer)
{
	if (initialized_)
		Shutdown();
	if (width == 0 || height == 0)
		return Fail(error, "warp image size must be non-zero");

	state_ = new State();
	State* s = state_;

	// Instance (no extensions needed headless). Validation is strictly
	// opt-in: the layer instruments every call and would tax the live
	// path. Set OPENWARP_VK_VALIDATION=1 to enable (no debug callback;
	// check stderr for layer output).
	bool wantValidation = std::getenv("OPENWARP_VK_VALIDATION") != nullptr;
	const char* layer = "VK_LAYER_KHRONOS_validation";
	uint32_t layerCount = 0;
	if (wantValidation) {
		uint32_t n = 0;
		if (vkEnumerateInstanceLayerProperties(&n, nullptr) == VK_SUCCESS && n > 0) {
			std::vector<VkLayerProperties> props(n);
			if (vkEnumerateInstanceLayerProperties(&n, props.data()) == VK_SUCCESS) {
				for (auto& p : props) {
					if (std::strcmp(p.layerName, layer) == 0) {
						layerCount = 1;
						break;
					}
				}
			}
		}
		if (!layerCount)
			std::cout << "[vulkan] OPENWARP_VK_VALIDATION=1 but layer not present" << std::endl;
	}
	VkInstanceCreateInfo ii{};
	ii.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	VkApplicationInfo app{};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "openwarp-vulkan-warp";
	app.apiVersion = VK_API_VERSION_1_0;
	ii.pApplicationInfo = &app;
	ii.enabledLayerCount = layerCount;
	ii.ppEnabledLayerNames = layerCount ? &layer : nullptr;
	if (!Check(vkCreateInstance(&ii, nullptr, &s->instance), "vkCreateInstance", error))
		return false;
	std::cout << "[vulkan] validation " << (layerCount ? "ON" : "off") << std::endl;

	// Physical device: follow the GL renderer when known (same GPU for
	// both APIs => frames never cross PCIe), discrete GPU as tiebreak,
	// OPENWARP_VK_DEVICE=<index> as final override.
	auto lower = [](const std::string& in) {
		std::string v(in);
		for (auto& c : v)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return v;
	};
	auto rendererMatches = [&](uint32_t vendorId, const char* deviceName) {
		if (!glRenderer || !glRenderer[0] || !deviceName)
			return false;
		std::string r = lower(glRenderer);
		std::string d = lower(deviceName);
		// Direct containment either way (llvmpipe, Mesa Intel/AMD with
		// near-identical strings).
		if (r.find(d) != std::string::npos || d.find(r) != std::string::npos)
			return true;
		auto has = [&](const char* t) { return r.find(t) != std::string::npos; };
		switch (vendorId) {
		case 0x10de: return has("nvidia") || has("geforce") || has("quadro") || has("tesla");
		case 0x8086: return has("intel") || has("uhd") || has("iris") || has("hd graphics");
		case 0x1002: return has("amd") || has("radeon");
		default: return false;
		}
	};
	if (glRenderer && glRenderer[0])
		std::cout << "[vulkan] GL renderer hint: " << glRenderer << std::endl;
	uint32_t devCount = 0;
	if (!Check(vkEnumeratePhysicalDevices(s->instance, &devCount, nullptr),
	           "vkEnumeratePhysicalDevices", error) ||
	    devCount == 0)
		return Fail(error, "no Vulkan physical devices");
	std::vector<VkPhysicalDevice> devs(devCount);
	vkEnumeratePhysicalDevices(s->instance, &devCount, devs.data());
	auto devScore = [](VkPhysicalDeviceType t) {
		switch (t) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
		case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
		default: return 0;
		}
	};
	int forcedIdx = -1;
	if (const char* env = std::getenv("OPENWARP_VK_DEVICE"))
		forcedIdx = std::atoi(env);
	int best = -1;
	int bestScore = -1;
	uint32_t bestFamily = 0;
	for (uint32_t di = 0; di < devCount; di++) {
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(devs[di], &props);
		uint32_t qCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &qCount, nullptr);
		std::vector<VkQueueFamilyProperties> qs(qCount);
		vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &qCount, qs.data());
		int fam = -1;
		for (uint32_t i = 0; i < qCount; i++) {
			if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				fam = static_cast<int>(i);
				break;
			}
		}
		const char* tn = "other";
		switch (props.deviceType) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: tn = "discrete"; break;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: tn = "integrated"; break;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: tn = "virtual"; break;
		case VK_PHYSICAL_DEVICE_TYPE_CPU: tn = "cpu"; break;
		default: break;
		}
		std::cout << "[vulkan] device " << di << ": " << props.deviceName << " (" << tn
		          << ", driver 0x" << std::hex << props.driverVersion << std::dec
		          << (fam >= 0 ? ", compute ok" : ", NO compute queue") << ")" << std::endl;
		if (fam < 0)
			continue;
		// Explicit override wins; otherwise a GL-renderer match dominates
		// and the discrete preference only breaks ties within a vendor.
		int score = devScore(props.deviceType);
		const char* why = "type";
		if (rendererMatches(props.vendorID, props.deviceName)) {
			score += 100;
			why = "GL-renderer match";
		}
		if (forcedIdx == static_cast<int>(di)) {
			score = 1000;
			why = "OPENWARP_VK_DEVICE override";
		}
		std::cout << "[vulkan]   -> score " << score << " (" << why << ")" << std::endl;
		if (score > bestScore) {
			bestScore = score;
			best = static_cast<int>(di);
			bestFamily = static_cast<uint32_t>(fam);
		}
	}
	if (best < 0)
		return Fail(error, "no compute-capable Vulkan queue family");
	s->phys = devs[static_cast<uint32_t>(best)];
	s->queueFamily = bestFamily;
	{
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(s->phys, &props);
		deviceName_ = props.deviceName;
		std::cout << "[vulkan] using device " << best << ": " << props.deviceName
		          << " (queue family " << s->queueFamily << ")" << std::endl;
	}

	// Format sanity: the warp needs these feature bits on OPTIMAL images.
	{
		VkFormatProperties fp{};
		vkGetPhysicalDeviceFormatProperties(s->phys, VK_FORMAT_R8G8B8A8_UNORM, &fp);
		if (!(fp.optimalTilingFeatures &
		      (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
		       VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
		       VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)))
			return Fail(error, "R8G8B8A8_UNORM lacks sampled+storage+blit+linear-filter support");
		vkGetPhysicalDeviceFormatProperties(s->phys, VK_FORMAT_R32_SFLOAT, &fp);
		if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
			return Fail(error, "R32_SFLOAT lacks sampled support");
		vkGetPhysicalDeviceFormatProperties(s->phys, VK_FORMAT_R32_UINT, &fp);
		if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
			return Fail(error, "R32_UINT lacks storage support");
	}

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qi{};
	qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qi.queueFamilyIndex = s->queueFamily;
	qi.queueCount = 1;
	qi.pQueuePriorities = &prio;
	VkDeviceCreateInfo di{};
	di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	di.queueCreateInfoCount = 1;
	di.pQueueCreateInfos = &qi;
	if (!Check(vkCreateDevice(s->phys, &di, nullptr, &s->device), "vkCreateDevice", error))
		return false;
	vkGetDeviceQueue(s->device, s->queueFamily, 0, &s->queue);

	VkCommandPoolCreateInfo pi{};
	pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pi.queueFamilyIndex = s->queueFamily;
	pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	if (!Check(vkCreateCommandPool(s->device, &pi, nullptr, &s->pool), "vkCreateCommandPool", error))
		return false;
	VkCommandBufferAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.commandPool = s->pool;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandBufferCount = 1;
	if (!Check(vkAllocateCommandBuffers(s->device, &ai, &s->cmd), "vkAllocateCommandBuffers", error))
		return false;
	VkFenceCreateInfo fi{};
	fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	if (!Check(vkCreateFence(s->device, &fi, nullptr, &s->fence), "vkCreateFence", error))
		return false;

	// Images.
	if (!CreateImage(s, width, height, VK_FORMAT_R8G8B8A8_UNORM,
	                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, s->srcColor,
	                 error))
		return false;
	if (!CreateImage(s, width, height, VK_FORMAT_R32_SFLOAT,
	                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, s->srcDepth,
	                 error))
		return false;
	if (!CreateImage(s, width, height, VK_FORMAT_R32_UINT,
	                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
	                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
	                 s->splat, error))
		return false;
	// Warp target: full mip chain (pull writes level 0, blits build the
	// rest for the fill pass, level 0 is downloaded). Needs STORAGE
	// (pull/fill) + SAMPLED (present copy? no — sampled mip view below)
	// + TRANSFER (uploads? no — blits + download).
	s->mipLevels = 1;
	for (uint32_t m = (uint32_t)std::max(width, height); m > 1; m >>= 1)
		s->mipLevels++;
	if (!CreateImage(s, width, height, VK_FORMAT_R8G8B8A8_UNORM,
	                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
	                 s->dst, error, s->mipLevels))
		return false;
	// Sampled view of levels 1..N for the fill pass (level 0 is written
	// while these are read, so it must not be in the sampled view).
	{
		VkImageViewCreateInfo vi{};
		vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		vi.image = s->dst.handle;
		vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
		vi.format = VK_FORMAT_R8G8B8A8_UNORM;
		vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1, s->mipLevels - 1, 0, 1};
		if (!Check(vkCreateImageView(s->device, &vi, nullptr, &s->dstMipView), "dst mip view",
		           error))
			return false;
	}

	// Buffers: UBOs + one staging buffer. Staging holds TWO planes
	// (color at +0, depth at +planeBytes): both upload copies are
	// recorded into a single command buffer, so they must not alias the
	// same staging bytes (the GPU reads them after the CPU has moved on).
	VkDeviceSize planeBytes = static_cast<VkDeviceSize>(width) * height * 4;
	if (!CreateBuffer(s, sizeof(SplatUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                  s->uboSplat, error))
		return false;
	if (!CreateBuffer(s, sizeof(PullUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                  s->uboPull, error))
		return false;
	if (!CreateBuffer(s, 2 * planeBytes,
	                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                  s->staging, error))
		return false;

	// Samplers: color linear/border-black, depth nearest/edge (dynacon parity).
	VkSamplerCreateInfo sci{};
	sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sci.magFilter = VK_FILTER_LINEAR;
	sci.minFilter = VK_FILTER_LINEAR;
	sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	// maxLod must be uncapped: the fill pass samples explicit mip LODs
	// via textureLod (the default 0.0 would clamp every tap to level 0).
	sci.minLod = 0.0f;
	sci.maxLod = VK_LOD_CLAMP_NONE;
	// Color sampler: linear, edge-clamped. Out-of-bounds taps are
	// already rejected in-shader (inBounds), so border-black would only
	// darken the valid 1-px rim where linear filtering touches the edge.
	sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	if (!Check(vkCreateSampler(s->device, &sci, nullptr, &s->colorSampler), "color sampler", error))
		return false;
	sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
	sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	if (!Check(vkCreateSampler(s->device, &sci, nullptr, &s->depthSampler), "depth sampler", error))
		return false;

	// Descriptor layouts + pool + sets (written once; only UBO contents change per warp).
	VkDescriptorSetLayoutBinding splatBindings[3]{};
	splatBindings[0].binding = 0;
	splatBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	splatBindings[0].descriptorCount = 1;
	splatBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	splatBindings[1].binding = 1;
	splatBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	splatBindings[1].descriptorCount = 1;
	splatBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	splatBindings[2].binding = 2;
	splatBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	splatBindings[2].descriptorCount = 1;
	splatBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	VkDescriptorSetLayoutCreateInfo li{};
	li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	li.bindingCount = 3;
	li.pBindings = splatBindings;
	if (!Check(vkCreateDescriptorSetLayout(s->device, &li, nullptr, &s->splatLayout),
	           "splat layout", error))
		return false;

	VkDescriptorSetLayoutBinding pullBindings[4]{};
	pullBindings[0].binding = 0;
	pullBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	pullBindings[0].descriptorCount = 1;
	pullBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pullBindings[1].binding = 1;
	pullBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pullBindings[1].descriptorCount = 1;
	pullBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pullBindings[2].binding = 2;
	pullBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	pullBindings[2].descriptorCount = 1;
	pullBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pullBindings[3].binding = 3;
	pullBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	pullBindings[3].descriptorCount = 1;
	pullBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	li.bindingCount = 4;
	li.pBindings = pullBindings;
	if (!Check(vkCreateDescriptorSetLayout(s->device, &li, nullptr, &s->pullLayout), "pull layout",
	           error))
		return false;

	// Fill layout: dst storage (read-write) + splat storage + dst
	// mip-chain sampled (single view covering levels 1..N).
	VkDescriptorSetLayoutBinding fillBindings[3]{};
	for (uint32_t i = 0; i < 3; i++) {
		fillBindings[i].binding = i;
		fillBindings[i].descriptorCount = 1;
		fillBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	fillBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	fillBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	fillBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	li.bindingCount = 3;
	li.pBindings = fillBindings;
	if (!Check(vkCreateDescriptorSetLayout(s->device, &li, nullptr, &s->fillLayout), "fill layout",
	           error))
		return false;

	VkDescriptorPoolSize poolSizes[3]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[0].descriptorCount = 3; // splat, pull, fill
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[1].descriptorCount = 2;
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[2].descriptorCount = 5; // splat, pull x2, fill x2
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = 3;
	dpi.poolSizeCount = 3;
	dpi.pPoolSizes = poolSizes;
	if (!Check(vkCreateDescriptorPool(s->device, &dpi, nullptr, &s->descPool), "desc pool", error))
		return false;
	VkDescriptorSetAllocateInfo alloc{};
	alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc.descriptorPool = s->descPool;
	alloc.descriptorSetCount = 1;
	alloc.pSetLayouts = &s->splatLayout;
	if (!Check(vkAllocateDescriptorSets(s->device, &alloc, &s->splatSet), "alloc splat set", error))
		return false;
	alloc.pSetLayouts = &s->pullLayout;
	if (!Check(vkAllocateDescriptorSets(s->device, &alloc, &s->pullSet), "alloc pull set", error))
		return false;
	alloc.pSetLayouts = &s->fillLayout;
	if (!Check(vkAllocateDescriptorSets(s->device, &alloc, &s->fillSet), "alloc fill set", error))
		return false;

	auto writeCombined = [&](VkDescriptorSet set, uint32_t binding, VkSampler sampler,
	                         VkImageView view,
	                         VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		VkDescriptorImageInfo info{};
		info.sampler = sampler;
		info.imageView = view;
		info.imageLayout = layout;
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = set;
		w.dstBinding = binding;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		w.pImageInfo = &info;
		vkUpdateDescriptorSets(s->device, 1, &w, 0, nullptr);
	};
	auto writeUniform = [&](VkDescriptorSet set, uint32_t binding, VulkanWarp::State::Buffer& b) {
		VkDescriptorBufferInfo info{};
		info.buffer = b.handle;
		info.range = VK_WHOLE_SIZE;
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = set;
		w.dstBinding = binding;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		w.pBufferInfo = &info;
		vkUpdateDescriptorSets(s->device, 1, &w, 0, nullptr);
	};
	auto writeStorage = [&](VkDescriptorSet set, uint32_t binding, VkImageView view) {
		VkDescriptorImageInfo info{};
		info.imageView = view;
		info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		VkWriteDescriptorSet w{};
		w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		w.dstSet = set;
		w.dstBinding = binding;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		w.pImageInfo = &info;
		vkUpdateDescriptorSets(s->device, 1, &w, 0, nullptr);
	};
	writeCombined(s->splatSet, 0, s->depthSampler, s->srcDepth.view);
	writeUniform(s->splatSet, 1, s->uboSplat);
	writeStorage(s->splatSet, 2, s->splat.view);
	writeCombined(s->pullSet, 0, s->colorSampler, s->srcColor.view);
	writeUniform(s->pullSet, 1, s->uboPull);
	writeStorage(s->pullSet, 2, s->splat.view);
	writeStorage(s->pullSet, 3, s->dst.view);
	writeStorage(s->fillSet, 0, s->dst.view);
	writeStorage(s->fillSet, 1, s->splat.view);
	writeCombined(s->fillSet, 2, s->colorSampler, s->dstMipView,
	              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	// Pipelines.
	std::vector<uint32_t> splatSpv, pullSpv, fillSpv;
	if (!LoadSpv(spvDir + "/warp_splat.spv", splatSpv, error) ||
	    !LoadSpv(spvDir + "/warp_pull.spv", pullSpv, error) ||
	    !LoadSpv(spvDir + "/warp_fill.spv", fillSpv, error))
		return false;
	std::cout << "[vulkan] SPIR-V: warp_splat.spv " << splatSpv.size() * 4 << " bytes, warp_pull.spv "
	          << pullSpv.size() * 4 << " bytes, warp_fill.spv " << fillSpv.size() * 4 << " bytes"
	          << std::endl;
	auto makePipe = [&](const std::vector<uint32_t>& spv, VkDescriptorSetLayout layout,
	                    VkPipelineLayout& pipeLayout, VkPipeline& pipe, const char* what,
	                    uint32_t pushSize = 0) {
		VkShaderModuleCreateInfo mi{};
		mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		mi.codeSize = spv.size() * sizeof(uint32_t);
		mi.pCode = spv.data();
		VkShaderModule mod = VK_NULL_HANDLE;
		if (!Check(vkCreateShaderModule(s->device, &mi, nullptr, &mod), what, error))
			return false;
		VkPipelineLayoutCreateInfo pli{};
		pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount = 1;
		pli.pSetLayouts = &layout;
		VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize };
		pli.pushConstantRangeCount = pushSize ? 1u : 0u;
		pli.pPushConstantRanges = pushSize ? &pushRange : nullptr;
		bool ok = Check(vkCreatePipelineLayout(s->device, &pli, nullptr, &pipeLayout), what, error);
		if (ok) {
			VkComputePipelineCreateInfo pci{};
			pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			pci.stage.module = mod;
			pci.stage.pName = "main";
			pci.layout = pipeLayout;
			ok = Check(vkCreateComputePipelines(s->device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe),
			           what, error);
		}
		vkDestroyShaderModule(s->device, mod, nullptr);
		return ok;
	};
	if (!makePipe(splatSpv, s->splatLayout, s->splatPipeLayout, s->splatPipe, "splat pipeline"))
		return false;
	if (!makePipe(pullSpv, s->pullLayout, s->pullPipeLayout, s->pullPipe, "pull pipeline"))
		return false;
	if (!makePipe(fillSpv, s->fillLayout, s->fillPipeLayout, s->fillPipe, "fill pipeline",
	              2 * sizeof(int32_t)))
		return false;

	// Timestamp queries: 0 submit-start, 1 after clear, 2 after splat,
	// 3 after pull, 4 after blits, 5 after fill, 6 after dst copy.
	// Optional; guarded everywhere.
	{
		uint32_t qCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(s->phys, &qCount, nullptr);
		std::vector<VkQueueFamilyProperties> qps(qCount);
		if (qCount > 0)
			vkGetPhysicalDeviceQueueFamilyProperties(s->phys, &qCount, qps.data());
		if (s->queueFamily < qps.size() && qps[s->queueFamily].timestampValidBits > 0) {
			VkPhysicalDeviceProperties devProps{};
			vkGetPhysicalDeviceProperties(s->phys, &devProps);
			s->timestampPeriodMs = devProps.limits.timestampPeriod / 1e6;
			VkQueryPoolCreateInfo qi{};
			qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
			qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
			qi.queryCount = 7;
			if (Check(vkCreateQueryPool(s->device, &qi, nullptr, &s->queryPool),
			          "timestamp query pool", nullptr)) {
				s->hasTimestamps = true;
			}
		}
		if (s->hasTimestamps)
			std::cout << "[vulkan] GPU timestamps available (period "
			          << s->timestampPeriodMs << " ms/tick)" << std::endl;
		else
			std::cout << "[vulkan] GPU timestamps unavailable (stage times off)" << std::endl;
	}

	width_ = width;
	height_ = height;
	initialized_ = true;
	return true;
}

void VulkanWarp::Shutdown()
{
	if (!state_)
		return;
	State* s = state_;
	if (s->device) {
		vkDeviceWaitIdle(s->device);
		if (s->queryPool)
			vkDestroyQueryPool(s->device, s->queryPool, nullptr);
		if (s->splatPipe)
			vkDestroyPipeline(s->device, s->splatPipe, nullptr);
		if (s->pullPipe)
			vkDestroyPipeline(s->device, s->pullPipe, nullptr);
		if (s->fillPipe)
			vkDestroyPipeline(s->device, s->fillPipe, nullptr);
		if (s->splatPipeLayout)
			vkDestroyPipelineLayout(s->device, s->splatPipeLayout, nullptr);
		if (s->pullPipeLayout)
			vkDestroyPipelineLayout(s->device, s->pullPipeLayout, nullptr);
		if (s->fillPipeLayout)
			vkDestroyPipelineLayout(s->device, s->fillPipeLayout, nullptr);
		if (s->descPool)
			vkDestroyDescriptorPool(s->device, s->descPool, nullptr);
		if (s->splatLayout)
			vkDestroyDescriptorSetLayout(s->device, s->splatLayout, nullptr);
		if (s->pullLayout)
			vkDestroyDescriptorSetLayout(s->device, s->pullLayout, nullptr);
		if (s->fillLayout)
			vkDestroyDescriptorSetLayout(s->device, s->fillLayout, nullptr);
		if (s->colorSampler)
			vkDestroySampler(s->device, s->colorSampler, nullptr);
		if (s->depthSampler)
			vkDestroySampler(s->device, s->depthSampler, nullptr);
		auto freeBuf = [&](State::Buffer& b) {
			if (b.mapped)
				vkUnmapMemory(s->device, b.memory);
			if (b.handle)
				vkDestroyBuffer(s->device, b.handle, nullptr);
			if (b.memory)
				vkFreeMemory(s->device, b.memory, nullptr);
		};
		freeBuf(s->uboSplat);
		freeBuf(s->uboPull);
		freeBuf(s->staging);
		auto freeImg = [&](State::Image& img) {
			if (img.view)
				vkDestroyImageView(s->device, img.view, nullptr);
			if (img.handle)
				vkDestroyImage(s->device, img.handle, nullptr);
			if (img.memory)
				vkFreeMemory(s->device, img.memory, nullptr);
		};
		freeImg(s->srcColor);
		freeImg(s->srcDepth);
		freeImg(s->splat);
		freeImg(s->dst);
		if (s->dstMipView)
			vkDestroyImageView(s->device, s->dstMipView, nullptr);
		if (s->fence)
			vkDestroyFence(s->device, s->fence, nullptr);
		if (s->pool)
			vkDestroyCommandPool(s->device, s->pool, nullptr);
		vkDestroyDevice(s->device, nullptr);
	}
	if (s->instance)
		vkDestroyInstance(s->instance, nullptr);
	delete s;
	state_ = nullptr;
	initialized_ = false;
}

bool VulkanWarp::Upload(const uint8_t* colorRGBA, const float* depth)
{
	if (!initialized_ || !colorRGBA || !depth)
		return Fail(nullptr, "Upload: not initialized or null plane");
	State* s = state_;
	VkDeviceSize colorBytes = static_cast<VkDeviceSize>(width_) * height_ * 4;
	VkDeviceSize depthBytes = static_cast<VkDeviceSize>(width_) * height_ * sizeof(float);
	VkDeviceSize planeBytes = colorBytes; // staging: color @+0, depth @+planeBytes
	if (!BeginOnce(s, &lastError_))
		return false;
	auto uploadPlane = [&](State::Image& img, const void* data, VkDeviceSize bytes,
	                         VkDeviceSize stagingOffset) {
		std::memcpy(static_cast<uint8_t*>(s->staging.mapped) + stagingOffset, data,
		             static_cast<size_t>(bytes));
		// Staging is HOST_COHERENT so no flush is needed.
		Transition(s, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
		           VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy copy{};
		copy.bufferOffset = stagingOffset;
		copy.imageExtent = {width_, height_, 1};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		vkCmdCopyBufferToImage(s->cmd, s->staging.handle, img.handle,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
		Transition(s, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
		           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	};
	uploadPlane(s->srcColor, colorRGBA, colorBytes, 0);
	uploadPlane(s->srcDepth, depth, depthBytes, planeBytes);
	if (!SubmitWait(s, &lastError_)) {
		if (lastError_.empty())
			lastError_ = "Upload: submit/wait failed";
		return false;
	}
	s->srcUploaded = true;
	return true;
}

bool VulkanWarp::Warp(const Pose& srcPose, const Mat4 srcProj, const Pose& dstPose, const Mat4 dstProj,
                      float holeRadius, std::vector<uint8_t>& outColor, bool countCoverage)
{
	if (!initialized_ || !state_->srcUploaded)
		return Fail(nullptr, "Warp: not initialized or nothing uploaded");
	State* s = state_;

	// Fold matrices on the CPU (same order as dynacon-vr depth_reprojector).
	Mat4 viewSrc, viewDst, renderInvP, renderInvV, dstInvP, dstInvV;
	{
		Mat4 camSrc, camDst;
		CameraFromPose(srcPose, camSrc);
		CameraFromPose(dstPose, camDst);
		if (!Inverse(camSrc, viewSrc) || !Inverse(camDst, viewDst))
			return Fail(nullptr, "Warp: singular camera matrix");
	}
	if (!Inverse(srcProj, renderInvP) || !Inverse(viewSrc, renderInvV) || !Inverse(dstProj, dstInvP) ||
	    !Inverse(viewDst, dstInvV))
		return Fail(nullptr, "Warp: singular projection/view matrix");
	SplatUBO splatUbo{};
	PullUBO pullUbo{};
	Multiply(renderInvV, renderInvP, splatUbo.renderInvVP);
	Multiply(dstProj, viewDst, splatUbo.warpVP);
	Multiply(dstInvV, dstInvP, pullUbo.dstInv);
	Multiply(srcProj, viewSrc, pullUbo.srcVP);
	pullUbo.tune[0] = fillWeightPow_;
	pullUbo.tune[1] = static_cast<float>(stretchRadius_);
	pullUbo.tune[2] = pullUbo.tune[3] = 0.0f;
	std::memcpy(s->uboSplat.mapped, &splatUbo, sizeof(splatUbo));
	std::memcpy(s->uboPull.mapped, &pullUbo, sizeof(pullUbo));

	const bool doFill = holeRadius > 0.0f;

	if (!BeginOnce(s, &lastError_))
		return false;
	if (s->hasTimestamps) {
		vkCmdResetQueryPool(s->cmd, s->queryPool, 0, 7);
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s->queryPool, 0);
	}

	// Splat scratch: clear to "empty" every warp, then splat.
	Transition(s, s->splat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
	           VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkClearColorValue clear{};
	clear.uint32[0] = kSplatEmpty;
	clear.uint32[1] = 0;
	clear.uint32[2] = 0;
	clear.uint32[3] = 0;
	VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vkCmdClearColorImage(s->cmd, s->splat.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
	                     &range);
	if (s->hasTimestamps)
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 1);
	Transition(s, s->splat, VK_IMAGE_LAYOUT_GENERAL,
	           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
	           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	// Dst target into GENERAL for storage writes.
	Transition(s, s->dst, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
	           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

	uint32_t gx = (width_ + 7) / 8;
	uint32_t gy = (height_ + 7) / 8;
	vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->splatPipe);
	vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->splatPipeLayout, 0, 1,
	                        &s->splatSet, 0, nullptr);
	vkCmdDispatch(s->cmd, gx, gy, 1);
	if (s->hasTimestamps)
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 2);

	// Splat writes (atomics) visible to pull reads: same queue.
	VkMemoryBarrier memBar{};
	memBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memBar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	memBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(s->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBar, 0, nullptr, 0, nullptr);

	vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pullPipe);
	vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pullPipeLayout, 0, 1,
	                        &s->pullSet, 0, nullptr);
	vkCmdDispatch(s->cmd, gx, gy, 1);
	if (s->hasTimestamps)
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 3);

	if (doFill) {
		// Mip-chain hole fill: blit the premultiplied pull output down
		// the chain, then patch black pixels with the local average.
		// A blit chain is fixed-function (no shader ALU), and sampling
		// the chain with divide-by-alpha yields the validity-weighted
		// mean: blurred fill, never stretched. Constant cost per hole.
		uint32_t mw = width_, mh = height_;
		Transition(s, s->dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		           VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1);
		for (uint32_t i = 1; i < s->mipLevels; i++) {
			Transition(s, s->dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, i, 1);
			VkImageBlit blit{};
			blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
			blit.srcOffsets[1] = {static_cast<int32_t>(mw), static_cast<int32_t>(mh), 1};
			mw = std::max(1u, mw / 2);
			mh = std::max(1u, mh / 2);
			blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
			blit.dstOffsets[1] = {static_cast<int32_t>(mw), static_cast<int32_t>(mh), 1};
			vkCmdBlitImage(s->cmd, s->dst.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			               s->dst.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
			               VK_FILTER_LINEAR);
			Transition(s, s->dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			           VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, i, 1);
		}
		if (s->hasTimestamps)
			vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 4);
	} else if (s->hasTimestamps) {
		// Keep query indices stable when the blit chain is skipped.
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 4);
	}

	// Fill always runs: valid pixels are un-premultiplied here, and
	// holes are patched when the mip chain was built above (or kept
	// black when hole-filling is off).
	// Mip levels 1..N become shader-readable for the fill tap.
	Transition(s, s->dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	           VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1,
	           s->mipLevels - 1);
	// Level 0 back to GENERAL for fill writes.
	Transition(s, s->dst, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
	           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1);
	vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->fillPipe);
	vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->fillPipeLayout, 0, 1,
	                        &s->fillSet, 0, nullptr);
	// View-relative LOD cap: view covers levels 1..N, so max lod is N-2.
	int32_t maxViewLod = doFill ? static_cast<int32_t>(s->mipLevels) - 2 : -1;
	int32_t fillMode = fillMode_;
	vkCmdPushConstants(s->cmd, s->fillPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                   sizeof(maxViewLod), &maxViewLod);
	vkCmdPushConstants(s->cmd, s->fillPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   sizeof(maxViewLod), sizeof(fillMode), &fillMode);
	vkCmdDispatch(s->cmd, gx, gy, 1);
	if (s->hasTimestamps)
		vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 5);

	// Read back dst (and the splat plane for coverage, same submit).
	Transition(s, s->dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
	           VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkDeviceSize colorBytes = static_cast<VkDeviceSize>(width_) * height_ * 4;
	VkBufferImageCopy copy{};
	copy.imageExtent = {width_, height_, 1};
	copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	vkCmdCopyImageToBuffer(s->cmd, s->dst.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       s->staging.handle, 1, &copy);
	if (countCoverage) {
		// Splat copy lands at staging offset colorBytes (staging is
		// 2 planes; Upload's contents were already consumed).
		Transition(s, s->splat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
		           VK_PIPELINE_STAGE_TRANSFER_BIT);
		VkBufferImageCopy copySplat{};
		copySplat.bufferOffset = colorBytes;
		copySplat.imageExtent = {width_, height_, 1};
		copySplat.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		vkCmdCopyImageToBuffer(s->cmd, s->splat.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		                       s->staging.handle, 1, &copySplat);
	}
	if (s->hasTimestamps)
	vkCmdWriteTimestamp(s->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s->queryPool, 6);
	if (!SubmitWait(s, &lastError_)) {
		if (lastError_.empty())
			lastError_ = "Warp: submit/wait failed";
		return false;
	}
	outColor.resize(static_cast<size_t>(colorBytes));
	std::memcpy(outColor.data(), s->staging.mapped, static_cast<size_t>(colorBytes));

	// GPU stage times (fence already waited, so results are ready).
	lastGpu_ = GpuTimes{};
	if (s->hasTimestamps) {
		uint64_t ts[7] = {};
		if (vkGetQueryPoolResults(s->device, s->queryPool, 0, 7, sizeof(ts), ts, sizeof(uint64_t),
		                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
		    VK_SUCCESS) {
			lastGpu_.valid = true;
			lastGpu_.clearMs = (ts[1] - ts[0]) * s->timestampPeriodMs;
			lastGpu_.splatMs = (ts[2] - ts[1]) * s->timestampPeriodMs;
			lastGpu_.pullMs = (ts[3] - ts[2]) * s->timestampPeriodMs;
			lastGpu_.fillMs = doFill ? (ts[5] - ts[3]) * s->timestampPeriodMs : 0.0;
			lastGpu_.copyMs = (ts[6] - (doFill ? ts[5] : ts[3])) * s->timestampPeriodMs;
		}
	}

	// Coverage: count non-empty splat texels from the same submit.
	if (countCoverage) {
		size_t filled = 0;
		size_t total = static_cast<size_t>(width_) * height_;
		const uint32_t* words =
		    static_cast<const uint32_t*>(static_cast<const void*>(static_cast<const uint8_t*>(
		        s->staging.mapped) + static_cast<size_t>(colorBytes)));
		for (size_t i = 0; i < total; i++) {
			if (words[i] != kSplatEmpty)
				filled++;
		}
		lastCoverage_ = total ? static_cast<float>(filled) / static_cast<float>(total) : 0.0f;
		lastSplat_.assign(words, words + total);
	}
	(void)kDepthScale;
	return true;
}

} // namespace openwarp_vk
