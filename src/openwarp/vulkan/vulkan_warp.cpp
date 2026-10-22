// Openwarp Vulkan compute warp implementation.
// Part of the Openwarp project, NCSA licensed (see LICENSE).

#include "vulkan_warp.hpp"

#include <vulkan/vulkan.h>

#include <Eigen/Dense>

#include <cmath>
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
	float hole[4];
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
	} srcColor, srcDepth, splat, dst;

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
	VkDescriptorPool descPool = VK_NULL_HANDLE;
	VkDescriptorSet splatSet = VK_NULL_HANDLE;
	VkDescriptorSet pullSet = VK_NULL_HANDLE;
	VkPipelineLayout splatPipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout pullPipeLayout = VK_NULL_HANDLE;
	VkPipeline splatPipe = VK_NULL_HANDLE;
	VkPipeline pullPipe = VK_NULL_HANDLE;

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
                 VulkanWarp::State::Image& img, std::string* error)
{
	VkImageCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ci.imageType = VK_IMAGE_TYPE_2D;
	ci.format = format;
	ci.extent = {w, h, 1};
	ci.mipLevels = 1;
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
	return true;
}

void Transition(VulkanWarp::State* s, VulkanWarp::State::Image& img, VkImageLayout next,
                VkAccessFlags dstAccess, VkPipelineStageFlags dstStage)
{
	if (img.layout == next)
		return; // No same-layout transitions are recorded by callers.
	VkImageMemoryBarrier bar{};
	bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	bar.oldLayout = img.layout;
	bar.newLayout = next;
	bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.image = img.handle;
	bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	bar.dstAccessMask = dstAccess;
	// Source masks derived from the tracked current layout.
	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	switch (img.layout) {
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
	return false;
}

bool VulkanWarp::Init(uint32_t width, uint32_t height, const std::string& spvDir, std::string* error)
{
	if (initialized_)
		Shutdown();
	if (width == 0 || height == 0)
		return Fail(error, "warp image size must be non-zero");

	state_ = new State();
	State* s = state_;

	// Instance (no extensions needed headless; try validation, ignore absence).
	VkApplicationInfo app{};
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "openwarp-vulkan-warp";
	app.apiVersion = VK_API_VERSION_1_0;
	const char* layer = "VK_LAYER_KHRONOS_validation";
	uint32_t layerCount = 0;
	{
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
	}
	VkInstanceCreateInfo ii{};
	ii.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ii.pApplicationInfo = &app;
	ii.enabledLayerCount = layerCount;
	ii.ppEnabledLayerNames = layerCount ? &layer : nullptr;
	if (!Check(vkCreateInstance(&ii, nullptr, &s->instance), "vkCreateInstance", error))
		return false;

	// Physical device with a compute-capable queue.
	uint32_t devCount = 0;
	if (!Check(vkEnumeratePhysicalDevices(s->instance, &devCount, nullptr),
	           "vkEnumeratePhysicalDevices", error) ||
	    devCount == 0)
		return Fail(error, "no Vulkan physical devices");
	std::vector<VkPhysicalDevice> devs(devCount);
	vkEnumeratePhysicalDevices(s->instance, &devCount, devs.data());
	bool picked = false;
	for (auto d : devs) {
		uint32_t qCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, nullptr);
		std::vector<VkQueueFamilyProperties> qs(qCount);
		vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, qs.data());
		for (uint32_t i = 0; i < qCount; i++) {
			if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				s->phys = d;
				s->queueFamily = i;
				picked = true;
				break;
			}
		}
		if (picked)
			break;
	}
	if (!picked)
		return Fail(error, "no compute-capable Vulkan queue family");

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
	if (!CreateImage(s, width, height, VK_FORMAT_R8G8B8A8_UNORM,
	                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, s->dst, error))
		return false;

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
	sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
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

	VkDescriptorPoolSize poolSizes[3]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[0].descriptorCount = 2;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[1].descriptorCount = 2;
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[2].descriptorCount = 3;
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = 2;
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

	auto writeCombined = [&](VkDescriptorSet set, uint32_t binding, VkSampler sampler,
	                         VkImageView view) {
		VkDescriptorImageInfo info{};
		info.sampler = sampler;
		info.imageView = view;
		info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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

	// Pipelines.
	std::vector<uint32_t> splatSpv, pullSpv;
	if (!LoadSpv(spvDir + "/warp_splat.spv", splatSpv, error) ||
	    !LoadSpv(spvDir + "/warp_pull.spv", pullSpv, error))
		return false;
	auto makePipe = [&](const std::vector<uint32_t>& spv, VkDescriptorSetLayout layout,
	                    VkPipelineLayout& pipeLayout, VkPipeline& pipe, const char* what) {
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
		if (s->splatPipe)
			vkDestroyPipeline(s->device, s->splatPipe, nullptr);
		if (s->pullPipe)
			vkDestroyPipeline(s->device, s->pullPipe, nullptr);
		if (s->splatPipeLayout)
			vkDestroyPipelineLayout(s->device, s->splatPipeLayout, nullptr);
		if (s->pullPipeLayout)
			vkDestroyPipelineLayout(s->device, s->pullPipeLayout, nullptr);
		if (s->descPool)
			vkDestroyDescriptorPool(s->device, s->descPool, nullptr);
		if (s->splatLayout)
			vkDestroyDescriptorSetLayout(s->device, s->splatLayout, nullptr);
		if (s->pullLayout)
			vkDestroyDescriptorSetLayout(s->device, s->pullLayout, nullptr);
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
		return false;
	State* s = state_;
	std::string error;
	VkDeviceSize colorBytes = static_cast<VkDeviceSize>(width_) * height_ * 4;
	VkDeviceSize depthBytes = static_cast<VkDeviceSize>(width_) * height_ * sizeof(float);
	VkDeviceSize planeBytes = colorBytes; // staging: color @+0, depth @+planeBytes
	if (!BeginOnce(s, nullptr))
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
	(void)error;
	if (!SubmitWait(s, nullptr))
		return false;
	s->srcUploaded = true;
	return true;
}

bool VulkanWarp::Warp(const Pose& srcPose, const Mat4 srcProj, const Pose& dstPose, const Mat4 dstProj,
                      float holeRadius, std::vector<uint8_t>& outColor)
{
	if (!initialized_ || !state_->srcUploaded)
		return false;
	State* s = state_;

	// Fold matrices on the CPU (same order as dynacon-vr depth_reprojector).
	Mat4 viewSrc, viewDst, renderInvP, renderInvV, dstInvP, dstInvV;
	{
		Mat4 camSrc, camDst;
		CameraFromPose(srcPose, camSrc);
		CameraFromPose(dstPose, camDst);
		if (!Inverse(camSrc, viewSrc) || !Inverse(camDst, viewDst))
			return false;
	}
	if (!Inverse(srcProj, renderInvP) || !Inverse(viewSrc, renderInvV) || !Inverse(dstProj, dstInvP) ||
	    !Inverse(viewDst, dstInvV))
		return false;
	SplatUBO splatUbo{};
	PullUBO pullUbo{};
	Multiply(renderInvV, renderInvP, splatUbo.renderInvVP);
	Multiply(dstProj, viewDst, splatUbo.warpVP);
	Multiply(dstInvV, dstInvP, pullUbo.dstInv);
	Multiply(srcProj, viewSrc, pullUbo.srcVP);
	pullUbo.hole[0] = holeRadius;
	pullUbo.hole[1] = pullUbo.hole[2] = pullUbo.hole[3] = 0.0f;
	std::memcpy(s->uboSplat.mapped, &splatUbo, sizeof(splatUbo));
	std::memcpy(s->uboPull.mapped, &pullUbo, sizeof(pullUbo));

	if (!BeginOnce(s, nullptr))
		return false;

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

	// Read back dst + coverage in one submit.
	Transition(s, s->dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
	           VK_PIPELINE_STAGE_TRANSFER_BIT);
	Transition(s, s->splat, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
	           VK_PIPELINE_STAGE_TRANSFER_BIT);
	VkDeviceSize colorBytes = static_cast<VkDeviceSize>(width_) * height_ * 4;
	VkBufferImageCopy copy{};
	copy.imageExtent = {width_, height_, 1};
	copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	vkCmdCopyImageToBuffer(s->cmd, s->dst.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       s->staging.handle, 1, &copy);
	if (!SubmitWait(s, nullptr))
		return false;
	outColor.resize(static_cast<size_t>(colorBytes));
	std::memcpy(outColor.data(), s->staging.mapped, static_cast<size_t>(colorBytes));

	// Coverage: count non-empty splat texels (second submit reuses staging).
	if (!BeginOnce(s, nullptr))
		return false;
	VkDeviceSize splatBytes = colorBytes;
	VkBufferImageCopy copySplat{};
	copySplat.imageExtent = {width_, height_, 1};
	copySplat.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	vkCmdCopyImageToBuffer(s->cmd, s->splat.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	                       s->staging.handle, 1, &copySplat);
	if (!SubmitWait(s, nullptr))
		return false;
	size_t filled = 0;
	size_t total = static_cast<size_t>(width_) * height_;
	const uint32_t* words = static_cast<const uint32_t*>(s->staging.mapped);
	for (size_t i = 0; i < total; i++) {
		if (words[i] != kSplatEmpty)
			filled++;
	}
	lastCoverage_ = total ? static_cast<float>(filled) / static_cast<float>(total) : 0.0f;
	(void)splatBytes;
	(void)kDepthScale;
	return true;
}

} // namespace openwarp_vk
