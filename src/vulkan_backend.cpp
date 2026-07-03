#include "gemmaedge/vulkan_backend.h"

#ifdef GEMMAEDGE_USE_VULKAN
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <cstring>

namespace gemmaedge {
namespace {

struct VulkanContext {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    uint32_t queue_family_index{0};
    bool initialized{false};
};

static VulkanContext g_ctx;

bool init_vulkan() {
    if (g_ctx.initialized) return true;

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "GemmaEdge";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "NoEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&create_info, nullptr, &g_ctx.instance) != VK_SUCCESS) {
        return false;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &device_count, nullptr);
    if (device_count == 0) return false;

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(g_ctx.instance, &device_count, devices.data());
    g_ctx.physical_device = devices[0]; // Simple selection: first device

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.physical_device, &queue_family_count, queue_families.data());

    bool found = false;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_ctx.queue_family_index = i;
            found = true;
            break;
        }
    }
    if (!found) return false;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = g_ctx.queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;

    if (vkCreateDevice(g_ctx.physical_device, &device_create_info, nullptr, &g_ctx.device) != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family_index, 0, &g_ctx.queue);
    g_ctx.initialized = true;
    return true;
}

} // namespace

bool is_vulkan_available() {
    static bool available = init_vulkan();
    return available;
}

bool vulkan_matvec(const void* matrix, std::size_t rows, std::size_t cols,
                   const float* vector, float* output) {
    if (!is_vulkan_available()) return false;
    // GPU math execution placeholder. Real production Vulkan driver would map descriptors, 
    // compile SPIR-V pipelines and submit command buffer work.
    (void)matrix; (void)rows; (void)cols; (void)vector; (void)output;
    return false;
}

} // namespace gemmaedge

#else

namespace gemmaedge {

bool is_vulkan_available() {
    return false;
}

bool vulkan_matvec(const void* matrix, std::size_t rows, std::size_t cols,
                   const float* vector, float* output) {
    (void)matrix; (void)rows; (void)cols; (void)vector; (void)output;
    return false;
}

} // namespace gemmaedge

#endif
