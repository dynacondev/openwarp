
#include "openwarp.hpp"
#include "OpenwarpApplication.hpp"
#include "Eigen/Dense"
#include <string>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
namespace fs = std::filesystem;
#include <iostream>
#include <fstream>
#include <glm/mat4x4.hpp>
#include "util/shader_util.hpp"
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/lib/stb_image_write.h"

using namespace Openwarp;

#define OBJ_DIR "../resources/"

OpenwarpApplication* OpenwarpApplication::instance;

const char* OpenwarpApplication::VulkanFillName(VulkanFillMethod m)
{
    switch (m) {
    case VulkanFillMethod::Off: return "Off (black)";
    case VulkanFillMethod::NnFill: return "NN fill (blur init)";
    default: return "<unknown>";
    }
}

OpenwarpApplication::VulkanFillMethod OpenwarpApplication::VulkanFillByIndex(int i)
{
    switch (i) {
    case 1: return VulkanFillMethod::NnFill;
    case 0:
    default: return VulkanFillMethod::Off;
    }
}

int OpenwarpApplication::VulkanFillIndex(VulkanFillMethod m)
{
    switch (m) {
    case VulkanFillMethod::Off: return 0;
    case VulkanFillMethod::NnFill: return 1;
    default: return 0;
    }
}

void OpenwarpApplication::CycleVulkanFill(int dir)
{
    int i = VulkanFillIndex(vulkanFill) + dir;
    i %= kVulkanFillCount;
    if (i < 0)
        i += kVulkanFillCount;
    vulkanFill = VulkanFillByIndex(i);
}

float OpenwarpApplication::VulkanFillHoleRadius() const
{
    // Maps the fill selector onto the VulkanWarp::Warp() holeRadius
    // knob. Only Off keeps 0; NnFill returns >0 to mark the hole-aware
    // path (the GPU pull math ignores it; the NN pass reads the splat
    // plane for its mask). HoleRadius stays plumbed so new methods slot
    // in without touching the Warp() call.
    switch (vulkanFill) {
    case VulkanFillMethod::Off:
    default: return 0.0f;
    case VulkanFillMethod::NnFill: return 1.0f;
    }
}

int OpenwarpApplication::VulkanFillMode() const
{
    // Maps onto the warp_fill.comp fillMode push constant (switch point
    // for future fills; only 0 = black is implemented).
    switch (vulkanFill) {
    case VulkanFillMethod::Off:
    default: return 0;
    case VulkanFillMethod::NnFill: return 1;
    }
}

OpenwarpApplication::OpenwarpApplication(size_t meshSize, Backend backend_) : backend(backend_){

    // For static callbacks.
    OpenwarpApplication::instance = this;

    if (backend == Backend::Vulkan) {
        std::cout << "Initializing Openwarp [Vulkan compute-warp backend requested]" << std::endl;
#ifndef OPENWARP_HAS_VULKAN
        std::cerr << "WARNING: --vulkan requested but this binary was built without Vulkan support; "
                     "falling back to the OpenGL mesh path until the Vulkan backend lands."
                  << std::endl;
#endif
        // Raymarch has no Vulkan port (mesh warp only). Vsync on: the
        // uncapped swap loop tears badly with variable warp latency, and
        // the ImGui toggle can still turn it off.
        useRay = false;
        useVsync = true;
    }

    std::cout << "Initializing Openwarp";
    std::cout << "Initializing GLFW...." << std::endl;
    if(!glfwInit()) {
        abort();
    }
    std::cout << "Initializing application window...." << std::endl;
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    const char* glsl_version = "#version 130";

    window = glfwCreateWindow(WIDTH, HEIGHT, "Openwarp Demo", nullptr, nullptr);
    if(!window) {
        std::cerr << "Failed to create window." << std::endl;
        glfwTerminate();
        abort();
    }
    glfwMakeContextCurrent(window);
    std::cout << "Application window successfully created...." << std::endl;

    glfwSetMouseButtonCallback(window, mouseClickCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

    if (glfwRawMouseMotionSupported())
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        glfwTerminate();
        throw std::runtime_error(std::string("Could initialize GLEW, error = ") +
                                (const char*)glewGetErrorString(err));
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    imgui_io = ImGui::GetIO();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    meshWidth = meshHeight = meshSize;
    // Adjust meshwarp bleed radius according to mesh size.
    bleedRadius = (1.0f/(meshWidth));

    initGL();
}



void OpenwarpApplication::DoFullTestRun(const TestRun& testRun) {

    // Create desired output dir if it doesn't exist
    if(!fs::exists(testRun.outputDir)){
        fs::create_directory(testRun.outputDir);
    }

    // Get timestamp of run.
    std::time_t now_tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* stamp = gmtime(&now_tt);
    char timestampBuffer[128];
    std::strftime(timestampBuffer, 32, "%d.%m.%Y_%H.%M.%S", stamp); 

    // Create the test run dir with timestamp.
    std::string runDir = testRun.outputDir + "/" + std::string(timestampBuffer);
    std::cout << "testRun.outputDir: " << testRun.outputDir << ", runDir: " << runDir << std::endl;
    fs::create_directory(runDir);

    std::string origin_tag = std::to_string(testRun.startPose.position[0]) + "_"
                            + std::to_string(testRun.startPose.position[1]) + "_"
                            + std::to_string(testRun.startPose.position[2]);

    // Write the test origin to a file.
    std::ofstream info_file;
    info_file.open(runDir + "/run_info.txt");
    info_file << origin_tag;
    info_file.close();

    // Create the ground truth and warp directories
    fs::create_directory(runDir + "/warped");
    fs::create_directory(runDir + "/ground_truth");

    // Render and write reprojected frames
    RunTest(testRun, runDir + "/warped", false, false);

    // Render and write ground truth frames
    RunTest(testRun, runDir + "/ground_truth", true, false);
}

void OpenwarpApplication::RunTest(const TestRun& testRun, std::string runDir, bool isGroundTruth, bool testUsesRay){

    // For GL_RGB8
    GLubyte* fb_data = (GLubyte*)malloc(WIDTH * HEIGHT * 3);

    // Need to flip vertically.
    stbi_flip_vertically_on_write(1);

    // shouldReproject = true makes renderScene draw to renderFBO.
    // shouldReproject = false makes renderScene draw to screen.
    
    if(isGroundTruth) {
        shouldReproject = false;
    } else {
        // If not the ground truth run, we render once
        // so that the reprojection has something to use.
        position = testRun.startPose.position;
        orientation = testRun.startPose.orientation;
        shouldReproject = true;
        renderScene();
    }

    for (auto &test : testRun) {
        position = test.position;
        orientation = test.orientation;

        glfwPollEvents();
        if(glfwWindowShouldClose(window)){
            break;
        }

        // Render
        if (isGroundTruth)
            renderScene();
        else
            doReprojection(testUsesRay);

        // Read pixels out from the screen.
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, fb_data);

        std::string filename = runDir + "/"
                                + std::to_string(test.relative_pos[0]) + "_"
                                + std::to_string(test.relative_pos[1]) + "_"
                                + std::to_string(test.relative_pos[2]) + ".png";
        std::cout << "Writing to " << filename << std::endl;
        stbi_write_png(filename.c_str(), WIDTH, HEIGHT, 3, fb_data, WIDTH * sizeof(GLubyte) * 3);

        glfwSwapBuffers(window);
        double instFps = 1.0 / (glfwGetTime() - lastSwapTime);
        presentationFramerate =
            (presentationFramerate <= 0.0) ? instFps : (0.95 * presentationFramerate + 0.05 * instFps);
        lastSwapTime = glfwGetTime();
    }
}

void OpenwarpApplication::Run(bool showGUI){
#ifdef OPENWARP_HAS_VULKAN
    guiEnabled = showGUI;
#endif
    if (backend == Backend::Vulkan && !vulkanModeLogged) {
        vulkanModeLogged = true;
        std::cout << "[vulkan] backend active: GL scene render, mesh warp on Vulkan compute "
                     "(raymarch disabled in this mode)." << std::endl;
        // Raymarch is GL-only; never raymarch under --vulkan.
        useRay = false;
    }
    while(!glfwWindowShouldClose(window)) {

        glfwPollEvents();
        imgui_io = ImGui::GetIO();
        
        processInput();

        // If it's time to render a frame (based on the desired render frequency)
        // we render, and schedule the next one from completion: with warps
        // slower than the render interval, absolute scheduling would lag
        // forever and render every iteration catching up (judder).
        // painted tracks whether the back buffer holds anything new this
        // frame. Swapping with no new paint alternates the two stale
        // buffers on screen (ghosting/double image at uncapped rates),
        // so idle frames hold the last presented frame instead.
        bool painted = false;
        if(glfwGetTime() >= nextRenderTime){

            if(shouldRenderScene) {
                renderScene();
                // Direct-to-screen renders paint the screen; renderFBO
                // renders do not (only the warp presents those).
                if (!shouldReproject)
                    painted = true;
            }

            nextRenderTime = glfwGetTime() + renderInterval;

        }

        // Reproject every frame.
        // If we've already rendered this frame, this
        // will not reproject. This is because we don't resample
        // user input before reprojection; in an XR application,
        // you would resample user pose every time before reprojection
        // for the most up-to-date pose.
        if (shouldReproject)
            painted = doReprojection(useRay) || painted;

        if (!painted && showGUI) {
            // Fully idle (rendering and reprojection both off, or
            // nothing yet painted): recompose the held frame so the GUI
            // — and its state readouts — stays alive instead of freezing
            // with no way to see or recover state. Swapping after a
            // redraw is safe (no stale-buffer alternation); swapping
            // with no redraw is what ghosts.
            bool recomposed = false;
#ifdef OPENWARP_HAS_VULKAN
            if (backend == Backend::Vulkan)
                recomposed = presentCachedVulkanFrame();
            else
#endif
            {
                // GL direct path keeps no copy: blit front->back so the
                // back buffer holds the last frame again, then the GUI
                // draws over it below.
                int fbW = WIDTH, fbH = HEIGHT;
                glfwGetFramebufferSize(window, &fbW, &fbH);
                glReadBuffer(GL_FRONT);
                glDrawBuffer(GL_BACK);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glBlitFramebuffer(0, 0, fbW, fbH, 0, 0, fbW, fbH,
                                  GL_COLOR_BUFFER_BIT, GL_NEAREST);
                glReadBuffer(GL_BACK);
                glDrawBuffer(GL_BACK);
                recomposed = true;
            }
            painted = recomposed;
        }

        if (painted) {
            if(showGUI)
                drawGUI();

            glfwSwapBuffers(window);
            // EMA so the overlay doesn't swing wildly with variable warp time.
            double instFps = 1.0 / (glfwGetTime() - lastSwapTime);
            presentationFramerate =
                (presentationFramerate <= 0.0) ? instFps : (0.95 * presentationFramerate + 0.05 * instFps);
            lastSwapTime = glfwGetTime();
        }
    }
}

void OpenwarpApplication::drawGUI(){
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const bool isVulkan = (backend == Backend::Vulkan);

    // GL-only windows: both are disconnected in Vulkan mode (GL mesh
    // bleed uniforms and raymarch uniforms are never uploaded there),
    // so hide them instead of showing dead sliders.
    if(showMeshConfig && !isVulkan) {
        ImGui::SetNextWindowPos(ImVec2(0, 1024), ImGuiCond_Once, ImVec2(0.0f, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(300,250), ImGuiCond_Always);
        
        ImGui::Begin("Mesh configuration", &showMeshConfig, ImGuiWindowFlags_NoResize);
        if (ImGui::CollapsingHeader("Edge bleed options", ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::Text("Edge bleed radius");
            ImGui::PushItemWidth(-1);
            ImGui::SliderFloat("##1", &bleedRadius, 0.0f, 0.05f);
            ImGui::Text("Edge bleed tolerance");
            ImGui::SliderFloat("##2", &bleedTolerance, 0.0f, 0.05f);
            ImGui::PopItemWidth();
        }
        ImGui::Checkbox("Show grid debug overlay", &showDebugGrid);
        ImGui::End();
    }

    // Vulkan-only window: lists the fill method radios (only Off while
    // fills are stripped) so the switch infra stays live for the next
    // fill. To add a fill: append a VulkanFillMethod value + its tuning
    // widgets here.
    if (showVulkanFillConfig && isVulkan) {
        ImGui::SetNextWindowPos(ImVec2(0, 1024), ImGuiCond_Once, ImVec2(0.0f, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_Always);

        ImGui::Begin("Vulkan fill", &showVulkanFillConfig, ImGuiWindowFlags_NoResize);
        ImGui::Text("Fill method (T cycles)");
        for (int i = 0; i < kVulkanFillCount; i++) {
            auto m = VulkanFillByIndex(i);
            bool active = (vulkanFill == m);
            if (ImGui::RadioButton(VulkanFillName(m), active))
                vulkanFill = m;
        }
        if (vulkanFill == VulkanFillMethod::NnFill) {
#ifdef OPENWARP_HAS_VULKAN
            ImGui::TextWrapped("weights: %s",
                nnFillModelPath.empty() ? "<pending first warp>" : nnFillModelPath.c_str());
#endif
        }
        ImGui::TextWrapped("Fill has no tunable parameters.");

        ImGui::End();
    }

    if(showRayConfig && !isVulkan) {
        ImGui::SetNextWindowPos(ImVec2(300, 1024), ImGuiCond_Once, ImVec2(0.0f, 1.0f));
        ImGui::SetNextWindowSize(ImVec2(300,250), ImGuiCond_Always);
        
        ImGui::Begin("Raymarch configuration", &showRayConfig, ImGuiWindowFlags_NoResize);
        ImGui::Text("Ray exponent power");
        ImGui::PushItemWidth(-1);
        ImGui::SliderFloat("##1", &rayPower, 0.0f, 1.0f);
        ImGui::Text("Ray step size");
        ImGui::SliderFloat("##2", &rayStepSize, 0.0f, 5.0f);
        ImGui::Text("Ray depth offset");
        ImGui::SliderFloat("##3", &rayDepthOffset, 0.0f, 2.0f);
        ImGui::Text("Occlusion detection threshold");
        ImGui::SliderFloat("##4", &occlusionThreshold, 0.0f, 0.03f);
        ImGui::Text("Occlusion offset");
        ImGui::SliderFloat("##5", &occlusionOffset, 0.0f, 1.0f);
        ImGui::PopItemWidth();
    
        ImGui::End();
    }

    // Stats overlay
    ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoDecoration |
                                    ImGuiWindowFlags_AlwaysAutoResize |
                                    ImGuiWindowFlags_NoSavedSettings |
                                    ImGuiWindowFlags_NoFocusOnAppearing |
                                    ImGuiWindowFlags_NoNav |
                                    ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(imgui_io.DisplaySize.x - 10.0f, 20.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(350,300), ImGuiCond_Always);
    ImGui::Begin("Stats", NULL, overlay_flags);
    ImGui::Text("Current stats:");
    ImGui::Separator();
    ImGui::Text("Render framerate: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%.2f hz", (float)(1.0f/renderInterval));
    ImGui::Text("Is rendering? ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), shouldRenderScene ? "Yes" : "No");
    ImGui::Text("Presentation framerate: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%.2f hz", (float)presentationFramerate);
    if (isVulkan) {
        ImGui::Text("Current fill: ");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", VulkanFillName(vulkanFill));
    } else {
        ImGui::Text("Current reprojection algo: ");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), useRay ? "Raymarch-based" : "Mesh-based");
    }
    ImGui::Text("Is reprojecting? ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), shouldReproject ? "Yes" : "No");
    ImGui::Text("Mesh size: ");
    ImGui::SameLine();
    if(isVulkan || useRay) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "N/A");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%d x %d", meshWidth, meshHeight);
    }
    
    if (useVsync == true)
    {
        ImGui::Button(" Vsync ON ");
        if (ImGui::IsItemClicked(0))
        {
            useVsync = !useVsync;
            glfwSwapInterval(useVsync ? 1 : 0);
        }
    } else {
        if (ImGui::Button(" Vsync OFF ")){
            useVsync = true;
            glfwSwapInterval(useVsync ? 1 : 0);
        }
    }
    
    
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    
    ImGui::Text("Controls:");
    ImGui::Separator();
    ImGui::Text("Look: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Mouse");
    ImGui::Text("Move: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WASD, QE");
    ImGui::Text("Change render FPS: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Scroll wheel");
    ImGui::Text("Toggle rendering: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "F");
    if (isVulkan) {
        ImGui::Text("Switch fill method: ");
    } else {
        ImGui::Text("Switch reprojection type: ");
    }
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "T");
    ImGui::Text("Toggle reprojection: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "R");
    ImGui::Text("Release mouse focus: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Esc");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void OpenwarpApplication::processInput(){
    
    bool isFocused = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
    if(isFocused) {

        // Query key input for translation input.
        Eigen::Vector3f translation = {
            (float)(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) - (float)(glfwGetKey(window, GLFW_KEY_A)),
            (float)(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) - (float)(glfwGetKey(window, GLFW_KEY_Q)),
            (float)(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) - (float)(glfwGetKey(window, GLFW_KEY_W))
        };

        // Query cursor position for rotation
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        orientation = Eigen::AngleAxisf(((xpos - xpos_onfocus) + xpos_unfocus) / WIDTH, -Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(((ypos - ypos_onfocus) + ypos_unfocus) / HEIGHT, -Eigen::Vector3f::UnitX());

        // Alter position based on user orientation and input.
        position += (orientation * translation) * (glfwGetTime() - lastInputTime);

        lastInputTime = glfwGetTime();
    }
}

#ifdef OPENWARP_HAS_VULKAN
// Resolves + loads the blur-init weights for mode 1. Search order: env
// OPENWARP_NN_MODEL, build-time default, then common relative layouts.
// Failure (or infer failure later) keeps the black-hole frame: the NN
// pass is strictly optional post-processing, never the warp itself.
bool OpenwarpApplication::EnsureNnFill()
{
    if(nnFill && nnFill->Available())
        return true;
    std::vector<std::string> cands;
    if(const char* env = std::getenv("OPENWARP_NN_MODEL"))
        cands.emplace_back(env);
#ifdef OPENWARP_NN_MODEL_DEFAULT
    cands.emplace_back(OPENWARP_NN_MODEL_DEFAULT);
#endif
    cands.emplace_back("../ml/weights/fill_blur_init.onnx");
    cands.emplace_back("./ml/weights/fill_blur_init.onnx");
    cands.emplace_back("ml/weights/fill_blur_init.onnx");
    std::string picked;
    for(auto& c : cands) {
        std::ifstream probe(c, std::ios::binary);
        if(probe) {
            picked = c;
            break;
        }
    }
    if(picked.empty()) {
        std::cerr << "[nn-fill] no weights found (set OPENWARP_NN_MODEL); staying black."
                  << std::endl;
        return false;
    }
    if(!nnFill)
        nnFill = std::make_unique<openwarp_vk::VulkanNnFill>();
    std::string error;
    if(!nnFill->Load(picked, &error)) {
        std::cerr << "[nn-fill] load failed (" << error << "); staying black." << std::endl;
        return false;
    }
    nnFillModelPath = picked;
    return true;
}

bool OpenwarpApplication::EnsureVulkanWarp()
{
    if(vulkanWarp && vulkanWarp->IsInitialized())
        return true;
    if(vulkanFailed)
        return false;

    // Locate the warp SPIR-V: env override, build-time default, then
    // common relative layouts.
    std::vector<std::string> dirs;
    if(const char* env = std::getenv("OPENWARP_SPV_DIR"))
        dirs.emplace_back(env);
#ifdef OPENWARP_SPV_DIR_DEFAULT
    dirs.emplace_back(OPENWARP_SPV_DIR_DEFAULT);
#endif
    dirs.emplace_back("../openwarp-shaders");
    dirs.emplace_back("./openwarp-shaders");
    std::string spvDir;
    for(auto& d : dirs) {
        std::ifstream probe(d + "/warp_splat.spv", std::ios::binary);
        if(probe) {
            spvDir = d;
            break;
        }
    }
    if(spvDir.empty()) {
        std::cerr << "[vulkan] no warp SPIR-V found (set OPENWARP_SPV_DIR); disabling." << std::endl;
        vulkanFailed = true;
        return false;
    }

    vulkanWarp = std::make_unique<openwarp_vk::VulkanWarp>();
    std::string error;
    const char* glRendererForVk =
        reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    if(!vulkanWarp->Init(WIDTH, HEIGHT, spvDir, &error, glRendererForVk)) {
        std::cerr << "[vulkan] VulkanWarp::Init failed: " << error << "; disabling." << std::endl;
        vulkanWarp.reset();
        vulkanFailed = true;
        return false;
    }
    std::cout << "[vulkan] compute warp ready (" << WIDTH << "x" << HEIGHT << ", spv: " << spvDir
              << ")" << std::endl;
    // Cross-check which GPU GL runs on vs Vulkan: on Optimus/PRIME they
    // can differ (GL NVIDIA + Vk iGPU), which doubles PCIe traffic. Both
    // names are logged; warn when they obviously disagree.
    const char* glRenderer =
        reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const std::string& vkDevice = vulkanWarp->DeviceName();
    std::cout << "[vulkan] GL renderer: " << (glRenderer ? glRenderer : "<unknown>")
              << " | Vk device: " << (vkDevice.empty() ? "<unknown>" : vkDevice) << std::endl;
    auto mentions = [](const char* hay, const char* needle) {
        return hay && std::strstr(hay, needle) != nullptr;
    };
    auto mentionsS = [](const std::string& hay, const char* needle) {
        return hay.find(needle) != std::string::npos;
    };
    bool glNvidia = mentions(glRenderer, "NVIDIA");
    bool vkNvidia = mentionsS(vkDevice, "NVIDIA");
    if(glRenderer && !vkDevice.empty() && (glNvidia != vkNvidia)) {
        std::cerr << "[vulkan] WARNING: GL and Vulkan look like different GPUs "
                     "(GL " << (glNvidia ? "" : "not ") << "NVIDIA, Vk " << (vkNvidia ? "" : "not ")
                  << "NVIDIA). Cross-GPU readbacks will hurt. "
                     "Override with OPENWARP_VK_DEVICE=<index>."
                  << std::endl;
    }
    return true;
}

// Redraws the last warped frame from the cached present texture.
// See the hpp for the ghosting rationale. False when nothing has been
// warped yet (no texture/buffers exist).
bool OpenwarpApplication::presentCachedVulkanFrame()
{
    if(vulkanPresentTex == 0 || vulkanPresentProgram == 0)
        return false;
    // Draw at the real framebuffer size (HiDPI/compositor scaling maps
    // 1:1 only if the viewport matches; a hardcoded 1024 viewport on a
    // 2x surface would upscale blurry / fill a quarter).
    int fbW = WIDTH;
    int fbH = HEIGHT;
    glfwGetFramebufferSize(window, &fbW, &fbH);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fbW, fbH);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glUseProgram(vulkanPresentProgram);
    if(vulkanPresentTexLoc == -2)
        vulkanPresentTexLoc = glGetUniformLocation(vulkanPresentProgram, "present_tex");
    if(vulkanPresentTexLoc >= 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, vulkanPresentTex);
        glUniform1i(vulkanPresentTexLoc, 0);
    }
    glBindVertexArray(vulkanPresentVAO);
    // (VAO holds the fullscreen-quad attrib setup from creation; no
    // per-warp vertex setup needed.)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    return true;
}

// Returns 1 when a frame was presented, 0 when the frame was
// skipped as identical (nothing painted: hold the last frame), -1
// when Vulkan is unavailable/failed (caller falls back to GL).
int OpenwarpApplication::doVulkanMeshWarp()
{
    if(!EnsureVulkanWarp())
        return -1;

    // Dirty tracking: the source frame only changes when renderScene runs
    // (15 Hz). Re-upload then; re-warp when the source, the target pose,
    // or the fill setting moved since the last warp. The fill setting
    // MUST participate: otherwise switching methods shows the stale
    // image. Otherwise the previously presented frame is still correct
    // — skip everything (this is what keeps an uncapped present loop
    // from re-warping identical frames at 1000 Hz).
    bool poseMoved = !haveLastWarpPose || lastWarpPos != position || lastWarpQuat != orientation;
    bool freshSource = (renderSeq != lastWarpRenderSeq);
    bool fillChanged = !lastWarpFillInit || vulkanFill != lastWarpFill;
    if(!poseMoved && !freshSource && !fillChanged) {
        vulkanStats.skips++;
        // Idle with nothing new: re-present the cached frame when the
        // GUI is up so the fill window stays interactive (same texture
        // redrawn into the back buffer — no stale-buffer ghosting, and
        // no 4 MiB re-upload). Headless holds the last frame instead.
        if(guiEnabled && presentCachedVulkanFrame())
            return 1;
        return 0;
    }

    double t0 = nowMs();

    // (Re)size hoisted host buffers once, never in the hot path.
    size_t px = static_cast<size_t>(WIDTH) * HEIGHT;
    if(vulkanColor.size() != px * 4)
        vulkanColor.resize(px * 4);
    if(vulkanDepth.size() != px)
        vulkanDepth.resize(px);
    if(vulkanOut.size() != px * 4)
        vulkanOut.resize(px * 4);

    double tReadback = t0;
    if(freshSource) {
        // Read back the GL-rendered scene. renderScene leaves renderFBO
        // bound; row 0 here is the frustum bottom for both planes, which is
        // exactly the row order the warp math assumes (clip y = guv*2-1).
        glBindFramebuffer(GL_READ_FRAMEBUFFER, renderFBO);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, vulkanColor.data());
        glReadPixels(0, 0, WIDTH, HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, vulkanDepth.data());
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        tReadback = nowMs();

        if(!vulkanWarp->Upload(vulkanColor.data(), vulkanDepth.data())) {
            std::cerr << "[vulkan] Upload failed (" << vulkanWarp->LastError() << "); disabling."
                      << std::endl;
            vulkanFailed = true;
            return -1;
        }
        vulkanStats.uploads++;
        lastWarpRenderSeq = renderSeq;
        if(!vulkanDepthLogged) {
            vulkanDepthLogged = true;
            float mn = vulkanDepth[0];
            float mx = vulkanDepth[0];
            for(float d : vulkanDepth) {
                mn = std::min(mn, d);
                mx = std::max(mx, d);
            }
            std::cout << "[vulkan] uploaded depth range [" << mn << ", " << mx
                      << "] (sky/clear pixels should sit at 1.0)" << std::endl;
        }
    }
    double tUpload = nowMs();

    // Source = rendered pose, target = current pose; same projection.
    // renderedCameraMatrix is camera-to-world (Translation * Rotation),
    // matching VulkanWarp::Pose (view = its inverse).
    openwarp_vk::Pose srcPose{};
    {
        Eigen::Vector3f t = renderedCameraMatrix.block<3, 1>(0, 3);
        Eigen::Quaternionf q(renderedCameraMatrix.block<3, 3>(0, 0));
        srcPose.pos[0] = t.x();
        srcPose.pos[1] = t.y();
        srcPose.pos[2] = t.z();
        srcPose.quat[0] = q.x();
        srcPose.quat[1] = q.y();
        srcPose.quat[2] = q.z();
        srcPose.quat[3] = q.w();
    }
    openwarp_vk::Pose dstPose{};
    dstPose.pos[0] = position.x();
    dstPose.pos[1] = position.y();
    dstPose.pos[2] = position.z();
    dstPose.quat[0] = orientation.x();
    dstPose.quat[1] = orientation.y();
    dstPose.quat[2] = orientation.z();
    dstPose.quat[3] = orientation.w();
    openwarp_vk::Mat4 srcProj{};
    openwarp_vk::Mat4 dstProj{};
    std::memcpy(srcProj, projection.data(), sizeof(srcProj));
    std::memcpy(dstProj, projection.data(), sizeof(dstProj));

    // Hole fill dispatch: Off maps to holeRadius 0 (honest black for
    // A/B). Mode 1 (NnFill) keeps the same GPU warp (fill.comp falls
    // back to black-hole normalize for unknown modes) and fills on the
    // CPU afterwards, so the mode-0 path below is bit-identical.
    // Coverage is sampled (1/60 warps): same-submit copy, negligible
    // cost — plus every warp under NnFill, which needs the splat plane
    // as its hole mask.
    bool wantCoverage = (vulkanStats.warps % 60) == 0 || vulkanFill == VulkanFillMethod::NnFill;
    vulkanWarp->SetFillMode(VulkanFillMode());
    if(!vulkanWarp->Warp(srcPose, srcProj, dstPose, dstProj, VulkanFillHoleRadius(), vulkanOut,
                         wantCoverage)) {
        std::cerr << "[vulkan] Warp failed (" << vulkanWarp->LastError() << "); disabling."
                  << std::endl;
        vulkanFailed = true;
        return -1;
    }
    if(vulkanFill == VulkanFillMethod::NnFill) {
        const uint32_t* splatPtr = nullptr;
        const std::vector<uint32_t>& splat = vulkanWarp->LastSplatPlane();
        if(wantCoverage && splat.size() == px)
            splatPtr = splat.data();
        // Any failure keeps vulkanOut as the black-hole frame (logged
        // inside VulkanNnFill); it can never paint garbage.
        if(EnsureNnFill())
            nnFill->Fill(WIDTH, HEIGHT, vulkanOut, splatPtr);
    }
    vulkanStats.warps++;
    haveLastWarpPose = true;
    lastWarpPos = position;
    lastWarpQuat = orientation;
    lastWarpFill = vulkanFill;
    lastWarpFillInit = true;
    if(wantCoverage)
        vulkanStats.lastCoverage = vulkanWarp->LastCoverage();
    {
        openwarp_vk::VulkanWarp::GpuTimes gpu = vulkanWarp->LastGpuTimes();
        if(gpu.valid) {
            vulkanStats.gpuSplatMs += gpu.splatMs;
            vulkanStats.gpuPullMs += gpu.pullMs;
            vulkanStats.gpuFillMs += gpu.fillMs;
            vulkanStats.gpuCopyMs += gpu.clearMs + gpu.copyMs;
            vulkanStats.gpuTimedWarps++;
        }
    }
    // Debug dumps (validity-mask discriminator for spurious black):
    // OPENWARP_VK_DUMP=1 writes the warped frame + splat-validity mask as
    // raw PPM/PGM on coverage-sampled warps. No image dependency.
    if(wantCoverage && std::getenv("OPENWARP_VK_DUMP")) {
        char base[64];
        std::snprintf(base, sizeof(base), "vulkan-dump-%04llu",
                      (unsigned long long)vulkanDumpCount++);
        {
            std::ofstream f(std::string(base) + ".ppm", std::ios::binary);
            f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
            std::vector<char> row(WIDTH * 3);
            for(uint32_t y = 0; y < HEIGHT; y++) {
                for(uint32_t x = 0; x < WIDTH; x++) {
                    size_t i = static_cast<size_t>(y) * WIDTH + x;
                    row[3 * x + 0] = static_cast<char>(vulkanOut[4 * i + 0]);
                    row[3 * x + 1] = static_cast<char>(vulkanOut[4 * i + 1]);
                    row[3 * x + 2] = static_cast<char>(vulkanOut[4 * i + 2]);
                }
                f.write(row.data(), row.size());
            }
        }
        const std::vector<uint32_t>& splat = vulkanWarp->LastSplatPlane();
        if(splat.size() == px) {
            std::ofstream f(std::string(base) + "-mask.pgm", std::ios::binary);
            f << "P5\n" << WIDTH << " " << HEIGHT << "\n255\n";
            std::vector<char> row(WIDTH);
            for(uint32_t y = 0; y < HEIGHT; y++) {
                for(uint32_t x = 0; x < WIDTH; x++) {
                    size_t i = static_cast<size_t>(y) * WIDTH + x;
                    row[x] = (splat[i] != 0xffffffffu) ? char(255) : char(0);
                }
                f.write(row.data(), row.size());
            }
        }
        std::cout << "[vulkan] wrote " << base << ".ppm + mask" << std::endl;
    }
    double tWarp = nowMs();

    // Present the warped result with a trivial GL textured quad.
    if(vulkanPresentTex == 0) {
        glGenTextures(1, &vulkanPresentTex);
        glBindTexture(GL_TEXTURE_2D, vulkanPresentTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        vulkanPresentProgram =
            init_and_link("../resources/shaders/present.vert", "../resources/shaders/present.frag");
        // Fullscreen triangle strip: xy + uv. Attrib setup lives in the
        // VAO once; the hot path only binds the VAO.
        const float quad[16] = {-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1};
        glGenVertexArrays(1, &vulkanPresentVAO);
        glBindVertexArray(vulkanPresentVAO);
        glGenBuffers(1, &vulkanPresentVBO);
        glBindBuffer(GL_ARRAY_BUFFER, vulkanPresentVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glBindVertexArray(0);
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, vulkanPresentTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE,
                     vulkanOut.data());

    // Draw the freshly uploaded frame (shared with the idle skip path:
    // same texture redrawn, so no stale-buffer ghosting).
    presentCachedVulkanFrame();
    double tPresent = nowMs();

    vulkanStats.readbackMs += (tReadback - t0);
    vulkanStats.uploadMs += (tUpload - tReadback);
    vulkanStats.warpMs += (tWarp - tUpload);
    vulkanStats.presentMs += (tPresent - tWarp);

    // Paced profiling: proves the Vulkan pipeline renders + shows where
    // time goes (rendering vs shading vs copies). Per-interval means
    // (reset each print) so trends/slopes are readable; the cumulative
    // alternative hides them. Every ~2 s.
    if(tPresent - vulkanStatsLastLog > 2000.0) {
        vulkanStatsLastLog = tPresent;
        uint64_t n = vulkanStats.warps ? vulkanStats.warps : 1;
        uint64_t g = vulkanStats.gpuTimedWarps ? vulkanStats.gpuTimedWarps : 1;
        std::cout << "[vulkan] stats: warps=" << vulkanStats.warps
                  << " uploads=" << vulkanStats.uploads << " skips=" << vulkanStats.skips
                  << " | ms/warp readback=" << vulkanStats.readbackMs / n
                  << " upload=" << vulkanStats.uploadMs / n << " cpu-warp=" << vulkanStats.warpMs / n
                  << " present=" << vulkanStats.presentMs / n;
        if(vulkanStats.gpuTimedWarps) {
            std::cout << " | gpu splat=" << vulkanStats.gpuSplatMs / g
                      << " pull=" << vulkanStats.gpuPullMs / g
                      << " fill=" << vulkanStats.gpuFillMs / g
                      << " copies=" << vulkanStats.gpuCopyMs / g;
        } else {
            std::cout << " | gpu timestamps off";
        }
        std::cout << " | coverage=" << vulkanStats.lastCoverage
                  << " | present-fps=" << presentationFramerate << std::endl;
        vulkanStats.warps = vulkanStats.uploads = vulkanStats.skips = 0;
        vulkanStats.readbackMs = vulkanStats.uploadMs = vulkanStats.warpMs = vulkanStats.presentMs =
            0.0;
        vulkanStats.gpuSplatMs = vulkanStats.gpuPullMs = vulkanStats.gpuFillMs =
            vulkanStats.gpuCopyMs = 0.0;
        vulkanStats.gpuTimedWarps = 0;
    }
    return 1;
}
#endif // OPENWARP_HAS_VULKAN

// Returns true when anything new was painted to the back buffer.
bool OpenwarpApplication::doReprojection(bool useRay){

    if(backend == Backend::Vulkan) {
        // Raymarch is GL-only; never raymarch under --vulkan.
        useRay = false;
#ifdef OPENWARP_HAS_VULKAN
        // The Vulkan compute warp IS the mesh warp in this mode.
        // 1 = presented, 0 = skipped identical frame (hold the last
        // one), -1 = fall through to the GL mesh path below.
        int vulkanResult = doVulkanMeshWarp();
        if(vulkanResult >= 0)
            return vulkanResult > 0;
        // Fell through: Vulkan unavailable/failed, use the GL mesh path
        // so the demo stays alive and frame-comparable.
        if(!vulkanModeLogged) {
            vulkanModeLogged = true;
            std::cout << "[vulkan] warp unavailable, falling back to GL mesh warp." << std::endl;
        }
#else
        if(!vulkanModeLogged) {
            vulkanModeLogged = true;
            std::cout << "[vulkan] binary built without Vulkan support; using GL mesh warp." << std::endl;
        }
#endif
    }

    if(useRay) {
        glBindVertexArray(rayProgram.vao);
        glUseProgram(rayProgram.program);

        // Upload matrices of the rendered frame.
        glUniformMatrix4fv(rayProgram.u_renderPV, 1, GL_FALSE, (GLfloat*)((projection * renderedCameraMatrix.inverse()).eval().data()));

        glUniform3fv(rayProgram.u_warpPos, 1, position.data());
        // Calculate a fresh camera matrix.
        auto freshCameraMatrix = createCameraMatrix(position, orientation);

        // Compute VP matrix for fresh pose.
        // auto freshVP = projection * freshCameraMatrix.inverse();

        glUniformMatrix4fv(rayProgram.u_warpInverseVP, 1, GL_FALSE, (GLfloat*)((freshCameraMatrix * projection.inverse()).eval().data()));

        // Uploade parameter/config uniforms
        glUniform1f(rayProgram.u_power, rayPower);
        glUniform1f(rayProgram.u_stepSize, rayStepSize);
        glUniform1f(rayProgram.u_depthOffset, rayDepthOffset);
        glUniform1f(rayProgram.u_occlusionThreshold, occlusionThreshold);
        glUniform1f(rayProgram.u_occlusionOffset, occlusionOffset);

    } else {
        glBindVertexArray(meshProgram.vao);
        glUseProgram(meshProgram.program);

        // Upload inverse view matrix (camera matrix) of the rendered frame.
        glUniformMatrix4fv(meshProgram.u_renderInverseV, 1, GL_FALSE, (GLfloat*)(renderedCameraMatrix.data()));

        // Calculate a fresh camera matrix.
        auto freshCameraMatrix = createCameraMatrix(position, orientation);

        // Compute VP matrix for fresh pose.
        auto freshVP = projection * freshCameraMatrix.inverse();

        // Upload the fresh VP matrix.
        glUniformMatrix4fv(meshProgram.u_warp_vp, 1, GL_FALSE, (GLfloat*)freshVP.eval().data());

        glUniform1f(meshProgram.u_bleedRadius, bleedRadius);
        glUniform1f(meshProgram.u_bleedTolerance, bleedTolerance);
        glUniform1f(meshProgram.u_debugOpacity, showDebugGrid ? 1.0f : 0.0f);
    }

    // Render directly to screen. If we were going to send this to a
    // lens undistort shader, we'd create another FBO and render to that.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Full framebuffer (not WIDTHxHEIGHT): on HiDPI the surface is
    // larger and a small viewport would leave a stale rim of the last
    // reprojected frame on screen.
    int glW = WIDTH, glH = HEIGHT;
    glfwGetFramebufferSize(window, &glW, &glH);
    glViewport(0, 0, glW, glH);
    glDisable(GL_CULL_FACE);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, useRay ? rayProgram.mesh_vertices_vbo : meshProgram.mesh_vertices_vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, uv));

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depthTexture);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, useRay ? rayProgram.mesh_indices_vbo : meshProgram.mesh_indices_vbo);
    glDrawElements(GL_TRIANGLES, useRay ? rayProgram.mesh_indices.size() : meshProgram.mesh_indices.size(), GL_UNSIGNED_INT, NULL);

    return true;
}

void OpenwarpApplication::renderScene(){
    // Render to FBO.
    glBindVertexArray(demoVAO);
    glUseProgram(demoShaderProgram);
#ifdef OPENWARP_HAS_VULKAN
    // Fresh source for the Vulkan warp path (dirty tracking).
    renderSeq++;
#endif

    // If reprojection is disabled, we render straight to the screen
    // (full framebuffer: a WIDTHxHEIGHT viewport on a larger HiDPI
    // surface would leave a stale rim of the last reprojected frame).
    if (shouldReproject) {
        glBindFramebuffer(GL_FRAMEBUFFER, renderFBO);
        glViewport(0, 0, WIDTH, HEIGHT);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        int fbW = WIDTH, fbH = HEIGHT;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
    }
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glClearDepth(1);

    // Set up user view matrix.
    // We save the camera matrix, so that when Openwarp runs, it can use both
    // the rendered camera matrix, as well as the updated "fresh" camera matrix.
    renderedCameraMatrix = createCameraMatrix(position, orientation);
    renderedView = renderedCameraMatrix.inverse();

    glUniformMatrix4fv(demoModelViewAttr, 1, GL_FALSE, (GLfloat*)(renderedView.data()));
    glUniformMatrix4fv(demoProjectionAttr, 1, GL_FALSE, (GLfloat*)(projection.data()));    

    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    demoscene.Draw();
    glFlush();
}

OpenwarpApplication::~OpenwarpApplication(){
    cleanupGL();
    glfwDestroyWindow(window);
    glfwTerminate();
}

int OpenwarpApplication::initGL(){

    glfwSwapInterval(useVsync ? 1 : 0);

    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );

    // Create both color and depth render textures
    createRenderTexture(&renderTexture, WIDTH, HEIGHT, false);
    createRenderTexture(&depthTexture, WIDTH, HEIGHT, true);

    // Create FBO that will render to them!
    createFBO(&renderTexture, &renderFBO, &depthTexture, WIDTH, HEIGHT);

    // Load the .obj-file-based that will be rendered for the demo scene.
    demoscene = ObjScene(std::string(OBJ_DIR), "scene.obj");

    projection = perspective(45.0, (double)(WIDTH)/(double)(HEIGHT), 0.1, 100.0);

    // DEMO rendering initialization
    ////////////////////////////////

    // Initialize and link shader program used to render demo scene
    demoShaderProgram = init_and_link("../resources/shaders/demo.vert", "../resources/shaders/demo.frag");

    // Get demo shader attributes
    demoVertexPosAttr = glGetAttribLocation(demoShaderProgram, "vertexPosition");
    demoModelViewAttr = glGetUniformLocation(demoShaderProgram, "u_modelview");
    demoProjectionAttr = glGetUniformLocation(demoShaderProgram, "u_projection");

    // Create and bind global VAO object
    glGenVertexArrays(1, &demoVAO);
    glBindVertexArray(demoVAO);

    // Openwarp-mesh rendering initialization
    ////////////////////////////////

    glGenVertexArrays(1, &meshProgram.vao);
    glBindVertexArray(meshProgram.vao);

    // Build the reprojection mesh for mesh-based Openwarp
	BuildMesh(meshWidth, meshHeight, meshProgram.mesh_indices, meshProgram.mesh_vertices);

    // Build and link shaders for openwarp-mesh.
	meshProgram.program = init_and_link("../resources/shaders/openwarp_mesh.vert", "../resources/shaders/openwarp_mesh.frag");

    // Get the color + depth samplers
    meshProgram.eye_sampler = glGetUniformLocation(meshProgram.program, "Texture");
    meshProgram.depth_sampler = glGetUniformLocation(meshProgram.program, "_Depth");

    // Get the warp matrix uniforms
    // Inverse V and P matrices of the rendered pose
    meshProgram.u_renderInverseP = glGetUniformLocation(meshProgram.program, "u_renderInverseP");
    meshProgram.u_renderInverseV = glGetUniformLocation(meshProgram.program, "u_renderInverseV");
    // VP matrix of the fresh pose
    meshProgram.u_warp_vp = glGetUniformLocation(meshProgram.program, "u_warpVP");

    // Mesh edge bleed parameters
    meshProgram.u_bleedRadius = glGetUniformLocation(meshProgram.program, "bleedRadius");
    meshProgram.u_bleedTolerance = glGetUniformLocation(meshProgram.program, "edgeTolerance");

    // Mesh edge bleed parameters
    meshProgram.u_debugOpacity = glGetUniformLocation(meshProgram.program, "u_debugOpacity");

    // Generate, bind, and fill mesh VBOs.
    glGenBuffers(1, &meshProgram.mesh_vertices_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, meshProgram.mesh_vertices_vbo);
    glBufferData(GL_ARRAY_BUFFER, meshProgram.mesh_vertices.size() * sizeof(vertex_t), &meshProgram.mesh_vertices.at(0), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &meshProgram.mesh_indices_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshProgram.mesh_indices_vbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshProgram.mesh_indices.size() * sizeof(GLuint), &meshProgram.mesh_indices.at(0), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Openwarp-ray rendering initialization
    //////////////////////////////

    glGenVertexArrays(1, &rayProgram.vao);
    glBindVertexArray(rayProgram.vao);

    // Build the reprojection mesh for ray-based Openwarp
    // 2x2 quad
	BuildMesh(2, 2, rayProgram.mesh_indices, rayProgram.mesh_vertices);

    // Build and link shaders for openwarp-ray.
	rayProgram.program = init_and_link("../resources/shaders/openwarp_ray.vert", "../resources/shaders/openwarp_ray.frag");

    // Get the color + depth samplers
    rayProgram.eye_sampler = glGetUniformLocation(rayProgram.program, "Texture");
    rayProgram.depth_sampler = glGetUniformLocation(rayProgram.program, "_Depth");

    // Get the warp matrix uniforms
    // Inverse V and P matrices of the rendered pose
    rayProgram.u_renderPV = glGetUniformLocation(rayProgram.program, "u_renderPV");
    rayProgram.u_warpInverseVP = glGetUniformLocation(rayProgram.program, "u_warpInverseVP");
    rayProgram.u_warpPos = glGetUniformLocation(rayProgram.program, "u_warpPos");

    rayProgram.u_power = glGetUniformLocation(rayProgram.program, "u_power");
    rayProgram.u_stepSize = glGetUniformLocation(rayProgram.program, "u_stepSize");
    rayProgram.u_depthOffset = glGetUniformLocation(rayProgram.program, "u_depthOffset");
    rayProgram.u_occlusionThreshold = glGetUniformLocation(rayProgram.program, "u_occlusionThreshold");
    rayProgram.u_occlusionOffset = glGetUniformLocation(rayProgram.program, "u_occlusionOffset");


    // Generate, bind, and fill mesh VBOs.
    glGenBuffers(1, &rayProgram.mesh_vertices_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, rayProgram.mesh_vertices_vbo);
    glBufferData(GL_ARRAY_BUFFER, rayProgram.mesh_vertices.size() * sizeof(vertex_t), &rayProgram.mesh_vertices.at(0), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &rayProgram.mesh_indices_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rayProgram.mesh_indices_vbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rayProgram.mesh_indices.size() * sizeof(GLuint), &rayProgram.mesh_indices.at(0), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Upload the projection matrix (and inverse projection matrix) to the
    // demo and openwarp-mesh programs. Should only need to do this once;
    // we won't be changing this projection matrix at runtime (non-resizeable window)
    glUseProgram(demoShaderProgram);
    glUniformMatrix4fv(demoProjectionAttr, 1, GL_FALSE, (GLfloat*)(projection.data()));
    glUseProgram(meshProgram.program);
    glUniformMatrix4fv(meshProgram.u_renderInverseP, 1, GL_FALSE, (GLfloat*)(projection.inverse().eval().data()));
    glUseProgram(rayProgram.program);
    glUseProgram(0);

    return 0;
}

int OpenwarpApplication::cleanupGL(){
    return 0;
}