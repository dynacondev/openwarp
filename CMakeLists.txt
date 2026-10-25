cmake_minimum_required(VERSION 3.10)

# Vendored 2020-era submodules (glfw/glm/glew-cmake) declare
# cmake_minimum_required(< 3.5), which CMake >= 4.0 rejects. Keep the pins
# (tested combo) and opt into the 3.5 compat floor instead of updating deps.
# No-op on CMake < 4. Equivalent to configuring with
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5. Remove when submodules are bumped.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Compat floor for vendored pre-3.5 submodules")
endif()



project(openwarp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("${CMAKE_CXX_COMPILER_ID}")

message("Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
    endif()
endif()

add_executable(openwarp
        src/openwarp/main.cpp
        src/openwarp/openwarp.hpp
        src/openwarp/OpenwarpApplication.cpp
        src/openwarp/OpenwarpApplication.hpp
        src/openwarp/testrun.cpp
        src/openwarp/testrun.hpp
        src/openwarp/util/lib/stb_image.h
        src/openwarp/util/lib/tiny_obj_loader.h
        src/openwarp/util/obj.hpp
        src/openwarp/util/obj.cpp
        
        # imgui does not support cmake... yet!
        include/imgui/imgui_widgets.cpp
        include/imgui/imgui_draw.cpp
        include/imgui/imgui_impl_glfw.cpp
        include/imgui/imgui_impl_opengl3.cpp
        include/imgui/imgui.cpp
        include/imgui/imgui_demo.cpp
)

# std::filesystem
target_link_libraries(openwarp PUBLIC stdc++fs)

# imgui does not support cmake... yet!
include_directories(include/imgui)

# Header-only libraries
include_directories(include/eigen)

# Build-type default: Release. The NN-fill CPU loops (downsample to
# 1x6x512x512, bilinear upsample + hard-mask composite at 1024x1024) run
# ~10x slower under -O0, which reads as "NN fill runs at 8fps" on a perf
# host that silently stayed Debug. An explicit -DCMAKE_BUILD_TYPE=... (or
# an existing cache value) still wins; this only fills in the unset case.
# 3060 host configure line:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release   (fresh dir; or wipe build/)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release recommended: NN fill is ~10x slower in Debug)" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} (NN fill needs Release/O2; Debug logs a stderr warning at nn-fill init)")

set_property(TARGET openwarp PROPERTY CXX_STANDARD 17)
target_compile_options(openwarp PRIVATE -Wall)

# glfw
add_subdirectory(include/glfw EXCLUDE_FROM_ALL)
target_link_libraries(openwarp PRIVATE glfw)

# glew
add_definitions(-DGLEW_STATIC)
add_subdirectory(include/glew-cmake EXCLUDE_FROM_ALL)
target_link_libraries(openwarp PRIVATE libglew_static)

# glm
add_subdirectory(include/glm EXCLUDE_FROM_ALL)
target_link_libraries(openwarp PRIVATE glm)

# Vulkan compute warp backend (optional -- build stays GL-only without it).
# The Vulkan SDK provides headers + loader; the backend itself lands in
# src/openwarp/vulkan/ (commit 2) and is compiled only when found.
option(OPENWARP_ENABLE_VULKAN "Build with Vulkan compute warp backend (needs Vulkan SDK)" ON)
if(OPENWARP_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan ${Vulkan_VERSION} found - enabling --vulkan backend scaffold")
        target_compile_definitions(openwarp PRIVATE OPENWARP_HAS_VULKAN=1)
        target_link_libraries(openwarp PRIVATE Vulkan::Vulkan)

        # Compile the compute warp shaders to SPIR-V at build time.
        find_program(OPENWARP_GLSLC glslc)
        if(NOT OPENWARP_GLSLC)
            message(WARNING "glslc not found - Vulkan warp shaders will not be built")
        else()
            set(OPENWARP_SPV_DIR "${CMAKE_BINARY_DIR}/openwarp-shaders")
            file(MAKE_DIRECTORY "${OPENWARP_SPV_DIR}")
            set(OPENWARP_SPV
                "${OPENWARP_SPV_DIR}/warp_splat.spv"
                "${OPENWARP_SPV_DIR}/warp_pull.spv"
                "${OPENWARP_SPV_DIR}/warp_fill.spv")
            add_custom_command(
                OUTPUT "${OPENWARP_SPV_DIR}/warp_splat.spv"
                COMMAND "${OPENWARP_GLSLC}" --target-env=vulkan1.0 -O
                        "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_splat.comp"
                        -o "${OPENWARP_SPV_DIR}/warp_splat.spv"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_splat.comp"
                COMMENT "Compiling warp_splat.comp")
            add_custom_command(
                OUTPUT "${OPENWARP_SPV_DIR}/warp_pull.spv"
                COMMAND "${OPENWARP_GLSLC}" --target-env=vulkan1.0 -O
                        "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_pull.comp"
                        -o "${OPENWARP_SPV_DIR}/warp_pull.spv"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_pull.comp"
                COMMENT "Compiling warp_pull.comp")
            add_custom_command(
                OUTPUT "${OPENWARP_SPV_DIR}/warp_fill.spv"
                COMMAND "${OPENWARP_GLSLC}" --target-env=vulkan1.0 -O
                        "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_fill.comp"
                        -o "${OPENWARP_SPV_DIR}/warp_fill.spv"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resources/shaders/vulkan/warp_fill.comp"
                COMMENT "Compiling warp_fill.comp")
            add_custom_target(openwarp_vulkan_shaders ALL DEPENDS ${OPENWARP_SPV})

            # Headless compute warp library (Option B port).
            add_library(openwarp_vulkan STATIC
                src/openwarp/vulkan/vulkan_warp.cpp
                src/openwarp/vulkan/vulkan_warp.hpp
                src/openwarp/vulkan/vulkan_nn_fill.cpp
                src/openwarp/vulkan/vulkan_nn_fill.hpp)
            target_include_directories(openwarp_vulkan PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/openwarp"
                include/eigen)
            target_link_libraries(openwarp_vulkan PRIVATE Vulkan::Vulkan)
            # Blur-init weights for the NN hole fill (mode 1). The app
            # also probes OPENWARP_NN_MODEL + relative ml/weights layouts.
            target_compile_definitions(openwarp_vulkan PRIVATE
                "OPENWARP_NN_MODEL_DEFAULT=\"${CMAKE_CURRENT_SOURCE_DIR}/ml/weights/fill_blur_init.onnx\"")
            # ONNX Runtime C++ (optional): real TinyFillNet inference.
            # Ubuntu ships no libonnxruntime-dev package and the pip
            # onnxruntime wheels carry no C++ headers, so point
            # ONNXRUNTIME_ROOT at a github.com/microsoft/onnxruntime
            # release tarball dir (include/ + lib/) to enable. Without
            # it the build stays green and VulkanNnFill runs the mock
            # box-blur backend (same plumbing + hard-mask composite).
            set(ONNXRUNTIME_ROOT "" CACHE PATH "Prefix of an onnxruntime release (include/ + lib/)")
            find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
                HINTS ${ONNXRUNTIME_ROOT}/include /usr/include/onnxruntime /usr/local/include/onnxruntime)
            find_library(ONNXRUNTIME_LIB onnxruntime
                HINTS ${ONNXRUNTIME_ROOT}/lib /usr/lib /usr/local/lib)
            if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
                message(STATUS "onnxruntime C++ found: ${ONNXRUNTIME_LIB} - NN fill uses ORT")
                target_include_directories(openwarp_vulkan PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
                target_link_libraries(openwarp_vulkan PRIVATE ${ONNXRUNTIME_LIB})
                target_compile_definitions(openwarp_vulkan PRIVATE OPENWARP_HAS_ORT=1)
                # CUDA-EP dep preload (dlopen-by-absolute-path in
                # vulkan_nn_fill.cpp) needs dlfcn; _GNU_SOURCE exposes
                # dladdr()/RTLD_DEFAULT. CMAKE_DL_LIBS is empty on glibc
                # >= 2.34 (dlopen in libc) — harmless either way.
                target_compile_definitions(openwarp_vulkan PRIVATE _GNU_SOURCE)
                target_link_libraries(openwarp_vulkan PRIVATE ${CMAKE_DL_LIBS})
            else()
                message(STATUS "onnxruntime C++ not found - NN fill uses mock box-blur backend")
            endif()
            # LibTorch C++ (optional, preferred over ORT when both are
            # present: torch's CUDA stacking works on the 3060 host where
            # ORT's prebuilt CUDA EP kept failing loader-side). Point
            # TORCH_ROOT at an unpacked libtorch dir (it holds
            # share/cmake/Torch/TorchConfig.cmake); without it (and without
            # ORT) the build stays green with the mock backend. Libtorch
            # bundles its own CUDA libs with $ORIGIN runpaths, so unlike
            # ORT it needs no build.sh CUDA staging. Requires C++17, which
            # this target already sets; no extra defines are added beyond
            # what Torch itself demands.
            set(TORCH_ROOT "" CACHE PATH "Prefix of an unpacked libtorch distribution (share/cmake/Torch/)")
            if(TORCH_ROOT)
                list(APPEND CMAKE_PREFIX_PATH "${TORCH_ROOT}")
            endif()
            find_package(Torch QUIET)
            if(Torch_FOUND)
                message(STATUS "libtorch found - NN fill uses TorchScript (preferred over ORT when both link)")
                target_link_libraries(openwarp_vulkan PRIVATE ${TORCH_LIBRARIES})
                target_compile_definitions(openwarp_vulkan PRIVATE OPENWARP_HAS_TORCH=1)
            else()
                message(STATUS "libtorch not found (TORCH_ROOT=${TORCH_ROOT}) - NN fill uses ORT/mock backend")
            endif()
            set_property(TARGET openwarp_vulkan PROPERTY CXX_STANDARD 17)
            add_dependencies(openwarp_vulkan openwarp_vulkan_shaders)
            target_link_libraries(openwarp PRIVATE openwarp_vulkan)
            target_compile_definitions(openwarp PRIVATE
                "OPENWARP_SPV_DIR_DEFAULT=\"${OPENWARP_SPV_DIR}\"")

            # Headless test: synthetic warp on lavapipe, no window needed.
            add_executable(openwarp-vulkan-test
                src/openwarp/vulkan/vulkan_warp_test.cpp)
            target_include_directories(openwarp-vulkan-test PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/openwarp"
                include/eigen)
            target_link_libraries(openwarp-vulkan-test PRIVATE openwarp_vulkan stdc++fs)
            set_property(TARGET openwarp-vulkan-test PROPERTY CXX_STANDARD 17)
            add_dependencies(openwarp-vulkan-test openwarp_vulkan_shaders)
        endif()
    else()
        message(STATUS "Vulkan not found - building GL-only; --vulkan will fall back with a warning")
    endif()
endif()

