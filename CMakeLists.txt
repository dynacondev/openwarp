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

set(CMAKE_BUILD_TYPE Debug)

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
                src/openwarp/vulkan/vulkan_warp.hpp)
            target_include_directories(openwarp_vulkan PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/openwarp"
                include/eigen)
            target_link_libraries(openwarp_vulkan PRIVATE Vulkan::Vulkan)
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

