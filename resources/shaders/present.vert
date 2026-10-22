// Fullscreen present quad for the --vulkan path: draws the Vulkan-warped
// result texture to the screen. Part of the Openwarp project (NCSA).
#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

out vec2 uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
