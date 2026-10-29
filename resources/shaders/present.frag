// Fullscreen present quad for the --vulkan path. Part of the Openwarp
// project (NCSA).
#version 410 core

uniform sampler2D present_tex;
in vec2 uv;
out vec4 outcolor;

void main() {
    outcolor = texture(present_tex, uv);
}
