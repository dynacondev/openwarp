#version 410 core

uniform sampler2D main_tex;
in vec2 uv;
out vec4 outcolor;

void main() {
    outcolor = texture(main_tex, uv);
}