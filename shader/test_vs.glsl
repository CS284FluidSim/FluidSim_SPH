#version 410

uniform mat4 P, V;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;

out vec3 color;

void main() {
    color = vertex_color;
		gl_Position = P * V * vec4 (vertex_position, 1.0);
}
