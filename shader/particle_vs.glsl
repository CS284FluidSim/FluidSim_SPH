#version 410

layout(location = 0) in vec3 vp; // positions from mesh
layout(location = 1) in vec3 vc; // normals from mesh
uniform mat4 P, V, M; // proj, view, model matrices

out vec3 color;

void main () {
	color = vc;
	gl_Position = P * V * M * vec4 (vp, 1.0);
}
