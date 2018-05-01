#version 410

in vec3 vp;
uniform mat4 M, P, V;
out vec3 texcoords;

void main () {
	texcoords = vp;
	gl_Position = P * V * M * vec4 (vp, 1.0);
}
