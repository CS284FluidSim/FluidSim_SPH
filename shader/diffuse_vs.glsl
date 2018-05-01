#version 410

const vec3 light_pos = vec3(0.0,1.0,3.0);

layout(location = 0) in vec3 vp; // positions from mesh
layout(location = 1) in vec3 vn; // normals from mesh
uniform mat4 P, V, M; // proj, view, model matrices
out vec3 vp_cam;
out vec3 vn_cam;
out vec3 ld_cam;

void main () {
	vp_cam = vec3 (V * M * vec4 (vp, 1.0));
	vn_cam = vec3 (V * M * vec4 (vn, 0.0));
  ld_cam = vec3 (V * M * vec4 (light_pos, 1.0)) - vp_cam;
	gl_Position = P * V * M * vec4 (vp, 1.0);
}
