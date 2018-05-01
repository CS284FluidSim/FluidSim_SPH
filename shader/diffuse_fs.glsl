#version 410

const vec3 I_l = vec3(0.3,0.3,0.3);
const vec3 I_a = vec3(0.0,0.2,0.3);

uniform samplerCube cube_texture;
uniform mat4 V; // view matrix

in vec3 vp_cam;
in vec3 vn_cam;
in vec3 ld_cam;

out vec4 frag_colour;

void main () {
	/* reflect ray around normal from eye to surface */
	vec3 dir_cam = normalize (vp_cam);
	vec3 dir_normal = normalize (vn_cam);
  float r = length(ld_cam);
  vec3 dir_l = normalize(ld_cam);

  vec3 dir_h = (dir_cam+dir_l)/length(dir_cam+dir_l);

  float k_a=0.4,k_d=0.0;
  vec3 L = k_a * I_a + k_d * I_l/(r*r)*clamp(dot(dir_normal,dir_l),0.0,1.0);
	frag_colour = vec4(L, 1.0);
}
