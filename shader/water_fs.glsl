#version 410

const float Air = 1.0;
const float Glass = 1.51714;

const float Eta = Air / Glass;

const float R0 = ((Air - Glass) * (Air - Glass)) / ((Air + Glass) * (Air + Glass));

in vec3 pos_eye;
in vec3 n_eye;
uniform samplerCube cube_texture;
uniform mat4 V;
out vec4 frag_colour;

void main () {
	/* reflect ray around normal from eye to surface */
	vec3 incident_eye = normalize (pos_eye);
	vec3 normal = normalize (n_eye);

	vec3 refracted = refract (incident_eye, normal, Eta);
	refracted = vec3 (inverse (V) * vec4 (refracted, 0.0));
	vec4 color_refracted = texture (cube_texture, refracted);

  vec3 reflected = reflect (incident_eye, normal);
	reflected = vec3 (inverse (V) * vec4 (reflected, 0.0));
  vec4 color_reflected = texture (cube_texture, reflected);

	float fresnel = R0 + (1.0 - R0) * pow((1.0 - dot(-incident_eye, normal)), 5.0);

	frag_colour = mix(color_refracted, color_reflected, fresnel);
}
