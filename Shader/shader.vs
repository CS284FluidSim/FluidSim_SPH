attribute vec3 position;
attribute vec3 normal;

varying vec3 fPosition;
varying vec3 fNormal;

void main() {
	// TODO: Part 5.1
    fPosition = (gl_ModelViewMatrix*vec4(position, 1.0)).xyz;
    fNormal = gl_Normal;
    gl_Position = gl_ModelViewProjectionMatrix*vec4(position, 1.0);
}
