varying vec3 fPosition;
varying vec3 fNormal;

void main() {
    vec3 lPosition = vec3(0.0,0.0,0.0);
    vec3 lIntensity = vec3(1.0,1.0,1.0);
    vec3 l = lPosition - fPosition;
    float r = length(l);
    l = normalize(l);
    float k_a = 0.3;
    float k_d = 1.0;
    vec3 I_a = vec3(1.0,1.0,1.0);
    vec3 L_d = k_a * I_a + k_d*lIntensity/(r*r)*clamp(dot(fNormal,l),0.0,1.0);
    gl_FragColor = vec4(fNormal, 1.0);
}
