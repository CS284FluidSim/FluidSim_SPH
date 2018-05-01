/******************************************************************************\
| OpenGL 4 Example Code.                                                       |
| Accompanies written series "Anton's OpenGL 4 Tutorials"                      |
| Email: anton at antongerdelan dot net                                        |
| First version 27 Jan 2014                                                    |
| Copyright Dr Anton Gerdelan, Trinity College Dublin, Ireland.                |
| See individual libraries separate legal notices                              |
|******************************************************************************|
| This is just a file holding some commonly-used "utility" functions to keep   |
| the main file a bit easier to read. You can might build up something like    |
| this as learn more GL. Note that you don't need much code here to do good GL.|
| If you have a big object-oriented engine then maybe you can ask yourself if  |
| it is really making life easier.                                             |
\******************************************************************************/
#include "fluidsim_gl_utils.h"
#include "maths_funcs.h"
#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#define GL_LOG_FILE "gl.log"
#define MAX_SHADER_LENGTH 262144

extern mat4 proj_mat;

/* big cube. returns Vertex Array Object */
GLuint make_cube(float size) {
	float points[] = {
		-size, size,	-size, -size, -size, -size, size,	-size, -size,
		size,	-size, -size, size,	size,	-size, -size, size,	-size,

		-size, -size, size,	-size, -size, -size, -size, size,	-size,
		-size, size,	-size, -size, size,	size,	-size, -size, size,

		size,	-size, -size, size,	-size, size,	size,	size,	size,
		size,	size,	size,	size,	size,	-size, size,	-size, -size,

		-size, -size, size,	-size, size,	size,	size,	size,	size,
		size,	size,	size,	size,	-size, size,	-size, -size, size,

		-size, size,	-size, size,	size,	-size, size,	size,	size,
		size,	size,	size,	-size, size,	size,	-size, size,	-size,

		-size, -size, -size, -size, -size, size,	size,	-size, -size,
		size,	-size, -size, -size, -size, size,	size,	-size, size
	};
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * 36 * sizeof(GLfloat), &points,
		GL_STATIC_DRAW);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	return vao;
}

GLuint make_triangle()
{
	/* OTHER STUFF GOES HERE NEXT */
	GLfloat points[] = { 0.0f, 1.0f, 0.2f, 0.5f, -0.5f, 0.2f, -0.5f, -0.5f, 0.2f };

	//GLfloat colours[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };

	GLuint points_vbo;
	glGenBuffers(1, &points_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(GLfloat), points, GL_STATIC_DRAW);

	/* create the VAO.
	we bind each VBO in turn, and call glVertexAttribPointer() to indicate where
	the memory should be fetched for vertex shader input variables 0, and 1,
	respectively. we also have to explicitly enable both 'attribute' variables.
	'attribute' is the older name for vertex shader 'in' variables. */
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	return vao;
}

/* use stb_image to load an image file into memory, and then into one side of
a cube-map texture. */
bool load_cube_map_side(GLuint texture, GLenum side_target,
	std::string file_name) {
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

	int x, y, n;
	int force_channels = 4;
	unsigned char *image_data = stbi_load(file_name.c_str(), &x, &y, &n, force_channels);
	if (!image_data) {
		fprintf(stderr, "ERROR: could not load %s\n", file_name);
		return false;
	}
	// non-power-of-2 dimensions check
	if ((x & (x - 1)) != 0 || (y & (y - 1)) != 0) {
		fprintf(stderr, "WARNING: image %s is not power-of-2 dimensions\n",
			file_name);
	}

	// copy image data into 'target' side of cube map
	glTexImage2D(side_target, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
		image_data);
	free(image_data);
	return true;
}

/* load all 6 sides of the cube-map from images, then apply formatting to the
final texture */
void create_cube_map(std::string base_path, GLuint *tex_cube) {
	// generate a cube-map texture to hold all the sides
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, tex_cube);

	// load each image and copy into a side of the cube-map texture
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, base_path + "negz.png "));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, base_path + "posz.png "));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, base_path + "posy.png "));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, base_path + "negy.png "));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, base_path + "negx.png "));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_X, base_path + "posx.png "));
	// format cube map texture
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/*--------------------------------LOG FUNCTIONS-------------------------------*/
bool restart_gl_log() {
	FILE *file = fopen(GL_LOG_FILE, "w");
	if (!file) {
		fprintf(stderr, "ERROR: could not open GL_LOG_FILE log file %s for writing\n",
			GL_LOG_FILE);
		return false;
	}
	time_t now = time(NULL);
	char *date = ctime(&now);
	fprintf(file, "GL_LOG_FILE log. local time %s\n", date);
	fclose(file);
	return true;
}

bool gl_log(const char *message, ...) {
	va_list argptr;
	FILE *file = fopen(GL_LOG_FILE, "a");
	if (!file) {
		fprintf(stderr, "ERROR: could not open GL_LOG_FILE %s file for appending\n",
			GL_LOG_FILE);
		return false;
	}
	va_start(argptr, message);
	vfprintf(file, message, argptr);
	va_end(argptr);
	fclose(file);
	return true;
}

/* same as gl_log except also prints to stderr */
bool gl_log_err(const char *message, ...) {
	va_list argptr;
	FILE *file = fopen(GL_LOG_FILE, "a");
	if (!file) {
		fprintf(stderr, "ERROR: could not open GL_LOG_FILE %s file for appending\n",
			GL_LOG_FILE);
		return false;
	}
	va_start(argptr, message);
	vfprintf(file, message, argptr);
	va_end(argptr);
	va_start(argptr, message);
	vfprintf(stderr, message, argptr);
	va_end(argptr);
	fclose(file);
	return true;
}

/*--------------------------------GLFW3 and GLEW------------------------------*/
bool start_gl() {
	gl_log("starting GLFW %s\n", glfwGetVersionString());

	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		fprintf(stderr, "ERROR: could not start GLFW3\n");
		return false;
	}

	/* We must specify 3.2 core if on Apple OS X -- other O/S can specify
	anything here. I defined 'APPLE' in the makefile for OS X */
#ifdef APPLE
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

	/*GLFWmonitor* mon = glfwGetPrimaryMonitor ();
	const GLFWvidmode* vmode = glfwGetVideoMode (mon);
	g_window = glfwCreateWindow (
	vmode->width, vmode->height, "Extended GL Init", mon, NULL
	);*/

	g_window =
		glfwCreateWindow(g_gl_width, g_gl_height, "Extended Init.", NULL, NULL);
	if (!g_window) {
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
		return false;
	}
	glfwSetFramebufferSizeCallback(g_window, glfw_framebuffer_size_callback);
	glfwMakeContextCurrent(g_window);

	glfwWindowHint(GLFW_SAMPLES, 4);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	// get version info
	const GLubyte *renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte *version = glGetString(GL_VERSION);		// version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
	gl_log("renderer: %s\nversion: %s\n", renderer, version);

	return true;
}

void glfw_error_callback(int error, const char *description) {
	fputs(description, stderr);
	gl_log_err("%s\n", description);
}
// a call-back function
void glfw_framebuffer_size_callback(GLFWwindow *window, int width, int height) {
	g_gl_width = width;
	g_gl_height = height;
	printf("width %i height %i\n", width, height);
	/* update any perspective matrices used here */
	glViewport(0, 0, width, height);
	proj_mat = perspective(45.f, (float)width / height, 0.1f, 100.f);
}

void _update_fps_counter(GLFWwindow *window) {
	static double previous_seconds = glfwGetTime();
	static int frame_count;
	double current_seconds = glfwGetTime();
	double elapsed_seconds = current_seconds - previous_seconds;
	if (elapsed_seconds > 0.25) {
		previous_seconds = current_seconds;
		double fps = (double)frame_count / elapsed_seconds;
		char tmp[128];
		sprintf(tmp, "opengl @ fps: %.2f", fps);
		glfwSetWindowTitle(window, tmp);
		frame_count = 0;
	}
	frame_count++;
}

/*----------------------------------TEXTURES----------------------------------*/
bool load_texture(const char *file_name, GLuint *tex) {
	int x, y, n;
	int force_channels = 4;
	unsigned char *image_data = stbi_load(file_name, &x, &y, &n, force_channels);
	if (!image_data) {
		fprintf(stderr, "ERROR: could not load %s\n", file_name);
		return false;
	}
	// NPOT check
	if ((x & (x - 1)) != 0 || (y & (y - 1)) != 0) {
		fprintf(stderr, "WARNING: texture %s is not power-of-2 dimensions\n",
			file_name);
	}
	int width_in_bytes = x * 4;
	unsigned char *top = NULL;
	unsigned char *bottom = NULL;
	unsigned char temp = 0;
	int half_height = y / 2;

	for (int row = 0; row < half_height; row++) {
		top = image_data + row * width_in_bytes;
		bottom = image_data + (y - row - 1) * width_in_bytes;
		for (int col = 0; col < width_in_bytes; col++) {
			temp = *top;
			*top = *bottom;
			*bottom = temp;
			top++;
			bottom++;
		}
	}
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
		image_data);
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	GLfloat max_aniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);
	// set the maximum!
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_aniso);
	return true;
}
