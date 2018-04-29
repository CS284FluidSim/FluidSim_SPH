#ifndef _FLUIDSIM_GL_UTILS_H_
#define _FLUIDSIM_GL_UTILS_H_

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <string>

using namespace std;

namespace FluidSim {

	GLuint create_shader_program(string vs_path, string fs_path);

	GLuint create_cube_vao(int size);

	bool load_cube_map_side(GLuint texture, GLenum side_target, const char *file_name);

	GLuint create_cube_map_tex(const char *front, const char *back, const char *top,
		const char *bottom, const char *left, const char *right);

	class Camera
	{
	private:

	public:
		Camera();
	};

}

#endif