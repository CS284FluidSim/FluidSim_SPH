#ifndef _FLUIDSIM_SHADER_H_
#define _FLUIDSIM_SHADER_H_

#include "fluidsim_gl_utils.h"

#include <string>
#include <unordered_map>

namespace FluidSim
{
	class Shader
	{
		GLuint vert_shader_;
		GLuint frag_shader_;
		GLuint shader_program_;
		bool is_created_ = false;
		std::unordered_map<std::string, unsigned int> uniform_loc_;
	public:
		bool create(std::string vs_path, std::string fs_path);
		void use();
		void add_uniform(std::string uniform_name);
		void set_uniform_matrix4fv(std::string uniform_name, float * matrix4fv);
		void set_uniform_vector3fv(std::string uniform_name, float * vector3fv);
	private:
		bool create_shader(const char *file_name, GLuint *shader, GLenum type);
		bool create_program(GLuint vert, GLuint frag, GLuint *programme);

		bool parse_file_into_str(const char *file_name, char *shader_str, int max_len);
		bool is_program_valid(GLuint sp);
		void print_program_info_log(GLuint sp);
		void print_shader_info_log(GLuint shader_index);
	};
}

#endif
