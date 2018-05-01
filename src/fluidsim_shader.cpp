#include "fluidsim_shader.h"

#include <iostream>

#define GL_LOG_FILE "gl.log"
#define MAX_SHADER_LENGTH 262144


bool FluidSim::Shader::create(std::string vs_path, std::string fs_path)
{
	if (create_shader(vs_path.c_str(), &vert_shader_, GL_VERTEX_SHADER) &&
		create_shader(fs_path.c_str(), &frag_shader_, GL_FRAGMENT_SHADER) &&
		create_program(vert_shader_, frag_shader_, &shader_program_))
	{
		is_created_ = true;
	}
	return is_created_;
}

void FluidSim::Shader::use()
{
	if (is_created_)
		glUseProgram(shader_program_);
	else
		std::cout << "Error: shader not created" << std::endl;
}

void FluidSim::Shader::add_uniform(std::string uniform_name)
{
	if (is_created_)
		uniform_loc_[uniform_name] = glGetUniformLocation(shader_program_, uniform_name.c_str());
	else
		std::cout << "Error: shader not created" << std::endl;
}


void FluidSim::Shader::set_uniform_matrix4fv(std::string uniform_name, float * matrix4fv)
{
	if (is_created_)
	{
		use();
		auto it = uniform_loc_.find(uniform_name);
		if (it != uniform_loc_.end())
			glUniformMatrix4fv(uniform_loc_[uniform_name], 1, FALSE, matrix4fv);
	}
	else
		std::cout << "Error: shader not created" << std::endl;
}

void FluidSim::Shader::set_uniform_vector3fv(std::string uniform_name, float * vector3fv)
{
	if (is_created_)
	{
		use();
		auto it = uniform_loc_.find(uniform_name);
		if (it != uniform_loc_.end())
			glUniform3fv(uniform_loc_[uniform_name], 1, vector3fv);
	}
	else
		std::cout << "Error: shader not created" << std::endl;
}

bool FluidSim::Shader::create_shader(const char * file_name, GLuint * shader, GLenum type)
{
	gl_log("creating shader from %s...\n", file_name);
	char shader_string[MAX_SHADER_LENGTH];
	parse_file_into_str(file_name, shader_string, MAX_SHADER_LENGTH);
	*shader = glCreateShader(type);
	const GLchar *p = (const GLchar *)shader_string;
	glShaderSource(*shader, 1, &p, NULL);
	glCompileShader(*shader);
	// check for compile errors
	int params = -1;
	glGetShaderiv(*shader, GL_COMPILE_STATUS, &params);
	if (GL_TRUE != params) {
		gl_log_err("ERROR: GL shader index %i did not compile\n", *shader);
		print_shader_info_log(*shader);
		return false; // or exit or something
	}
	gl_log("shader compiled. index %i\n", *shader);
	return true;
}

bool FluidSim::Shader::create_program(GLuint vert, GLuint frag, GLuint * programme)
{
	*programme = glCreateProgram();
	gl_log("created programme %u. attaching shaders %u and %u...\n", *programme,
		vert, frag);
	glAttachShader(*programme, vert);
	glAttachShader(*programme, frag);
	// link the shader programme. if binding input attributes do that before link
	glLinkProgram(*programme);
	GLint params = -1;
	glGetProgramiv(*programme, GL_LINK_STATUS, &params);
	if (GL_TRUE != params) {
		gl_log_err("ERROR: could not link shader programme GL index %u\n",
			*programme);
		print_program_info_log(*programme);
		return false;
	}
	(is_program_valid(*programme));
	// delete shaders here to free memory
	glDeleteShader(vert);
	glDeleteShader(frag);
	return true;
}

bool FluidSim::Shader::parse_file_into_str(const char * file_name, char * shader_str, int max_len)
{
	shader_str[0] = '\0'; // reset string
	FILE *file = fopen(file_name, "r");
	if (!file) {
		gl_log_err("ERROR: opening file for reading: %s\n", file_name);
		return false;
	}
	int current_len = 0;
	char line[2048];
	strcpy(line, ""); // remember to clean up before using for first time!
	while (!feof(file)) {
		if (NULL != fgets(line, 2048, file)) {
			current_len += strlen(line); // +1 for \n at end
			if (current_len >= max_len) {
				gl_log_err("ERROR: shader length is longer than string buffer length %i\n",
					max_len);
			}
			strcat(shader_str, line);
		}
	}
	if (EOF == fclose(file)) { // probably unnecesssary validation
		gl_log_err("ERROR: closing file from reading %s\n", file_name);
		return false;
	}
	return true;
}

bool FluidSim::Shader::is_program_valid(GLuint sp)
{
	glValidateProgram(sp);
	GLint params = -1;
	glGetProgramiv(sp, GL_VALIDATE_STATUS, &params);
	if (GL_TRUE != params) {
		gl_log_err("program %i GL_VALIDATE_STATUS = GL_FALSE\n", sp);
		print_program_info_log(sp);
		return false;
	}
	gl_log("program %i GL_VALIDATE_STATUS = GL_TRUE\n", sp);
	return true;
}

void FluidSim::Shader::print_program_info_log(GLuint sp)
{
	int max_length = 2048;
	int actual_length = 0;
	char log[2048];
	glGetProgramInfoLog(sp, max_length, &actual_length, log);
	printf("program info log for GL index %u:\n%s", sp, log);
	gl_log("program info log for GL index %u:\n%s", sp, log);
}

void FluidSim::Shader::print_shader_info_log(GLuint shader_index)
{
	int max_length = 2048;
	int actual_length = 0;
	char log[2048];
	glGetShaderInfoLog(shader_index, max_length, &actual_length, log);
	printf("shader info log for GL index %i:\n%s\n", shader_index, log);
	gl_log("shader info log for GL index %i:\n%s\n", shader_index, log);
}
