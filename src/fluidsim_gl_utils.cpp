#include "fluidsim_gl_utils.h"
#include "stb_image.h"

#include <fstream>

namespace FluidSim
{
	GLuint create_shader_program(string vs_path, string fs_path)
	{
		char *vs = NULL;
		char *fs = NULL;

		vs = (char *)malloc(sizeof(char) * 10000);
		fs = (char *)malloc(sizeof(char) * 10000);
		memset(vs, 0, sizeof(char) * 10000);
		memset(fs, 0, sizeof(char) * 10000);

		FILE *fp;
		char c;
		int count;
		GLuint v;
		GLuint f;
		GLuint p;

		fp = fopen(vs_path.c_str(), "r");
		count = 0;
		while ((c = fgetc(fp)) != EOF)
		{
			vs[count] = c;
			count++;
		}
		fclose(fp);

		fp = fopen(fs_path.c_str(), "r");
		count = 0;
		while ((c = fgetc(fp)) != EOF)
		{
			fs[count] = c;
			count++;
		}
		fclose(fp);

		v = glCreateShader(GL_VERTEX_SHADER);
		f = glCreateShader(GL_FRAGMENT_SHADER);

		const char *vv;
		const char *ff;
		vv = vs;
		ff = fs;

		glShaderSource(v, 1, &vv, NULL);
		glShaderSource(f, 1, &ff, NULL);

		int success;

		glCompileShader(v);
		glGetShaderiv(v, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			char info_log[5000];
			glGetShaderInfoLog(v, 5000, NULL, info_log);
			printf("Error in vertex shader compilation!\n");
			printf("Info Log: %s\n", info_log);
		}

		glCompileShader(f);
		glGetShaderiv(f, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			char info_log[5000];
			glGetShaderInfoLog(f, 5000, NULL, info_log);
			printf("Error in fragment shader compilation!\n");
			printf("Info Log: %s\n", info_log);
		}

		p = glCreateProgram();
		glAttachShader(p, v);
		glAttachShader(p, f);
		glLinkProgram(p);

		free(vs);
		free(fs);
		return p;
	}

	GLuint create_cube_vao(int size)
	{
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

	bool load_cube_map_side(GLuint texture, GLenum side_target,
		const char *file_name) {
		glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

		int x, y, n;
		int force_channels = 4;
		unsigned char *image_data = stbi_load(file_name, &x, &y, &n, force_channels);
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

	GLuint create_cube_map_tex(const char *front, const char *back, const char *top,
		const char *bottom, const char *left, const char *right)
	{
		GLuint tex_cube;
		// generate a cube-map texture to hold all the sides
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &tex_cube);

		// load each image and copy into a side of the cube-map texture
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, front));
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, back));
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, top));
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, bottom));
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, left));
		(load_cube_map_side(tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_X, right));
		// format cube map texture
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		return tex_cube;
	}
}
