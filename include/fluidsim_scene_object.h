#ifndef _FLUIDSIM_SCENEOBJECT_H_
#define _FLUIDSIM_SCENEOBJECT_H_

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <string>
#include <vector>
#include <maths_funcs.h>
#include "fluidsim_shader.h"

namespace FluidSim
{
	class SceneObject
	{
	protected:
		GLuint vao_ = 0;
		GLuint vbo_ = 0;
		GLuint ebo_ = 0;
		GLuint nvbo_ = 0;
		Shader *shader_ = 0;
		vec3 position_;
		mat4 M_;
	public:
		virtual void render() = 0;
		void set_shader(Shader *shader) { shader_ = shader; }
		vec3 get_position() { return position_; };
	};

	class Cube :public SceneObject
	{
	private:
		vec3 side_;
	public:
		Cube(vec3 position, vec3 side);
		vec3 get_side() { return side_; };
		virtual void render();
	};

	class TexturedCube : public Cube
	{
	private:
		GLuint tex_id_;
	public:
		TexturedCube(vec3 position, vec3 side, GLuint tex_id);
		virtual void render();
	};

	class Sphere :public SceneObject
	{
	private:
		float radius_;
		int lats_;
		int longs_;
		int num_vertices_;
	public:
		Sphere(vec3 position, float radius, int lats = 40, int longs = 40);
		float get_radius() { return radius_; };
		virtual void render();
	};

	class Model :public SceneObject
	{
	private:
		float *vertices_;
		float scale_;
		int num_vertices_;
	public:
		Model(std::string path, vec3 position, float scale);
		~Model();
		virtual void render();
	};

}

#endif