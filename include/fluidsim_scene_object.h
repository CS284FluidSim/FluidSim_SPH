#ifndef _FLUIDSIM_SCENEOBJECT_H_
#define _FLUIDSIM_SCENEOBJECT_H_

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace FluidSim
{
	class SceneObject
	{
	protected:
		GLuint vao_ = 0;
		GLuint vbo_ = 0;
		GLuint ebo_ = 0;
		GLuint nvbo_ = 0;
		GLuint shader_ = 0;
		float3 position_ = {0.f,0.f,0.f};
	public:
		virtual void render() = 0;
		void set_shader(GLuint shader) {
			shader_ = shader;
		}
		float3 get_position()
		{
			return position_;
		}
	};

	class Cube :public SceneObject
	{
	private:
		float3 side_;
	public:
		Cube(float3 position, float3 side);
		float3 get_position() { return position_; };
		float3 get_side() { return side_; };
		virtual void render();
	};

	class TexturedCube : public Cube
	{
	private:
		GLuint tex_id;
	public:
		TexturedCube(float3 position, float3 side, GLuint tex_id);
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
		Sphere(float3 position, float radius, int lats = 40, int longs = 40);
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
		Model(std::string path, float3 position, float scale);
		~Model();
		virtual void render();
	};
}

#endif