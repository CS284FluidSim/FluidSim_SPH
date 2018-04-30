#include "fluidsim_scene_object.h"

namespace FluidSim
{
	Cube::Cube(float3 position, float3 side)
	{
		position_ = position;
		side_ = side;

		float points[] = {
			-1.f, 1.f,	-1.f, -1.f, -1.f, -1.f, 1.f,	-1.f, -1.f,
			1.f,	-1.f, -1.f, 1.f,	1.f,	-1.f, -1.f, 1.f,	-1.f,

			-1.f, -1.f, 1.f,	-1.f, -1.f, -1.f, -1.f, 1.f,	-1.f,
			-1.f, 1.f,	-1.f, -1.f, 1.f,	1.f,	-1.f, -1.f, 1.f,

			1.f,	-1.f, -1.f, 1.f,	-1.f, 1.f,	1.f,	1.f,	1.f,
			1.f,	1.f,	1.f,	1.f,	1.f,	-1.f, 1.f,	-1.f, -1.f,

			-1.f, -1.f, 1.f,	-1.f, 1.f,	1.f,	1.f,	1.f,	1.f,
			1.f,	1.f,	1.f,	1.f,	-1.f, 1.f,	-1.f, -1.f, 1.f,

			-1.f, 1.f,	-1.f, 1.f,	1.f,	-1.f, 1.f,	1.f,	1.f,
			1.f,	1.f,	1.f,	-1.f, 1.f,	1.f,	-1.f, 1.f,	-1.f,

			-1.f, -1.f, -1.f, -1.f, -1.f, 1.f,	1.f,	-1.f, -1.f,
			1.f,	-1.f, -1.f, -1.f, -1.f, 1.f,	1.f,	-1.f, 1.f
		};

		GLuint vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 3 * 36 * sizeof(GLfloat), &points,
			GL_STATIC_DRAW);

		glGenVertexArrays(1, &vao_);
		glBindVertexArray(vao_);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Cube::render()
	{
		glPushMatrix();
		glTranslatef(position_.x, position_.y, position_.z);
		glScalef(side_.x/2.f, side_.y/2.f, side_.z/2.f);
		glBindVertexArray(vao_);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		glPopMatrix();
	}

	Sphere::Sphere(float3 position, float radius)
	{
		position_ = position;
		radius_ = radius;
	}

	void Sphere::render()
	{
		glPushMatrix();
		glTranslatef(position_.x, position_.y, position_.z);
		glutSolidSphere(radius_, 40, 40);
		glPopMatrix();
	}
}