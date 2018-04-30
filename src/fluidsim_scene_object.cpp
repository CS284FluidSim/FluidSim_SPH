#include "fluidsim_scene_object.h"

#define M_PI 3.14159265358979323846

#include <cmath>

#include <fstream>
#include <iostream>

using namespace std;

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
		glBufferData(GL_ARRAY_BUFFER, 3 * 36 * sizeof(GLfloat), &points, GL_STATIC_DRAW);

		glGenVertexArrays(1, &vao_);
		glBindVertexArray(vao_);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Cube::render()
	{
		//glPushMatrix();
		//glTranslatef(position_.x, position_.y, position_.z);
		//glScalef(side_.x/2.f, side_.y/2.f, side_.z/2.f);
		glBindVertexArray(vao_);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		//glPopMatrix();
	}

	Sphere::Sphere(float3 position, float radius, int lats, int longs)
	{
		position_ = position;
		radius_ = radius;
		lats_ = lats;
		longs_ = longs;

		int i, j;
		std::vector<GLfloat> vertices;
		std::vector<GLfloat> normals;
		std::vector<GLuint> indices;
		int indicator = 0;
		for (i = 0; i <= lats; i++) {
			double lat0 = M_PI * (-0.5 + (double)(i - 1) / lats);
			double z0 = sin(lat0);
			double zr0 = cos(lat0);

			double lat1 = M_PI * (-0.5 + (double)i / lats);
			double z1 = sin(lat1);
			double zr1 = cos(lat1);

			for (j = 0; j <= longs; j++) {
				double lng = 2 * M_PI * (double)(j - 1) / longs;
				double x = cos(lng);
				double y = sin(lng);

				vertices.push_back(x * zr0);
				vertices.push_back(y * zr0);
				vertices.push_back(z0);
				normals.push_back(x * zr0);
				normals.push_back(y * zr0);
				normals.push_back(z0);
				indices.push_back(indicator);
				indicator++;

				vertices.push_back(x * zr1);
				vertices.push_back(y * zr1);
				vertices.push_back(z1);
				normals.push_back(x * zr1);
				normals.push_back(y * zr1);
				normals.push_back(z1);
				indices.push_back(indicator);
				indicator++;
			}
			indices.push_back(GL_PRIMITIVE_RESTART_FIXED_INDEX);
		}

		glGenVertexArrays(1, &vao_);
		glBindVertexArray(vao_);

		glGenBuffers(1, &vbo_);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenBuffers(1, &ebo_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, nvbo_);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), &normals[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		num_vertices_ = indices.size();
	}

	void Sphere::render()
	{
		glPushMatrix();
		glTranslatef(position_.x, position_.y, position_.z);
		glScalef(radius_, radius_, radius_);
		glBindVertexArray(vao_);
		glEnable(GL_PRIMITIVE_RESTART);
		glPrimitiveRestartIndex(GL_PRIMITIVE_RESTART_FIXED_INDEX);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
		glDrawElements(GL_QUAD_STRIP, num_vertices_, GL_UNSIGNED_INT, NULL);
		glPopMatrix();
	}

	Model::Model(std::string path, float3 position, float scale)
	{
		position_ = position;
		scale_ = scale;

		vector<float3> vec_points;
		// read object point cloud coordinate
		std::string line;
		std::ifstream myfile(path);
		if (myfile.is_open())
		{
			getline(myfile, line);

			std::string delimiter = " ";
			float coord;
			float3 pos;

			while (getline(myfile, line))
			{
				std::vector<float> tmp;
				size_t Pos = 0;
				while ((Pos = line.find(delimiter)) != std::string::npos)
				{
					coord = stof(line.substr(0, Pos));
					tmp.push_back(coord);
					line.erase(0, Pos + delimiter.length());
				}
				if (line.length() != 0)
				{
					coord = stof(line.substr(0, line.length()));
					tmp.push_back(coord);
				}
				pos.x = tmp[0];
				pos.y = tmp[1];
				pos.z = tmp[2];
				
				vec_points.push_back(pos);
			}
			myfile.close();
			num_vertices_ = vec_points.size();
			vertices_ = new float[3 * vec_points.size()];
			for (int i = 0; i < vec_points.size(); ++i)
			{
				vertices_[3 * i] = vec_points[i].x;
				vertices_[3 * i + 1] = vec_points[i].y;
				vertices_[3 * i + 2] = vec_points[i].z;
			}

			GLuint vbo;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices_ * sizeof(GLfloat), &vertices_, GL_STATIC_DRAW);

			glGenVertexArrays(1, &vao_);
			glBindVertexArray(vao_);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		}
		else
		{
			std::cout << "Object file not found" << std::endl;
		}
	}
	Model::~Model()
	{
		delete[] vertices_;
	}
	void Model::render()
	{
		glPushMatrix();
		glTranslatef(position_.x, position_.y, position_.z);
		glTranslatef(scale_, scale_, scale_);
		glBindVertexArray(vao_);
		glDrawArrays(GL_POINTS, 0, num_vertices_);
		glPopMatrix();
	}
}