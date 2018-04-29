#include "fluidsim_scene_object.h"

namespace FluidSim
{
	Cube::Cube(float3 position, float3 side)
	{
		position_ = position;
	}

	void Cube::render()
	{
		glDepthMask(GL_FALSE);
		glUseProgram(shader_);
		glBindVertexArray(vao_);
		glDrawArrays(GL_TRIANGLES, 0, 36);
		glDepthMask(GL_TRUE);
		glUseProgram(0);
	}
}