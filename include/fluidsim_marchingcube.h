#ifndef __MARCHINGCUBE_H__
#define __MARCHINGCUBE_H__

#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifndef BUILD_CUDA
#include "fluidsim_types.h"
#else
#include "cuda_runtime.h"
#endif

namespace FluidSim {
	typedef unsigned int uint;

	class MarchingCube
	{
	private:

		uint row_vox;
		uint col_vox;
		uint len_vox;
		uint tot_vox;
		float step;

		float *scalar;
		float3 *normal;
		float3 *pos;
		float3 origin;
		float3 sim_ratio;

		float isovalue;

	public:

		MarchingCube(uint _row_vox, uint _col_vox, uint _len_vox, float *_scalar, float3 *_pos, float3 _sim_ratio, float3 _origin, float _step, float _isovalue);
		void set_scalar_pos(float *_scalar, float3 *_pos);
		~MarchingCube();
		void run();
	};
}

#endif
