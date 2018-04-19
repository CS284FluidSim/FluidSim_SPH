#ifndef __FLUIDSIM_MARCHINGCUBE_CUH__
#define __FLUIDSIM_MARCHINGCUBE_CUH__

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_math.h>

#include "gpu/fluidsim_particle.cuh"
#include "gpu/fluidsim_marchingcube_param.cuh"

namespace FluidSim {
	namespace gpu {

		struct Triangle
		{
			float3 v[3];
			float3 n[3];
			float valid = 0.0f;
		};

		struct is_non_empty_tri
		{
			__host__ __device__
				bool operator()(const Triangle &tri)
			{
				return tri.valid != 0.0f;
			}
		};

		class MarchingCube
		{
		private:
			MarchingCubeParam *param_;
			MarchingCubeParam *dev_param_;

			float3 *dev_pos_;
			float *dev_scalar_;
			float3 *dev_normal_;

			//Output Triangle
			Triangle *dev_tri_;
			Triangle *dev_tri_non_empty;
			Triangle *tri_;

		public:
			__host__
				MarchingCube(uint3 dim_vox, float3 sim_ratio, float3 origin, float step, float isovalue);
			__host__
				~MarchingCube();
			__host__
				void init(uint num_particles);
			__host__
				void compute(Particle *dev_particles);
			__host__
				void render();

		private:
			__host__
				uint calc_block_size(uint num_element, uint num_thread)
			{
				return (num_element % num_thread != 0) ? (num_element / num_thread + 1) : (num_element / num_thread);
			}

			__host__
				void calc_grid_size(uint num_particle, uint block_size, uint &num_blocks, uint &num_threads)
			{
				num_threads = min(block_size, num_particle);
				num_blocks = calc_block_size(num_particle, num_threads);
			}
		};
	}
}

#endif
