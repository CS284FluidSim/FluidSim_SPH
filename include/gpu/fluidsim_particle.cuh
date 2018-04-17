#ifndef __FLUIDSIM_PARTICLE_CUH__
#define __FLUIDSIM_PARTICLE_CUH__

#include <cuda_runtime.h>

namespace FluidSim {

	namespace gpu {

		class Particle {
		public:
			float3 pos;
			float3 vel;
			float3 force;

			float dens;
			float pres;
			float surf_norm;
		};
	}
}

#endif