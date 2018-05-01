#ifndef __FLUIDSIM_PCI_PARTICLE_CUH__
#define __FLUIDSIM_PCI_PARTICLE_CUH__

#include <cuda_runtime.h>

namespace FluidSim {

	namespace gpu {

		class ParticlePCI {
		public:
			float3 pos;
			float3 vel;
			float3 force;
			float3 force_pres;

			float dens;
			float pres;
			float surf_norm;

			float3 pred_pos;
			float3 pred_vel;
		};
	}
}

#endif