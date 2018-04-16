#ifndef _FLUIDSIM_TYPES_H_
#define _FLUIDSIM_TYPES_H_

namespace FluidSim
{
#ifndef BUILD_CUDA
	struct float3
	{
		float x;
		float y;
		float z;
	};
}
#endif

#endif