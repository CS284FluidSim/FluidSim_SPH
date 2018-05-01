#ifndef _FLUIDSIM_PCI_SYSTEM_CUH_
#define _FLUIDSIM_PCI_SYSTEM_CUH_

#define PI 3.141592f
#define EPS_F 1E-12f
#define BOUNDARY 0.0001f

#include "gpu/fluidsim_system.cuh"
#include "gpu/fluidsim_pci_particle.cuh"

namespace FluidSim {

	namespace gpu {
		class SimulateSystemPCI : public SimulateSystem
		{
		public:
			__host__
				SimulateSystemPCI(float3 world_size, float3 sim_ratio, float3 world_origin,
					int max_particles = 500000, float h = 0.04f, float mass = 0.02f, float3 gravity = { 0.f,-9.8f,0.f }, float bound_damping = -0.5f,
					float rest_dens = 1000.f, float gas_const = 1.f, float visc = 6.5f, float timestep = 0.002f, float surf_norm = 3.f, float surf_coef = 0.2f);
			__host__
				~SimulateSystemPCI();
			__host__
				void animation();
			__host__
				void add_cube_fluid(const float3 &pos_min, const float3 &pos_max, const float gap);
			__host__
				void start();
			__host__
				ParticlePCI *get_particles() 
			{
				return particlesPCI_;
			}
		private:
			__host__
				void add_particle(const float3 &pos, const float3 &vel);
			__host__
				void build_table();
			__host__
				void comp_dens();
			__host__
				void comp_other_force();
			__host__
				void comp_threshold_delta();
			__host__
				void pred_vel_pos();
			__host__
				void pred_dens_update_pres(float &maxDensVar);
			__host__
				void comp_pres_force();
			__host__
				void integratePCI();

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

		private:
			ParticlePCI * particlesPCI_;
			ParticlePCI * dev_particlesPCI_;

			float* dev_maxDensVariance;

		};

		__device__ __host__ __forceinline__
			int3 calc_cell_pos(float3 p, float cell_size);

		__device__ __forceinline__
			unsigned calc_cell_hash(int3 cell_pos, uint3 grid_size);

		__device__ __forceinline__
			float calc_cell_density(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param);

		__device__ __forceinline__
			float3 calc_cell_force(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, float3 &grad_color, float &lplc_color, SysParam* dev_sys_param);

		__device__ __forceinline__
			float calc_cell_pred_density(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param);

		__device__ __forceinline__
			float3 calc_cell_pres_force(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param);


	}
}

#endif