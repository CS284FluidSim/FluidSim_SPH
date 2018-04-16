#ifndef _FLUIDSIM_SYSTEM_GPU_CUH_
#define _FLUIDSIM_SYSTEM_GPU_CUH_

#define PI 3.141592f
#define INF 1E-12f
#define BOUNDARY 0.0001f

#include <cuda_runtime.h>
#include <helper_math.h>
#include <iostream>

namespace FluidSim {

	class SysParam
	{
	public:
		uint max_particles;
		uint num_particles;

		float kernel;
		float mass;

		float3 world_size;
		float cell_size;
		uint3 grid_size;
		uint total_cells;

		float3 gravity;
		float rest_dens;
		float gas_const;
		float visc;
		float timestep;
		float bound_damping;

		float surf_norm;
		float surf_coef;

		float poly6_value;
		float spiky_value;
		float visco_value;

		float grad_poly6;
		float lplc_poly6;

		float kernel2;
		float self_dens;
		float self_lplc_color;

		float3 sim_ratio;
		float3 sim_origin;
	};

	class Particle {
	public:
		float3 pos;
		float3 vel;
		float3 acc;
		float3 ev;

		float dens;
		float pres;
		float surf_norm;
	};

	class SimulateSystem {
	public:
		__host__
			SimulateSystem(float world_size_x, float world_size_y, float world_size_z);
		__host__
			~SimulateSystem();
		__host__
			void add_cube_fluid(const float3 &pos_min, const float3 &pos_max);
		__host__
			void start();
		__host__
			void animation();
		__host__
			bool is_running() {
			return sys_running_;
		}

		__host__
			SysParam *get_sys_pararm() {
			return sys_param_;
		}

		__host__
			int get_num_particles() {
			return sys_param_->num_particles;
		}

		__host__
			Particle *get_particles() {
			return particles_;
		}

	private:
		__host__
			void add_particle(const float3 &pos, const float3 &vel);
		__host__
			void build_table();
		__host__
			void comp_dens_pres();
		__host__
			void comp_force();
		__host__
			void integrate();

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
		bool sys_running_;

		SysParam * sys_param_;
		SysParam * dev_sys_param_;

		Particle * particles_;
		Particle * dev_particles_;

		uint * dev_hash_;
		uint * dev_index_;
		uint* dev_start_;
		uint* dev_end_;
	};

	__device__ __forceinline__
		int3 calc_cell_pos(float3 p, float cell_size);

	__device__ __forceinline__
		unsigned calc_cell_hash(int3 cell_pos, uint3 grid_size);

	__device__ __forceinline__
		float calc_cell_density(uint index, int3 neighbor, Particle *dMem, uint *dHash, uint *dIndex, uint *dStart, uint *dEnd, SysParam* dSysParam);

	__device__ __forceinline__
		float3 calc_cell_force(uint index, int3 neighbor, Particle *dMem, uint *dHash, uint *dIndex, uint *dStart, uint *dEnd, float3 &grad_color, float &lplc_color, SysParam* dSysParam);

}

#endif