#ifndef _FLUIDSIM_SYSTEM_GPU_CUH_
#define _FLUIDSIM_SYSTEM_GPU_CUH_

#include <cuda_runtime.h>

namespace FluidSim {

	class Particle {
	public:
		unsigned int id;
		float3 pos;
		float3 vel;
		float3 acc;
		float3 ev;

		float dens;
		float pres;
		float surf_norm;
		
		Particle *next;
	};

	class SimulateSystem {
	public:
		__host__
		SimulateSystem();
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
			return sys_running;
		}
		__host__
		float3 get_world_size() {
			return world_size_;
		}
		__host__
		int get_num_particles() {
			return num_particles_;
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
		
		__host__ __device__
		int3 calc_cell_pos(float3 p);

		__host__ __device__
		unsigned calc_cell_hash(int3 cell_pos);

	private:
		bool sys_running;
		int max_particles_;
		int num_particles_;

		float3 world_size_;
		int3 grid_size_;
		float cell_size_;
		int total_cells_;

		float3 gravity_;
		float rest_dens_;
		float gas_const_;
		float kernel_;
		float kernel2_;
		float mass_;
		float visc_;
		float timestep_;
		float bound_damping_;

		float surf_norm_;
		float surf_coef_;

		float poly6_value_;
		float spiky_value_;
		float visco_value_;
		float grad_poly6_;
		float lplc_poly6_;

		float self_dens_;
		float self_lplc_color_;

		Particle * particles_;
		Particle ** cells_;
	};
}

#endif