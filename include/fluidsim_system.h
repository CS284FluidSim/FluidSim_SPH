#ifndef _FLUIDSIM_SYSTEM_H_
#define _FLUIDSIM_SYSTEM_H_

#include <vector>

#include <Eigen/Dense>
using namespace Eigen;

namespace FluidSim {

	class Particle {
	public:
		unsigned int id;

		float radius;

		Vector3f pos;
		Vector3f vel;
		Vector3f acc;
		Vector3f ev;
		Vector3f pres_force;

		float dens;
		float pres;
		float surf_norm;

		Vector3f pred_vel;
		Vector3f pred_pos;
		
		Particle *next;
	};

	class SimulateSystem {
	public:
		SimulateSystem();
		~SimulateSystem();
		void add_cube_fluid(const Vector3f &pos_min, const Vector3f &pos_max);
		void start();
		void animation();
		bool is_running() {
			return sys_running;
		}
		Vector3f get_world_size() {
			return world_size_;
		}
		int get_num_particles() {
			return num_particles_;
		}
		Particle *get_particles() {
			return particles_;
		}

	private:
		void add_particle(const Vector3f &pos, const Vector3f &vel);
		void build_table();
		void comp_dens_pres();
		void comp_force();
		void integrate();

		void init_dens();
		void init_force();
		void pred_vel_pos();
		void update_dens_var_scale();
		void predDensVar_updatePres();
		void update_presForce();

		Vector3i calc_cell_pos(Vector3f p);
		unsigned calc_cell_hash(Vector3i cell_pos);

	private:
		bool sys_running;
		int max_particles_;
		int num_particles_;
		int minIteration;
		int maxIteration;

		Vector3f world_size_;
		Vector3i grid_size_;
		float cell_size_;
		int total_cells_;

		Vector3f gravity_;
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

		float particle_radius;
		float delta;
		float aveDensityVariance;
		float maxDensityVariance;
		float densityVarianceThreshold;

		Particle * particles_;
		Particle ** cells_;
	};
}

#endif