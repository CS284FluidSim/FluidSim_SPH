#ifndef _FLUIDSIM_SYSTEM_H_
#define _FLUIDSIM_SYSTEM_H_

#include <vector>

#include <eigen3\Eigen\Dense>
using namespace Eigen;

namespace FluidSimPCI {

	class Particle {
	public:
		unsigned int id;
		Vector3f curr_pos;
		Vector3f new_pos;
		Vector3f curr_vel;
		Vector3f new_vel;
		Vector3f force;
		Vector3f pres_force;
		Vector3f damped_vel;
		Vector3f normal;

		float dens;
		float mass;
		float pres;
		float surf_norm;
		
		Particle *next;
	};

	class SimulateSystem {
	public:
		SimulateSystem();
		~SimulateSystem();
		void add_cube_fluid(const Vector3f &pos_min, const Vector3f &pos_max);
		void add_boundary();
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
		int get_num_boundary_particles() {
			return num_boundary_particles_;
		}
		Particle *get_boundary_particles() {
			return boundary_particles_;
		}

	private:
		void add_particle(const Vector3f &pos, const Vector3f &vel);
		void add_boundary_particle(const Vector3f &pos, const Vector3f &normal);
		void build_table();
		void add_boundary_particles();
		void build_boundary_table();
		void massify_boundary();

		void init_densities();
		void init_normals();
		void comp_force();
		void init_pres_pres_force();
		void pred_vel_pos();
		void pred_dens_dens_var_update_pres();
		void update_dens_var_scale();
		void comp_pres_force();

		void integrate();

		Vector3i calc_cell_pos(Vector3f p);
		unsigned calc_cell_hash(Vector3i cell_pos);

	private:
		bool sys_running;
		int max_particles_;
		int num_particles_;
		int num_boundary_particles_;

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
		float mass2_;
		float radius_;
		float visc_;
		float timestep_;
		float bound_damping_;
		float max_dens_var_;
		float avg_dens_var_;
		float dens_var_scale_;
		float dens_var_limit_;
		float min_iteration_;
		float max_iteration_;

		float surf_coef_;

		float poly6_value_;
		float spiky_value_;
		float visco_value_;
		float surf_value_;
		float surf_offset_;
		float grad_poly6_;
		float lplc_poly6_;

		float self_dens_;
		float self_lplc_color_;

		Particle * particles_;
		Particle ** cells_;
		Particle * boundary_particles_;
		Particle ** boundary_cells_;
	};
}

#endif