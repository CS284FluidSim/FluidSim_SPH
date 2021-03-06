#ifndef _FLUIDSIM_SYSTEM_CUH_
#define _FLUIDSIM_SYSTEM_CUH_

#define PI 3.141592f
#define EPS_F 1E-12f
#define BOUNDARY 0.0001f

#include <cuda_runtime.h>
#include <helper_math.h>

#include <iostream>

#include "gpu/fluidsim_marchingcube.cuh"
#include "fluidsim_scene_object.h"

#include <fstream>
#include <string>

namespace FluidSim {

	namespace gpu {

		class SysParam
		{
		public:
			uint max_particles;
			uint num_particles;

			float h;
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

			float poly6;
			float grad_spiky;
			float lplc_visco;

			float grad_poly6;
			float lplc_poly6;

			float h2;
			float self_lplc_color;

			float3 sim_ratio;
			float3 sim_origin;

			// parameters used for PCISPH
			int minIteration;
			int maxIteration;
			float eta;
			float threshold;
			float delta;
			float maxDensVariance;
		};

		class Object 
		{
		public:
			float3 pos;
		};

		class StaticObject
		{
		public:
			float3 pos;
		};

		class SimulateSystem {
		public:
			enum RenderMode
			{
				SURFACE, DENS, PRESS, FORCE
			};
		public:
			__host__
				SimulateSystem();
			__host__
				SimulateSystem(float3 world_size, float3 sim_ratio, float3 world_origin,
					int max_particles = 500000, float h = 0.04f, float mass = 0.02f, float3 gravity = {0.f,-9.8f,0.f}, float bound_damping=-0.5f,
					float rest_dens = 1000.f, float gas_const=1.f, float visc = 6.5f, float timestep = 0.002f, float surf_norm=3.f, float surf_coef=0.2f);
			__host__
				~SimulateSystem();
			__host__
				void add_cube_fluid(const float3 &pos_min, const float3 &pos_max, const float gap);
			__host__
				void add_fluid(const float3 &cube_pos_min, const float3 &cube_pos_max, float3 velocity = {0.0f, 0.0f, 0.0f}); // add cude fluid
			__host__
				void add_fluid(const float3 &sphere_pos, const float &radius, float3 velocity = { 0.0f, 0.0f, 0.0f });  // add sphere fluid
			__host__
				void add_fluid(const float3 &scale_const); // add object fluid
			__host__
				void add_static_object(Cube *cube, bool isCollider = true);
			__host__
				void add_static_object(Sphere *sphere);
			__host__
				void add_static_object(Model *model);
			__host__
				void start();
			__host__
				void reset()
			{
				sys_param_->num_particles = 0;
			}
			__host__
				virtual void animation();
			__host__
				void render_particles(RenderMode rm=RenderMode::DENS);
			__host__
				void render_surface(MarchingCube::RenderMode rm);
			__host__
				void render_static_object();
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
			__host__
				void change_mass(float mass)
			{
				sys_param_->mass = mass;
			}

		protected:
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

		protected:
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

		protected:
			bool sys_running_;

			MarchingCube *marchingCube_;

			SysParam * sys_param_;
			SysParam * dev_sys_param_;

			Particle * particles_;
			Particle * dev_particles_;

			float3 *normals_;
			float3 *vertexs_;

			int *occupied_;
			int *dev_occupied_;
			float3 *dev_normals_;
			float3 *dev_vertexs_;
			uint * dev_hash_;
			uint * dev_index_;
			uint* dev_start_;
			uint* dev_end_;

			GLuint vao_;
			GLuint p_vbo_;
			GLuint c_vbo_;
			std::vector<float> vec_p_;
			std::vector<float> vec_c_;

			std::vector<SceneObject *> scene_objects;
		};

		__device__ __host__ __forceinline__
			int3 calc_cell_pos(float3 p, float cell_size);

		__device__ __forceinline__
			unsigned calc_cell_hash(int3 cell_pos, uint3 grid_size);

		__device__ __forceinline__
			float calc_cell_density(uint index, int3 neighbor, Particle *dMem, uint *dHash, uint *dIndex, uint *dStart, uint *dEnd, SysParam* dSysParam);

		__device__ __forceinline__
			float3 calc_cell_force(uint index, int3 neighbor, Particle *dMem, uint *dHash, uint *dIndex, uint *dStart, uint *dEnd, float3 &grad_color, float &lplc_color, SysParam* dSysParam);
	}
}

#endif