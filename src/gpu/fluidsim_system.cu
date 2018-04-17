#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cmath>

#include "gpu/fluidsim_system.cuh"

namespace FluidSim {

	namespace gpu {
		__global__
			void calc_hash(uint *dev_hash, uint *dev_index, Particle *dev_particles, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 grid_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);
			uint hash = calc_cell_hash(grid_pos, dev_sys_param->grid_size);

			dev_hash[index] = hash;
			dev_index[index] = index;
		}

		__global__
			void find_start_end_kernel(uint *dev_start, uint *dev_end, uint *dev_hash, uint *dev_index, uint num_particle)
		{
			extern __shared__ uint shared_hash[];
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			uint hash;

			if (index < num_particle)
			{
				hash = dev_hash[index];
				shared_hash[threadIdx.x + 1] = hash;

				if (index > 0 && threadIdx.x == 0)
				{
					shared_hash[0] = dev_hash[index - 1];
				}
			}

			__syncthreads();

			if (index < num_particle)
			{
				if (index == 0 || hash != shared_hash[threadIdx.x])
				{
					dev_start[hash] = index;

					if (index > 0)
					{
						dev_end[shared_hash[threadIdx.x]] = index;
					}
				}

				if (index == num_particle - 1)
				{
					dev_end[hash] = index + 1;
				}
			}
		}

		__global__
			void calc_dens_pres_kernel(Particle *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam *dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 cell_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);

			float total_density = 0;

			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbor_pos = cell_pos + make_int3(x, y, z);
						total_density = total_density + calc_cell_density(index, neighbor_pos, dev_particles, dev_hash, dev_index, dev_start, dev_end, dev_sys_param);
					}
				}
			}

			total_density = total_density + dev_sys_param->self_dens;
			dev_particles[index].dens = total_density;

			if (total_density < INF)
			{
				dev_particles[index].dens = dev_sys_param->rest_dens;
			}

			dev_particles[index].pres = (pow(dev_particles[index].dens / dev_sys_param->rest_dens, 7) - 1) * dev_sys_param->gas_const;
		}

		__global__
			void calc_force_kernel(Particle *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 cell_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);

			float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
			float3 grad_color = make_float3(0.0f);
			float lplc_color = 0.0f;

			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbor_pos = cell_pos + make_int3(x, y, z);
						total_force = total_force + calc_cell_force(index, neighbor_pos, dev_particles, dev_hash, dev_index, dev_start, dev_end, grad_color, lplc_color, dev_sys_param);
					}
				}
			}
			dev_particles[index].force = total_force;

			lplc_color += dev_sys_param->self_lplc_color / dev_particles[index].dens;
			dev_particles[index].surf_norm = sqrt(grad_color.x*grad_color.x + grad_color.y*grad_color.y + grad_color.z*grad_color.z);
			float3 force;

			if (dev_particles[index].surf_norm > dev_sys_param->surf_norm)
			{
				force = dev_sys_param->surf_coef * lplc_color * grad_color / dev_particles[index].surf_norm;
			}
			else
			{
				force = make_float3(0.0f);
			}

			dev_particles[index].force += force;
		}

		__global__
			void integrate_kernel(Particle* dev_particles, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			Particle *p = &(dev_particles[index]);

			p->vel = p->vel + p->force*dev_sys_param->timestep / p->dens + dev_sys_param->gravity*dev_sys_param->timestep;
			p->pos = p->pos + p->vel*dev_sys_param->timestep;

			if (p->pos.x >= dev_sys_param->world_size.x - BOUNDARY)
			{
				p->vel.x = p->vel.x*dev_sys_param->bound_damping;
				p->pos.x = dev_sys_param->world_size.x - BOUNDARY;
			}

			if (p->pos.x < 0.0f)
			{
				p->vel.x = p->vel.x*dev_sys_param->bound_damping;
				p->pos.x = 0.0f;
			}

			if (p->pos.y >= dev_sys_param->world_size.y - BOUNDARY)
			{
				p->vel.y = p->vel.y*dev_sys_param->bound_damping;
				p->pos.y = dev_sys_param->world_size.y - BOUNDARY;
			}

			if (p->pos.y < 0.0f)
			{
				p->vel.y = p->vel.y*dev_sys_param->bound_damping;
				p->pos.y = 0.0f;
			}

			if (p->pos.z >= dev_sys_param->world_size.z - BOUNDARY)
			{
				p->vel.z = p->vel.z*dev_sys_param->bound_damping;
				p->pos.z = dev_sys_param->world_size.z - BOUNDARY;
			}

			if (p->pos.z < 0.0f)
			{
				p->vel.z = p->vel.z*dev_sys_param->bound_damping;
				p->pos.z = 0.0f;
			}

		}

		__host__
			SimulateSystem::SimulateSystem(float world_size_x, float world_size_y, float world_size_z)
		{
			sys_running_ = false;
			sys_param_ = new SysParam();
			sys_param_->num_particles = 0;

			sys_param_->max_particles = 500000;
			sys_param_->kernel = 0.04f;
			sys_param_->mass = 0.02f;

			sys_param_->world_size.x = world_size_x;
			sys_param_->world_size.y = world_size_y;
			sys_param_->world_size.z = world_size_z;
			sys_param_->cell_size = sys_param_->kernel;
			sys_param_->grid_size.x = (int)ceil(sys_param_->world_size.x / sys_param_->cell_size);
			sys_param_->grid_size.y = (int)ceil(sys_param_->world_size.y / sys_param_->cell_size);
			sys_param_->grid_size.z = (int)ceil(sys_param_->world_size.z / sys_param_->cell_size);
			sys_param_->total_cells = sys_param_->grid_size.x*sys_param_->grid_size.y*sys_param_->grid_size.z;

			sys_param_->gravity.x = 0.f;
			sys_param_->gravity.y = -9.8f;
			sys_param_->gravity.z = 0.0f;
			sys_param_->bound_damping = -0.5f;
			sys_param_->rest_dens = 1000.f;
			sys_param_->gas_const = 1.0f;
			sys_param_->visc = 6.5f;
			sys_param_->timestep = 0.002f;
			sys_param_->surf_norm = 6.0f;
			sys_param_->surf_coef = 0.1f;

			sys_param_->poly6_value = 315.0f / (64.0f * PI * pow(sys_param_->kernel, 9));
			sys_param_->spiky_value = -45.0f / (PI * pow(sys_param_->kernel, 6));
			sys_param_->visco_value = 45.0f / (PI * pow(sys_param_->kernel, 6));

			sys_param_->grad_poly6 = -945 / (32 * PI * pow(sys_param_->kernel, 9));
			sys_param_->lplc_poly6 = -945 / (8 * PI * pow(sys_param_->kernel, 9));

			sys_param_->kernel2 = sys_param_->kernel*sys_param_->kernel;
			sys_param_->self_dens = sys_param_->mass*sys_param_->poly6_value*pow(sys_param_->kernel, 6);
			sys_param_->self_lplc_color = sys_param_->lplc_poly6*sys_param_->mass*sys_param_->kernel2*(0 - 3 / 4 * sys_param_->kernel2);

			cudaMalloc(&dev_sys_param_, sizeof(SysParam));

			particles_ = (Particle *)malloc(sizeof(Particle)*sys_param_->max_particles);
			cudaMalloc(&dev_particles_, sys_param_->max_particles * sizeof(Particle));
			cudaMalloc(&dev_hash_, sizeof(uint)*sys_param_->max_particles);
			cudaMalloc(&dev_index_, sizeof(uint)*sys_param_->max_particles);
			cudaMalloc(&dev_start_, sizeof(uint)*sys_param_->total_cells);
			cudaMalloc(&dev_end_, sizeof(uint)*sys_param_->total_cells);
		}

		__host__
			SimulateSystem::~SimulateSystem() {
			free(particles_);
			cudaFree(dev_sys_param_);
			cudaFree(dev_particles_);
			cudaFree(dev_hash_);
			cudaFree(dev_index_);
			cudaFree(dev_start_);
			cudaFree(dev_end_);
		}

		__host__
			void SimulateSystem::start() {
			sys_running_ = true;
			cudaMemcpy(dev_particles_, particles_, sys_param_->num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_sys_param_, sys_param_, sizeof(SysParam), cudaMemcpyHostToDevice);
			std::cout << "Started simulation with " << sys_param_->num_particles << " particles" << std::endl;
		}

		__host__
			void SimulateSystem::add_cube_fluid(const float3 &pos_min, const float3 &pos_max)
		{
			float3 pos;
			float3 vel;

			vel.x = 0.f;
			vel.y = 0.f;
			vel.z = 0.f;

			for (pos.x = sys_param_->world_size.x*pos_min.x; pos.x < sys_param_->world_size.x*pos_max.x; pos.x += (sys_param_->kernel*0.5f))
			{
				for (pos.y = sys_param_->world_size.y*pos_min.y; pos.y < sys_param_->world_size.y*pos_max.y; pos.y += (sys_param_->kernel*0.5f))
				{
					for (pos.z = sys_param_->world_size.z*pos_min.z; pos.z < sys_param_->world_size.z*pos_max.z; pos.z += (sys_param_->kernel*0.5f))
					{
						add_particle(pos, vel);
					}
				}
			}
		}

		__host__
			void SimulateSystem::add_particle(const float3 &pos, const float3 &vel)
		{
			Particle *p = &(particles_[sys_param_->num_particles]);

			p->pos = pos;
			p->vel = vel;

			p->force.x = 0.0f;
			p->force.x = 0.0f;
			p->force.x = 0.0f;

			p->dens = sys_param_->rest_dens;
			p->pres = 0.0f;

			sys_param_->num_particles++;
		}

		__host__
			void SimulateSystem::animation()
		{
			if (sys_running_ == 0)
			{
				return;
			}

			build_table();
			comp_dens_pres();
			comp_force();
			integrate();
			cudaMemcpy(particles_, dev_particles_, sizeof(Particle)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
		}

		__host__
			void SimulateSystem::build_table()
		{
			if (sys_param_->num_particles == 0)
				return;
			uint num_threads;
			uint num_blocks;

			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);
			calc_hash << < num_blocks, num_threads >> > (dev_hash_, dev_index_, dev_particles_, dev_sys_param_);


			thrust::sort_by_key(thrust::device_ptr<uint>(dev_hash_),
				thrust::device_ptr<uint>(dev_hash_ + sys_param_->num_particles),
				thrust::device_ptr<uint>(dev_index_));


			cudaMemset(dev_start_, 0xffffffff, sys_param_->total_cells * sizeof(uint));
			cudaMemset(dev_end_, 0x0, sys_param_->total_cells * sizeof(uint));

			uint smemSize = sizeof(int)*(num_threads + 1);

			find_start_end_kernel << < num_blocks, num_threads, smemSize >> > (dev_start_, dev_end_, dev_hash_, dev_index_, sys_param_->num_particles);
		}

		__host__
			void SimulateSystem::comp_dens_pres()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			calc_dens_pres_kernel << < num_blocks, num_threads >> > (dev_particles_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);
		}

		__host__
			void SimulateSystem::comp_force()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);
			calc_force_kernel << < num_blocks, num_threads >> > (dev_particles_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);
		}

		__host__
			void SimulateSystem::integrate()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			integrate_kernel << < num_blocks, num_threads >> > (dev_particles_, dev_sys_param_);
		}

		__device__ __forceinline__
			int3 calc_cell_pos(float3 p, float cell_size)
		{
			int3 cell_pos;
			cell_pos.x = int(floor((p.x) / cell_size));
			cell_pos.y = int(floor((p.y) / cell_size));
			cell_pos.z = int(floor((p.z) / cell_size));

			return cell_pos;
		}

		__device__ __forceinline__
			unsigned int calc_cell_hash(int3 cell_pos, uint3 grid_size)
		{
			if (cell_pos.x < 0 || cell_pos.x >= (int)grid_size.x
				|| cell_pos.y < 0 || cell_pos.y >= (int)grid_size.y
				|| cell_pos.z < 0 || cell_pos.z >= (int)grid_size.z)
			{
				return (unsigned int)0xffffffff;
			}

			cell_pos.x = cell_pos.x & (grid_size.x - 1);
			cell_pos.y = cell_pos.y & (grid_size.y - 1);
			cell_pos.z = cell_pos.z & (grid_size.z - 1);

			return ((unsigned int)(cell_pos.z))*grid_size.y*grid_size.x
				+ ((unsigned int)(cell_pos.y))*grid_size.x
				+ (unsigned int)(cell_pos.x);
		}

		__device__ __forceinline__
			float calc_cell_density(uint index, int3 neighbor, Particle *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			float total_cell_density = 0.0f;
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);
			if (grid_hash == 0xffffffff)
			{
				return total_cell_density;
			}
			uint start_index = dev_start[grid_hash];

			float mass = dev_sys_param->mass;
			float kernel2 = dev_sys_param->kernel2;
			float poly6_value = dev_sys_param->poly6_value;

			float3 rel_pos;
			float r2;

			Particle *p = &(dev_particles[index]);
			Particle *np;
			uint neighbor_index;

			if (start_index != 0xffffffff)
			{
				uint end_index = dev_end[grid_hash];

				for (uint count_index = start_index; count_index < end_index; count_index++)
				{
					neighbor_index = dev_index[count_index];
					np = &(dev_particles[neighbor_index]);

					rel_pos = np->pos - p->pos;
					r2 = rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z;

					if (r2 < INF || r2 >= kernel2)
					{
						continue;
					}

					total_cell_density = total_cell_density + mass * poly6_value * pow(kernel2 - r2, 3);
				}
			}

			return total_cell_density;
		}

		__device__ __forceinline__
			float3 calc_cell_force(uint index, int3 neighbor, Particle *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, float3 &grad_color, float &lplc_color, SysParam* dev_sys_param)
		{
			float3 total_cell_force = make_float3(0.0f);
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);

			if (grid_hash == 0xffffffff)
			{
				return total_cell_force;
			}

			uint start_index = dev_start[grid_hash];

			float kernel = dev_sys_param->kernel;
			float mass = dev_sys_param->mass;
			float kernel2 = dev_sys_param->kernel2;

			uint neighbor_index;

			Particle *p = &(dev_particles[index]);
			Particle *np;

			float3 rel_pos;
			float r2;
			float r;

			float V;
			float kernel_r;

			float pressure_kernel;
			float temp_force;

			float3 rel_vel;
			float viscosity_kernel;

			if (start_index != 0xffffffff)
			{
				uint end_index = dev_end[grid_hash];

				for (uint count_index = start_index; count_index < end_index; count_index++)
				{
					neighbor_index = dev_index[count_index];

					np = &(dev_particles[neighbor_index]);

					rel_pos = p->pos - np->pos;
					r2 = rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z;

					if (r2 < kernel2 && r2 > INF)
					{
						r = sqrt(r2);
						V = mass / np->dens / 2;
						kernel_r = kernel - r;

						pressure_kernel = dev_sys_param->spiky_value * kernel_r * kernel_r;
						temp_force = V * (p->pres + np->pres) * pressure_kernel;
						total_cell_force = total_cell_force - rel_pos*temp_force / r;

						rel_vel = np->vel - p->vel;
						viscosity_kernel = dev_sys_param->visco_value*(kernel - r);
						temp_force = V * dev_sys_param->visc * viscosity_kernel;
						total_cell_force = total_cell_force + rel_vel*temp_force;

						float temp = (-1) * dev_sys_param->grad_poly6 * V * pow(kernel2 - r2, 2);
						grad_color += temp * rel_pos;
						lplc_color += dev_sys_param->lplc_poly6 * V * (kernel2 - r2) * (r2 - 3 / 4 * (kernel2 - r2));
					}
				}
			}

			return total_cell_force;
		}
	}
}