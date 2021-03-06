#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cmath>

#include "gpu/fluidsim_pci_system.cuh"

#include <iostream>
#include <fstream>
#include <string>

namespace FluidSim{
	
	namespace gpu{
		
		__global__
			void find_start_end_kernel_pci(uint *dev_start, uint *dev_end, uint *dev_hash, uint *dev_index, uint num_particle)
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
			void calc_hash(uint *dev_hash, uint *dev_index, ParticlePCI *dev_particles, SysParam* dev_sys_param)
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
			void calc_dens_kernel(ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam *dev_sys_param)
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

			dev_particles[index].dens = total_density;

			if (total_density < EPS_F)
			{
				dev_particles[index].dens = dev_sys_param->rest_dens;
			}

			//dev_particles[index].pres = (pow(dev_particles[index].dens / dev_sys_param->rest_dens, 7) - 1) * dev_sys_param->gas_const;
			dev_particles[index].pres = 0.0f;
			dev_particles[index].pred_pos = { 0.0f, 0.0f, 0.0f };
			dev_particles[index].pred_vel = { 0.0f, 0.0f, 0.0f };
		}

		__global__
			void calc_force_kernel(ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index == 0)
			{
				dev_sys_param->threshold = dev_sys_param->eta * dev_sys_param->rest_dens;

				// calculate delta
				float3 gradientKernelSum = { 0.0f, 0.0f, 0.0f };
				float squaredGradientKernelSum = 0.f;
				for (float i = -dev_sys_param->h; i <= dev_sys_param->h; i += 0.7f * dev_sys_param->h)    // The distance 0.5f * sys_param_->h can be changed!
				{
					for (float j = -dev_sys_param->h; j <= dev_sys_param->h; j += 0.7f * dev_sys_param->h)
					{
						for (float k = -dev_sys_param->h; k <= dev_sys_param->h; k += 0.7f * dev_sys_param->h)
						{
							float3 p = { i, j, k };
							float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
							if (r2 < dev_sys_param->h2)
							{
								float3 gradientKernel = dev_sys_param->grad_poly6 * pow(dev_sys_param->h2 - r2, 2) * p;
								gradientKernelSum += gradientKernel;
								squaredGradientKernelSum += dot(gradientKernel, gradientKernel);
							}
						}
					}
				}
				float sumGradientSquared = dot(gradientKernelSum, gradientKernelSum);
				float beta = 2.f * pow(dev_sys_param->mass * dev_sys_param->timestep / dev_sys_param->rest_dens, 2);
				dev_sys_param->delta = -1.f / beta / (-sumGradientSquared - squaredGradientKernelSum);
			}

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 cell_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);

			float3 f_total = make_float3(0.0f, 0.0f, 0.0f);
			float3 grad_color = make_float3(0.0f);
			float lplc_color = 0.0f;

			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbor_pos = cell_pos + make_int3(x, y, z);
						f_total = f_total + calc_cell_force(index, neighbor_pos, dev_particles, dev_hash, dev_index, dev_start, dev_end, grad_color, lplc_color, dev_sys_param);
					}
				}
			}
			dev_particles[index].force = f_total;

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
			void pred_vel_pos_kernel(ParticlePCI* dev_particles, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			ParticlePCI *p = &(dev_particles[index]);

			float3 acc = (p->force + p->force_pres) / p->dens + dev_sys_param->gravity;

			p->pred_vel = p->vel + acc * dev_sys_param->timestep;
			p->pred_pos = p->pos + p->pred_vel * dev_sys_param->timestep;
		}

		__global__
			void pred_dens_update_pres_kernel(ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam *dev_sys_param)
		{
			uint index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 cell_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);

			float pred_density = 0;

			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbor_pos = cell_pos + make_int3(x, y, z);
						pred_density = pred_density + calc_cell_pred_density(index, neighbor_pos, dev_particles, dev_hash, dev_index, dev_start, dev_end, dev_sys_param);
					}
				}
			}

			float dens_var = max(0.0f, pred_density - dev_sys_param->rest_dens);
			dev_sys_param->maxDensVariance = max(dev_sys_param->maxDensVariance, dens_var);  // The maxDensVariance will be back to 0 in the next step

			dev_particles[index].pres += dev_sys_param->delta * dens_var;
		}

		__global__
			void calc_pres_force_kernel(ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index == 0)
			{
				dev_sys_param->maxDensVariance = 0.0f;
			}

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			int3 cell_pos = calc_cell_pos(dev_particles[index].pos, dev_sys_param->cell_size);

			float3 pres_force = make_float3(0.0f, 0.0f, 0.0f);

			for (int z = -1; z <= 1; z++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int x = -1; x <= 1; x++)
					{
						int3 neighbor_pos = cell_pos + make_int3(x, y, z);
						pres_force = pres_force + calc_cell_pres_force(index, neighbor_pos, dev_particles, dev_hash, dev_index, dev_start, dev_end, dev_sys_param);
					}
				}
			}
			dev_particles[index].force_pres = pres_force;
		}

		__global__
			void integrate_kernel_pci(ParticlePCI* dev_particles, int* dev_occupied, SysParam* dev_sys_param)
		{
			uint index = blockIdx.x*blockDim.x + threadIdx.x;

			if (index >= dev_sys_param->num_particles)
			{
				return;
			}

			ParticlePCI *p = &(dev_particles[index]);

			float3 acc = (p->force + p->force_pres) / p->dens + dev_sys_param->gravity;

			p->vel = p->vel + acc*dev_sys_param->timestep;
			p->pos = p->pos + p->vel*dev_sys_param->timestep;

			int3 cell_pos = calc_cell_pos(p->pos, dev_sys_param->cell_size);
			int cell_index = cell_pos.z*dev_sys_param->grid_size.x*dev_sys_param->grid_size.y + cell_pos.y*dev_sys_param->grid_size.x + cell_pos.x;
			if (dev_occupied[cell_index] == 1)
			{
				p->vel = p->vel*dev_sys_param->bound_damping;
				p->pos = p->pos + p->vel*0.002;
			}

			if (p->pos.x > dev_sys_param->world_size.x - BOUNDARY)
			{
				p->vel.x = p->vel.x*dev_sys_param->bound_damping;
				p->pos.x = dev_sys_param->world_size.x - BOUNDARY;
			}

			if (p->pos.x < 0.0f)
			{
				p->vel.x = p->vel.x*dev_sys_param->bound_damping;
				p->pos.x = BOUNDARY;
			}

			if (p->pos.y > dev_sys_param->world_size.y - BOUNDARY)
			{
				p->vel.y = p->vel.y*dev_sys_param->bound_damping;
				p->pos.y = dev_sys_param->world_size.y - BOUNDARY;
			}

			if (p->pos.y < 0.0f)
			{
				p->vel.y = p->vel.y*dev_sys_param->bound_damping;
				p->pos.y = BOUNDARY;
			}

			if (p->pos.z > dev_sys_param->world_size.z - BOUNDARY)
			{
				p->vel.z = p->vel.z*dev_sys_param->bound_damping;
				p->pos.z = dev_sys_param->world_size.z - BOUNDARY;
			}

			if (p->pos.z < 0.0f)
			{
				p->vel.z = p->vel.z*dev_sys_param->bound_damping;
				p->pos.z = BOUNDARY;
			}

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


		__host__
			SimulateSystemPCI::SimulateSystemPCI(float3 world_size, float3 sim_ratio, float3 world_origin,
				int max_particles, float h, float mass, float3 gravity, float bound_damping,
				float rest_dens, float gas_const, float visc, float timestep, float surf_norm, float surf_coef)
		{
			sys_running_ = false;
			sys_param_ = new SysParam();
			sys_param_->num_particles = 0;

			sys_param_->max_particles = max_particles;
			sys_param_->h = h;
			sys_param_->mass = mass;

			sys_param_->world_size = world_size;
			sys_param_->cell_size = sys_param_->h;
			sys_param_->grid_size.x = (int)ceil(sys_param_->world_size.x / sys_param_->cell_size);
			sys_param_->grid_size.y = (int)ceil(sys_param_->world_size.y / sys_param_->cell_size);
			sys_param_->grid_size.z = (int)ceil(sys_param_->world_size.z / sys_param_->cell_size);
			sys_param_->total_cells = sys_param_->grid_size.x*sys_param_->grid_size.y*sys_param_->grid_size.z;

			sys_param_->gravity = gravity;
			sys_param_->bound_damping = bound_damping;
			sys_param_->rest_dens = rest_dens;
			sys_param_->gas_const = gas_const;
			sys_param_->visc = visc;
			sys_param_->timestep = timestep;
			sys_param_->surf_norm = surf_norm;
			sys_param_->surf_coef = surf_coef;

			sys_param_->poly6 = 315.0f / (64.0f * PI * pow(sys_param_->h, 9));
			sys_param_->grad_poly6 = -945 / (32 * PI * pow(sys_param_->h, 9));
			sys_param_->lplc_poly6 = 945 / (8 * PI * pow(sys_param_->h, 9));

			sys_param_->grad_spiky = -45.0f / (PI * pow(sys_param_->h, 6));
			sys_param_->lplc_visco = 45.0f / (PI * pow(sys_param_->h, 6));

			sys_param_->h2 = sys_param_->h*sys_param_->h;
			sys_param_->self_lplc_color = sys_param_->lplc_poly6*sys_param_->mass*sys_param_->h2*(0 - 3.f / 4.f * sys_param_->h2);


			cudaMalloc(&dev_sys_param_, sizeof(SysParam));

			particlesPCI_ = (ParticlePCI *)malloc(sizeof(ParticlePCI)*sys_param_->max_particles);
			occupied_ = (int *)malloc(sizeof(int)*sys_param_->total_cells);
			memset(occupied_, 0, sizeof(int)*sys_param_->total_cells);
			cudaMalloc(&dev_occupied_, sys_param_->total_cells * sizeof(int));
			cudaMemset(dev_occupied_, 0, sizeof(int)*sys_param_->total_cells);
			cudaMalloc(&dev_particlesPCI_, sys_param_->max_particles * sizeof(ParticlePCI));
			cudaMalloc(&dev_hash_, sizeof(uint)*sys_param_->max_particles);
			cudaMalloc(&dev_index_, sizeof(uint)*sys_param_->max_particles);
			cudaMalloc(&dev_start_, sizeof(uint)*sys_param_->total_cells);
			cudaMalloc(&dev_end_, sizeof(uint)*sys_param_->total_cells);

			//Marching Cubes
			float vox_size = 0.02f;
			uint3 dim_vox = make_uint3(ceil(world_size.x / vox_size),
				ceil(world_size.y / vox_size),
				ceil(world_size.z / vox_size));
			marchingCube_ = new MarchingCube(dim_vox, sim_ratio, world_origin, vox_size, sys_param_->rest_dens, sys_param_->max_particles);

			// PCISPH parameters
			sys_param_->minIteration = 3;
			sys_param_->maxIteration = 100;
			sys_param_->eta = 0.02f;
			sys_param_->maxDensVariance = 0.0f;
		}

		__host__
			SimulateSystemPCI::~SimulateSystemPCI()
		{
			for (int i = 0; i < scene_objects.size(); ++i)
				delete scene_objects[i];
			free(particlesPCI_);
			free(occupied_);
			delete marchingCube_;
			cudaFree(dev_occupied_);
			cudaFree(dev_sys_param_);
			cudaFree(dev_particlesPCI_);
			cudaFree(dev_hash_);
			cudaFree(dev_index_);
			cudaFree(dev_start_);
			cudaFree(dev_end_);
		}

		__host__
			void SimulateSystemPCI::start() {
			sys_running_ = true;
			cudaMemcpy(dev_particlesPCI_, particlesPCI_, sys_param_->num_particles * sizeof(ParticlePCI), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_sys_param_, sys_param_, sizeof(SysParam), cudaMemcpyHostToDevice);
			//marchingCube_->init(sys_param_->num_particles);
			std::cout << "Started simulation with " << sys_param_->num_particles << " particles" << std::endl;
		}

		__host__
			void SimulateSystemPCI::animation()
		{
			if (sys_running_ == 0 || sys_param_->num_particles == 0)
			{
				return;
			}

			build_table();
			comp_dens();
			comp_other_force();
			//comp_threshold_delta();

			int k = 0;
			float maxDensVar = 0.0f;
			while (maxDensVar > sys_param_->threshold || k < sys_param_->minIteration)
			{
				maxDensVar = 0.0f;
				if (k > sys_param_->maxIteration)
					break;
				pred_vel_pos();
				pred_dens_update_pres(maxDensVar);
				comp_pres_force();
				k++;
			}

			integratePCI();


			//marchingCube_->compute(dev_particles_);

			cudaMemcpy(particlesPCI_, dev_particlesPCI_, sizeof(ParticlePCI)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
		}

		__host__
			void SimulateSystemPCI::add_cube_fluid(const float3 &pos_min, const float3 &pos_max, const float gap)
		{
			float3 pos;
			float3 vel;

			vel.x = 0.f;
			vel.y = 0.f;
			vel.z = 0.f;

			for (pos.x = sys_param_->world_size.x*pos_min.x; pos.x < sys_param_->world_size.x*pos_max.x; pos.x += (sys_param_->h*gap))
			{
				for (pos.y = sys_param_->world_size.y*pos_min.y; pos.y < sys_param_->world_size.y*pos_max.y; pos.y += (sys_param_->h*gap))
				{
					for (pos.z = sys_param_->world_size.z*pos_min.z; pos.z < sys_param_->world_size.z*pos_max.z; pos.z += (sys_param_->h*gap))
					{
						add_particle(pos, vel);
					}
				}
			}

			cudaMemcpy(dev_particlesPCI_, particlesPCI_, sys_param_->num_particles * sizeof(ParticlePCI), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_sys_param_, sys_param_, sizeof(SysParam), cudaMemcpyHostToDevice);
		}

		__host__
			void SimulateSystemPCI::add_particle(const float3 &pos, const float3 &vel)
		{
			ParticlePCI *p = &(particlesPCI_[sys_param_->num_particles]);

			p->pos = pos;
			p->vel = vel;

			p->force.x = 0.0f;
			p->force.y = 0.0f;
			p->force.z = 0.0f;

			p->force_pres.x = 0.0f;
			p->force_pres.y = 0.0f;
			p->force_pres.z = 0.0f;

			p->dens = sys_param_->rest_dens;
			p->pres = 0.0f;

			sys_param_->num_particles++;
		}

		__host__
			void SimulateSystemPCI::build_table()
		{
			if (sys_param_->num_particles == 0)
				return;
			uint num_threads;
			uint num_blocks;

			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);
			calc_hash << < num_blocks, num_threads >> > (dev_hash_, dev_index_, dev_particlesPCI_, dev_sys_param_);

			/*uint * debug_hash = (uint*)malloc(sizeof(uint) * sys_param_->num_particles);
			cudaMemcpy(debug_hash, dev_hash_, sizeof(uint)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
			for (int i = 0; i < sys_param_->num_particles; i++)
			{
				std::cout << debug_hash[i] << std::endl;
			}
			free(debug_hash);*/

			thrust::sort_by_key(thrust::device_ptr<uint>(dev_hash_),
				thrust::device_ptr<uint>(dev_hash_ + sys_param_->num_particles),
				thrust::device_ptr<uint>(dev_index_));

			cudaMemset(dev_start_, 0xffffffff, sys_param_->total_cells * sizeof(uint));
			cudaMemset(dev_end_, 0x0, sys_param_->total_cells * sizeof(uint));

			uint smemSize = sizeof(int)*(num_threads + 1);

			find_start_end_kernel_pci << < num_blocks, num_threads, smemSize >> > (dev_start_, dev_end_, dev_hash_, dev_index_, sys_param_->num_particles);
		}
		

		__host__
			void SimulateSystemPCI::comp_dens()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			calc_dens_kernel << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);

			/*uint * debug_hash = (uint*)malloc(sizeof(uint) * sys_param_->num_particles);
			ParticlePCI * debug_particle = (ParticlePCI*)malloc(sizeof(ParticlePCI) * sys_param_->num_particles);
			cudaMemcpy(debug_hash, dev_hash_, sizeof(uint)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_particle, dev_particlesPCI_, sizeof(ParticlePCI) * sys_param_->num_particles, cudaMemcpyDeviceToHost);
			for (int i = 0; i < sys_param_->num_particles; i++)
			{
				std::cout << "hash: " << debug_hash[i] << std::endl;
				std::cout << "particles: " << debug_particle[i].dens << std::endl;
			}
			free(debug_hash);
			free(debug_particle);*/
		}
		
		__device__ __forceinline__
			float calc_cell_density(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			float total_cell_density = 0.0f;
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);
			if (grid_hash == 0xffffffff)
			{
				return total_cell_density;
			}
			uint start_index = dev_start[grid_hash];

			float mass = dev_sys_param->mass;
			float h2 = dev_sys_param->h2;
			float poly6 = dev_sys_param->poly6;

			float3 rel_pos;
			float r2;

			ParticlePCI *p = &(dev_particles[index]);
			ParticlePCI *np;
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

					if (r2 >= h2)
						continue;

					total_cell_density += mass * poly6 * pow(h2 - r2, 3);
				}
			}

			return total_cell_density;
		}



		__host__
			void SimulateSystemPCI::comp_other_force()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);
			calc_force_kernel << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);

			SysParam * param = (SysParam*)malloc(sizeof(SysParam));
			cudaMemcpy(param, dev_sys_param_, sizeof(SysParam), cudaMemcpyDeviceToHost);
			sys_param_->threshold = param->threshold;
			free(param);

			//uint * debug_hash = (uint*)malloc(sizeof(uint) * sys_param_->num_particles);
			//ParticlePCI * debug_particle = (ParticlePCI*)malloc(sizeof(ParticlePCI) * sys_param_->num_particles);
			//cudaMemcpy(debug_hash, dev_hash_, sizeof(uint)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//cudaMemcpy(debug_particle, dev_particlesPCI_, sizeof(ParticlePCI) * sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//for (int i = 0; i < sys_param_->num_particles; i++)
			//{
			//	//std::cout << "hash: " << debug_hash[i] << std::endl;
			//	std::cout << "particles: " << debug_particle[i].force.x << " " << debug_particle[i].force.y << " " << debug_particle[i].force.z << std::endl;
			//}
			//free(debug_hash);
			//free(debug_particle);
		}
		
		__device__ __forceinline__
			float3 calc_cell_force(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, float3 &grad_color, float &lplc_color, SysParam* dev_sys_param)
		{
			float3 total_cell_force = make_float3(0.0f);
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);

			if (grid_hash == 0xffffffff)
			{
				return total_cell_force;
			}

			uint start_index = dev_start[grid_hash];

			float h = dev_sys_param->h;
			float mass = dev_sys_param->mass;
			float h2 = dev_sys_param->h2;

			uint neighbor_index;

			ParticlePCI *p = &(dev_particles[index]);
			ParticlePCI *np;

			float3 rel_pos;
			float r2;
			float r;
			float vol;
			float h_r;

			float3 f_pressure;
			float3 f_visco;

			float3 rel_vel;

			if (start_index != 0xffffffff)
			{
				uint end_index = dev_end[grid_hash];

				for (uint count_index = start_index; count_index < end_index; count_index++)
				{
					neighbor_index = dev_index[count_index];

					np = &(dev_particles[neighbor_index]);

					rel_pos = p->pos - np->pos;
					r2 = rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z;

					if (r2 < h2 && r2 > EPS_F)
					{
						vol = mass / np->dens;
						// norm of relative pos
						r = sqrt(r2);

						// diff for kernel size and relative pos
						h_r = h - r;

						// calculate pressure force
						//f_pressure = rel_pos / r * vol * (p->pres + np->pres) / 2.f  * dev_sys_param->grad_spiky * h_r * h_r;
						//total_cell_force -= f_pressure;

						// calculate viscosity force
						f_visco = dev_sys_param->visc * vol * (np->vel - p->vel) * dev_sys_param->lplc_visco * h_r;
						total_cell_force += f_visco;

						// calculate color field according to paper Realtime particle-based fluid simulation [Stefan Auer et. al]
						grad_color -= rel_pos * dev_sys_param->grad_poly6 * vol * pow(h2 - r2, 2);
						lplc_color += dev_sys_param->lplc_poly6 * vol * (h2 - r2) * (r2 - 3.f / 4.f * (h2 - r2));
					}
				}
			}

			return total_cell_force;
		}

		

		__host__
			void SimulateSystemPCI::comp_threshold_delta()
		{
			cudaMemcpy(sys_param_, dev_sys_param_, sizeof(SysParam), cudaMemcpyDeviceToHost);
			
			// calculate threshold
			sys_param_->threshold = sys_param_->eta * sys_param_->rest_dens;

			// calculate delta
			float3 gradientKernelSum = { 0.0f, 0.0f, 0.0f };
			float squaredGradientKernelSum = 0.f;
			for (float i = -sys_param_->h; i <= sys_param_->h; i += 0.5f * sys_param_->h)    // The distance 0.5f * sys_param_->h can be changed!
			{
				for (float j = -sys_param_->h; j <= sys_param_->h; j += 0.5f * sys_param_->h)
				{
					for (float k = -sys_param_->h; k <= sys_param_->h; k += 0.5f * sys_param_->h)
					{
						float3 p = { i, j, k };
						float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
						if (r2 < sys_param_->h2)
						{
							float3 gradientKernel = sys_param_->grad_poly6 * std::pow(sys_param_->h2 - r2, 2) * p;
							gradientKernelSum += gradientKernel;
							squaredGradientKernelSum += dot(gradientKernel, gradientKernel);
						}
					}
				}
			}
			float sumGradientSquared = dot(gradientKernelSum, gradientKernelSum);
			float beta = 2.f * std::pow(sys_param_->mass * sys_param_->timestep / sys_param_->rest_dens, 2);
			sys_param_->delta = -1.f / beta / (-sumGradientSquared - squaredGradientKernelSum);

			std::cout << sys_param_->delta << std::endl;

			cudaMemcpy(dev_sys_param_, sys_param_, sizeof(SysParam), cudaMemcpyHostToDevice);
		}

		__host__
			void SimulateSystemPCI::pred_vel_pos()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			pred_vel_pos_kernel << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_sys_param_);

			/*SysParam * debug_param = (SysParam*)malloc(sizeof(SysParam));
			cudaMemcpy(debug_param, dev_sys_param_, sizeof(SysParam), cudaMemcpyDeviceToHost);
			std::cout << "delta: " << debug_param->delta << std::endl;
			std::cout << debug_param->eta << std::endl;
			std::cout << debug_param->threshold << std::endl;
			free(debug_param);*/

			//uint * debug_hash = (uint*)malloc(sizeof(uint) * sys_param_->num_particles);
			//ParticlePCI * debug_particle = (ParticlePCI*)malloc(sizeof(ParticlePCI) * sys_param_->num_particles);
			//cudaMemcpy(debug_hash, dev_hash_, sizeof(uint)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//cudaMemcpy(debug_particle, dev_particlesPCI_, sizeof(ParticlePCI) * sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//for (int i = 0; i < sys_param_->num_particles; i++)
			//{
			//	//std::cout << "hash: " << debug_hash[i] << std::endl;
			//	std::cout << "particles: " << debug_particle[i].pred_pos.x << " " << debug_particle[i].pred_pos.y << " " << debug_particle[i].pred_pos.z << std::endl;
			//}
			//free(debug_hash);
			//free(debug_particle);
		}
		

		__host__
			void SimulateSystemPCI::pred_dens_update_pres(float &maxDensVar)  //there may be problems with maxDensVar! Does maxDensVar keep changing?
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			pred_dens_update_pres_kernel << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);

			SysParam * param = (SysParam*)malloc(sizeof(SysParam));
			cudaMemcpy(param, dev_sys_param_, sizeof(SysParam), cudaMemcpyDeviceToHost);
			maxDensVar = param->maxDensVariance;
			free(param);

			//std::cout << maxDensVar << std::endl;

			//SysParam * debug_param = (SysParam*)malloc(sizeof(SysParam));
			//cudaMemcpy(debug_param, dev_sys_param_, sizeof(SysParam), cudaMemcpyDeviceToHost);
			//std::cout << "delta: " << debug_param->delta << std::endl;
			//std::cout << debug_param->eta << std::endl;
			//std::cout << debug_param->threshold << std::endl;
			//free(debug_param);

			//uint * debug_hash = (uint*)malloc(sizeof(uint) * sys_param_->num_particles);
			//ParticlePCI * debug_particle = (ParticlePCI*)malloc(sizeof(ParticlePCI) * sys_param_->num_particles);
			//cudaMemcpy(debug_hash, dev_hash_, sizeof(uint)*sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//cudaMemcpy(debug_particle, dev_particlesPCI_, sizeof(ParticlePCI) * sys_param_->num_particles, cudaMemcpyDeviceToHost);
			//for (int i = 0; i < sys_param_->num_particles; i++)
			//{
			//	//std::cout << "hash: " << debug_hash[i] << std::endl;
			//	std::cout << "particles: " << debug_particle[i].pres << std::endl;
			//}
			//free(debug_hash);
			//free(debug_particle);
		}
		
		__device__ __forceinline__
			float calc_cell_pred_density(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			float total_cell_density = 0.0f;
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);
			if (grid_hash == 0xffffffff)
			{
				return total_cell_density;
			}
			uint start_index = dev_start[grid_hash];

			float mass = dev_sys_param->mass;
			float h2 = dev_sys_param->h2;
			float poly6 = dev_sys_param->poly6;

			float3 rel_pos;
			float r2;

			ParticlePCI *p = &(dev_particles[index]);
			ParticlePCI *np;
			uint neighbor_index;

			if (start_index != 0xffffffff)
			{
				uint end_index = dev_end[grid_hash];

				for (uint count_index = start_index; count_index < end_index; count_index++)
				{
					neighbor_index = dev_index[count_index];
					np = &(dev_particles[neighbor_index]);

					rel_pos = np->pred_pos - p->pred_pos;
					r2 = rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z;

					if (r2 >= h2)
						continue;

					total_cell_density += mass * poly6 * pow(h2 - r2, 3);
				}
			}

			return total_cell_density;
		}



		__host__
			void SimulateSystemPCI::comp_pres_force()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);
			calc_pres_force_kernel << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_hash_, dev_index_, dev_start_, dev_end_, dev_sys_param_);
		}
		
		__device__ __forceinline__
			float3 calc_cell_pres_force(uint index, int3 neighbor, ParticlePCI *dev_particles, uint *dev_hash, uint *dev_index, uint *dev_start, uint *dev_end, SysParam* dev_sys_param)
		{
			float3 total_cell_force = make_float3(0.0f);
			uint grid_hash = calc_cell_hash(neighbor, dev_sys_param->grid_size);

			if (grid_hash == 0xffffffff)
			{
				return total_cell_force;
			}

			uint start_index = dev_start[grid_hash];

			float h = dev_sys_param->h;
			float mass = dev_sys_param->mass;
			float h2 = dev_sys_param->h2;

			uint neighbor_index;

			ParticlePCI *p = &(dev_particles[index]);
			ParticlePCI *np;

			float3 rel_pos;
			float r2;
			float r;
			float vol;
			float h_r;

			float3 f_pressure;

			float3 rel_vel;

			if (start_index != 0xffffffff)
			{
				uint end_index = dev_end[grid_hash];

				for (uint count_index = start_index; count_index < end_index; count_index++)
				{
					neighbor_index = dev_index[count_index];

					np = &(dev_particles[neighbor_index]);

					rel_pos = p->pos - np->pos;
					r2 = rel_pos.x*rel_pos.x + rel_pos.y*rel_pos.y + rel_pos.z*rel_pos.z;

					if (r2 < h2 && r2 > EPS_F)
					{
						// norm of relative pos
						r = sqrt(r2);

						// diff for kernel size and relative pos
						h_r = h - r;

						// calculate pressure force
						f_pressure = rel_pos / r * mass * mass * (p->pres / (p->dens * p->dens) + np->pres / (np->dens * np->dens)) * dev_sys_param->grad_spiky * h_r * h_r;
						total_cell_force -= f_pressure;

					}
				}
			}

			return total_cell_force;
		}


		__host__
			void SimulateSystemPCI::integratePCI()
		{
			uint num_threads;
			uint num_blocks;
			calc_grid_size(sys_param_->num_particles, 512, num_blocks, num_threads);

			integrate_kernel_pci << < num_blocks, num_threads >> > (dev_particlesPCI_, dev_occupied_, dev_sys_param_);
		}
		


	}
}
