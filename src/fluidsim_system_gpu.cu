#include "fluidsim_system_gpu.cuh"
#include <cmath>
#include <helper_math.h>
#define PI 3.141592f
#define INF 1E-12f
#define BOUNDARY 0.0001f

namespace FluidSim {

	__global__
	void compute_dens_pres_kernel(Particle *dev_particles)
	{
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		dev_particles[i].pos = dev_particles[i].pos + make_float3(0.0001f);
	}

	SimulateSystem::SimulateSystem()
	{
		sys_running = false;
		num_particles_ = 0;

		max_particles_ = 30000;
		kernel_ = 0.04f;
		mass_ = 0.02f;

		world_size_.x = 0.64f;
		world_size_.y = 0.64f;
		world_size_.z = 0.64f;
		cell_size_ = kernel_;
		grid_size_.x = (int)ceil(world_size_.x / cell_size_);
		grid_size_.y = (int)ceil(world_size_.y / cell_size_);
		grid_size_.z = (int)ceil(world_size_.z / cell_size_);
		total_cells_ = grid_size_.x*grid_size_.y*grid_size_.z;

		gravity_.x = 0.f;
		gravity_.y = -9.8f;
		gravity_.z = 0.0f;
		bound_damping_ = -0.5f;
		rest_dens_ = 1000.f;
		gas_const_ = 1.0f;
		visc_ = 6.5f;
		timestep_ = 0.003f;
		surf_norm_ = 6.0f;
		surf_coef_ = 0.1f;

		poly6_value_ = 315.0f / (64.0f * PI * pow(kernel_, 9));
		spiky_value_ = -45.0f / (PI * pow(kernel_, 6));
		visco_value_ = 45.0f / (PI * pow(kernel_, 6));

		grad_poly6_ = -945 / (32 * PI * pow(kernel_, 9));
		lplc_poly6_ = -945 / (8 * PI * pow(kernel_, 9));

		kernel2_ = kernel_*kernel_;
		self_dens_ = mass_*poly6_value_*pow(kernel_, 6);
		self_lplc_color_ = lplc_poly6_*mass_*kernel2_*(0 - 3 / 4 * kernel2_);

		particles_ = (Particle *)malloc(sizeof(Particle)*max_particles_);
		cells_ = (Particle **)malloc(sizeof(Particle *)*total_cells_);
	}

	SimulateSystem::~SimulateSystem() {
		free(particles_);
		free(cells_);
	}

	void SimulateSystem::start() {
		sys_running = true;
	}

	void SimulateSystem::add_cube_fluid(const float3 &pos_min, const float3 &pos_max)
	{
		float3 pos;
		float3 vel;

		vel.x = 0.f;
		vel.y = 0.f;
		vel.z = 0.f;

		for (pos.x = world_size_.x*pos_min.x; pos.x < world_size_.x*pos_max.x; pos.x += (kernel_*0.5f))
		{
			for (pos.y = world_size_.y*pos_min.y; pos.y < world_size_.y*pos_max.y; pos.y += (kernel_*0.5f))
			{
				for (pos.z = world_size_.z*pos_min.z; pos.z < world_size_.z*pos_max.z; pos.z += (kernel_*0.5f))
				{
					add_particle(pos, vel);
				}
			}
		}
	}

	void SimulateSystem::add_particle(const float3 &pos, const float3 &vel)
	{
		Particle *p = &(particles_[num_particles_]);

		p->id = num_particles_;

		p->pos = pos;
		p->vel = vel;

		p->acc.x = 0.0f;
		p->acc.x = 0.0f;
		p->acc.x = 0.0f;
		p->ev.x = 0.0f;
		p->ev.y = 0.0f;
		p->ev.z = 0.0f;

		p->dens = rest_dens_;
		p->pres = 0.0f;

		p->next = nullptr;

		num_particles_++;
	}

	void SimulateSystem::animation()
	{
		if (sys_running == 0)
		{
			return;
		}

		build_table();
		comp_dens_pres();
		comp_force();
		integrate();
	}

	void SimulateSystem::build_table()
	{
		Particle *p;
		unsigned int hash;

		for (unsigned int i = 0; i < total_cells_; i++)
		{
			cells_[i] = NULL;
		}

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			hash = calc_cell_hash(calc_cell_pos(p->pos));

			if (cells_[hash] == NULL)
			{
				p->next = NULL;
				cells_[hash] = p;
			}
			else
			{
				p->next = cells_[hash];
				cells_[hash] = p;
			}
		}
	}

	void SimulateSystem::comp_dens_pres()
	{
		Particle *dev_particles = nullptr;
		cudaMalloc(&dev_particles, num_particles_ * sizeof(Particle));
		cudaMemcpy(dev_particles, particles_, num_particles_ * sizeof(Particle), cudaMemcpyHostToDevice);
		compute_dens_pres_kernel <<< (num_particles_ + 255) / 256, 256 >>>(dev_particles);
		cudaMemcpy(particles_, dev_particles, num_particles_ * sizeof(Particle), cudaMemcpyDeviceToHost);
		cudaFree(dev_particles);
		/*Particle *p;
		Particle *np;

		int3 cell_pos;
		int3 near_pos;
		unsigned int hash;

		float3 rel_pos;
		float r2;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->pos);

			p->dens = 0.0f;
			p->pres = 0.0f;

			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						near_pos = cell_pos + make_int3(x, y, z);
						hash = calc_cell_hash(near_pos);

						if (hash == 0xffffffff)
						{
							continue;
						}

						np = cells_[hash];
						while (np != NULL)
						{
							rel_pos = (np->pos - p->pos);
							r2 = dot(rel_pos, rel_pos);;

							if (r2 < INF || r2 >= kernel2_)
							{
								np = np->next;
								continue;
							}

							p->dens = p->dens + mass_ * poly6_value_ * pow(kernel2_ - r2, 3);

							np = np->next;
						}
					}
				}
			}

			p->dens = p->dens + self_dens_;
			p->pres = (pow(p->dens / rest_dens_, 7) - 1) *gas_const_;
		}*/
	}

	void SimulateSystem::comp_force()
	{
		Particle *p;
		Particle *np;

		int3 cell_pos;
		int3 near_pos;
		unsigned int hash;

		float3 rel_pos;
		float3 rel_vel;

		float r2;
		float r;
		float kernel_r;
		float V;

		float pres_kernel;
		float visc_kernel;
		float temp_force;

		float3 grad_color;
		float lplc_color;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->pos);

			p->acc.x = 0.0f;
			p->acc.y = 0.0f;
			p->acc.z = 0.0f;

			grad_color.x = 0.0f;
			grad_color.y = 0.0f;
			grad_color.z = 0.0f;
			lplc_color = 0.0f;

			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						near_pos = cell_pos + make_int3(x, y, z);
						hash = calc_cell_hash(near_pos);

						if (hash == 0xffffffff)
						{
							continue;
						}

						np = cells_[hash];
						while (np != NULL)
						{
							rel_pos = p->pos - np->pos;
							r2 = dot(rel_pos, rel_pos);

							if (r2 < kernel2_ && r2 > INF)
							{
								r = sqrt(r2);
								V = mass_ / np->dens / 2;
								kernel_r = kernel_ - r;

								pres_kernel = spiky_value_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->acc -= rel_pos*temp_force / r;

								rel_vel = np->ev - p->ev;

								visc_kernel = visco_value_*(kernel_ - r);
								temp_force = V * visc_ * visc_kernel;
								p->acc += rel_vel*temp_force;

								float temp = (-1) * grad_poly6_ * V * pow(kernel2_ - r2, 2);
								grad_color += temp * rel_pos;
								lplc_color += lplc_poly6_ * V * (kernel2_ - r2) * (r2 - 3 / 4 * (kernel2_ - r2));
							}

							np = np->next;
						}
					}
				}
			}

			lplc_color += self_lplc_color_ / p->dens;
			p->surf_norm = length(grad_color);

			if (p->surf_norm > surf_norm_)
			{
				p->acc += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			}
		}
	}

	void SimulateSystem::integrate()
	{
		Particle *p;
		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);

			p->vel += p->acc*timestep_ / p->dens + gravity_*timestep_;
			p->pos += p->vel*timestep_;

			if (p->pos.x >= world_size_.x - BOUNDARY)
			{
				p->vel.x = p->vel.x*bound_damping_;
				p->pos.x = world_size_.x - BOUNDARY;
			}

			if (p->pos.x < 0.0f)
			{
				p->vel.x = p->vel.x*bound_damping_;
				p->pos.x = 0.0f;
			}

			if (p->pos.y >= world_size_.y - BOUNDARY)
			{
				p->vel.y = p->vel.y*bound_damping_;
				p->pos.y = world_size_.y - BOUNDARY;
			}

			if (p->pos.y < 0.0f)
			{
				p->vel.y = p->vel.y*bound_damping_;
				p->pos.y = 0.0f;
			}

			if (p->pos.z >= world_size_.z - BOUNDARY)
			{
				p->vel.z = p->vel.z*bound_damping_;
				p->pos.z = world_size_.z - BOUNDARY;
			}

			if (p->pos.z < 0.0f)
			{
				p->vel.z = p->vel.z*bound_damping_;
				p->pos.z = 0.0f;
			}

			p->ev = (p->ev + p->vel) / 2;
		}
	}

	__host__ __device__
		int3 SimulateSystem::calc_cell_pos(float3 p)
	{
		int3 cell_pos;
		cell_pos.x = int(floor((p.x) / cell_size_));
		cell_pos.y = int(floor((p.y) / cell_size_));
		cell_pos.z = int(floor((p.z) / cell_size_));

		return cell_pos;
	}

	__host__ __device__
		unsigned int SimulateSystem::calc_cell_hash(int3 cell_pos)
	{
		if (cell_pos.x < 0 || cell_pos.x >= (int)grid_size_.x
			|| cell_pos.y < 0 || cell_pos.y >= (int)grid_size_.y
			|| cell_pos.z < 0 || cell_pos.z >= (int)grid_size_.z)
		{
			return (unsigned int)0xffffffff;
		}

		cell_pos.x = cell_pos.x & (grid_size_.x - 1);
		cell_pos.y = cell_pos.y & (grid_size_.y - 1);
		cell_pos.z = cell_pos.z & (grid_size_.z - 1);

		return ((unsigned int)(cell_pos.z))*grid_size_.y*grid_size_.x
			+ ((unsigned int)(cell_pos.y))*grid_size_.x
			+ (unsigned int)(cell_pos.x);
	}
}