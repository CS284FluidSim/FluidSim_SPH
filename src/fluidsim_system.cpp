#include "fluidsim_system.h"
#define PI 3.141592f
#define INF 1E-12f
#define BOUNDARY 0.0001f

namespace FluidSim {
	SimulateSystem::SimulateSystem(float world_size_x, float world_size_y, float world_size_z)
	{
		sys_running = false;
		num_particles_ = 0;

		max_particles_ = 250000;
		kernel_ = 0.04f;
		mass_ = 0.02f;

		world_size_(0) = world_size_x;
		world_size_(1) = world_size_y;
		world_size_(2) = world_size_z;
		cell_size_ = kernel_;
		grid_size_(0) = (int)ceil(world_size_(0) / cell_size_);
		grid_size_(1) = (int)ceil(world_size_(1) / cell_size_);
		grid_size_(2) = (int)ceil(world_size_(2) / cell_size_);
		total_cells_ = grid_size_(0)*grid_size_(1)*grid_size_(2);

		gravity_(0) = 0.f;
		gravity_(1) = -9.8f;
		gravity_(2) = 0.0f;
		bound_damping_ = -0.5f;
		rest_dens_ = 1000.f;
		gas_const_ = 1.0f;
		visc_ = 6.5f;
		timestep_ = 0.003f;
		surf_norm_ = 6.0f;
		surf_coef_ = 0.1f;

		poly6_value_ = 315.0f / (64.0f * PI * pow(kernel_, 9));
		grad_poly6_ = -945 / (32 * PI * pow(kernel_, 9));
		grad_spiky_ = -45.0f / (PI * pow(kernel_, 6));
		lplc_poly6_ = -945 / (8 * PI * pow(kernel_, 9));
		lplc_visco_ = 45.0f / (PI * pow(kernel_, 6));

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

	void SimulateSystem::add_cube_fluid(const Vector3f &pos_min,const Vector3f &pos_max)
	{
		Vector3f pos;
		Vector3f vel;

		vel(0) = 0.f;
		vel(1) = 0.f;
		vel(2) = 0.f;

		for (pos(0) = world_size_(0)*pos_min(0); pos(0)<world_size_(0)*pos_max(0); pos(0) += (kernel_*0.5f))
		{
			for (pos(1) = world_size_(1)*pos_min(1); pos(1)<world_size_(1)*pos_max(1); pos(1) += (kernel_*0.5f))
			{
				for (pos(2) = world_size_(2)*pos_min(2); pos(2)<world_size_(2)*pos_max(2); pos(2) += (kernel_*0.5f))
				{
					add_particle(pos, vel);
				}
			}
		}
	}

	void SimulateSystem::add_particle(const Vector3f &pos, const Vector3f &vel)
	{
		Particle *p = &(particles_[num_particles_]);

		p->id = num_particles_;

		p->pos = pos;
		p->vel = vel;

		p->force(0) = 0.0f;
		p->force(0) = 0.0f;
		p->force(0) = 0.0f;

		p->dens = rest_dens_;
		p->pres = 0.0f;

		p->surf_norm = 0.f;
		p->normal = Vector3f(0.f, 0.f, 0.f);

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

		for (unsigned int i = 0; i<total_cells_; i++)
		{
			cells_[i] = NULL;
		}

		for (unsigned int i = 0; i<num_particles_; i++)
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
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		float r2;

		for (unsigned int i = 0; i<num_particles_; i++)
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
						near_pos = cell_pos + Vector3i(x, y, z);
						hash = calc_cell_hash(near_pos);

						if (hash == 0xffffffff)
						{
							continue;
						}

						np = cells_[hash];
						while (np != NULL)
						{
							//relative position of the particle and its neighbour particle in the cell
							rel_pos = (np->pos - p->pos).cast<float>();
							//length between a particle and its neighbour particle
							r2 = rel_pos.squaredNorm();

							if (r2<INF || r2 >= kernel2_)
							{
								np = np->next;
								continue;
							}

							//density
							p->dens += mass_ * poly6_value_ * pow(kernel2_ - r2, 3);
							//next particle in cell
							np = np->next;
						}
					}
				}
			}

			//density = rest density + density
			p->dens = p->dens + self_dens_;
			//tait equation to compute for pressure
			p->pres = (pow(p->dens / rest_dens_, 7) - 1) *gas_const_;
		}
	}

	void SimulateSystem::comp_force()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		Vector3f rel_vel;

		float r2;
		float r;
		float kernel_r;
		float V;

		float pres_kernel;
		float visc_kernel;
		float temp_force;

		Vector3f grad_color;
		float lplc_color;

		for (unsigned int i = 0; i<num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->pos);

			p->force(0) = 0.0f;
			p->force(1) = 0.0f;
			p->force(2) = 0.0f;

			grad_color(0) = 0.0f;
			grad_color(1) = 0.0f;
			grad_color(2) = 0.0f;
			lplc_color = 0.0f;

			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						near_pos = cell_pos + Vector3i(x,y,z);
						hash = calc_cell_hash(near_pos);

						if (hash == 0xffffffff)
						{
							continue;
						}

						np = cells_[hash];
						while (np != NULL)
						{
							rel_pos = p->pos - np->pos;
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > INF)
							{
								r = sqrt(r2);
								V = mass_ / np->dens / 2;
								kernel_r = kernel_ - r;
								
								//-rij.normalized()*MASS*(pi.p + pj.p)/(2.f * pj.rho) * SPIKY_GRAD*pow(H-r,2.f);
								pres_kernel = grad_spiky_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->force -=rel_pos*temp_force / r;

								rel_vel = np->vel - p->vel;
								//VISC*MASS*(pj.v - pi.v)/pj.rho * VISC_LAP*(H-r);
								visc_kernel = lplc_visco_*(kernel_ - r);
								temp_force = V * visc_ * visc_kernel;
								p->force += rel_vel*temp_force;

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
			p->surf_norm = grad_color.norm();
			p->normal = grad_color/p->surf_norm;
			if (p->surf_norm > surf_norm_)
			{
				p->force += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			}
		}
	}

	void SimulateSystem::integrate()
	{
		Particle *p;
		for (unsigned int i = 0; i<num_particles_; i++)
		{
			p = &(particles_[i]);

			p->vel += p->force*timestep_ / p->dens + gravity_*timestep_;
			p->pos += p->vel*timestep_;

			if (p->pos(0) >= world_size_(0) - BOUNDARY)
			{
				p->vel(0) = p->vel(0)*bound_damping_;
				p->pos(0) = world_size_(0) - BOUNDARY;
			}

			if (p->pos(0) < 0.0f)
			{
				p->vel(0) = p->vel(0)*bound_damping_;
				p->pos(0) = 0.0f;
			}

			if (p->pos(1) >= world_size_(1) - BOUNDARY)
			{
				p->vel(1) = p->vel(1)*bound_damping_;
				p->pos(1) = world_size_(1) - BOUNDARY;
			}

			if (p->pos(1) < 0.0f)
			{
				p->vel(1) = p->vel(1)*bound_damping_;
				p->pos(1) = 0.0f;
			}

			if (p->pos(2) >= world_size_(2) - BOUNDARY)
			{
				p->vel(2) = p->vel(2)*bound_damping_;
				p->pos(2) = world_size_(2) - BOUNDARY;
			}

			if (p->pos(2) < 0.0f)
			{
				p->vel(2) = p->vel(2)*bound_damping_;
				p->pos(2) = 0.0f;
			}
		}
	}

	Vector3i SimulateSystem::calc_cell_pos(Vector3f p)
	{
		Vector3i cell_pos;
		cell_pos(0) = int(floor((p(0)) / cell_size_));
		cell_pos(1) = int(floor((p(1)) / cell_size_));
		cell_pos(2) = int(floor((p(2)) / cell_size_));

		return cell_pos;
	}

	unsigned int SimulateSystem::calc_cell_hash(Vector3i cell_pos)
	{
		if (cell_pos(0)<0 || cell_pos(0) >= (int)grid_size_(0) 
			|| cell_pos(1)<0 || cell_pos(1) >= (int)grid_size_(1) 
			|| cell_pos(2)<0 || cell_pos(2) >= (int)grid_size_(2))
		{
			return (unsigned int)0xffffffff;
		}

		cell_pos(0) = cell_pos(0) & (grid_size_(0) - 1);
		cell_pos(1) = cell_pos(1) & (grid_size_(1) - 1);
		cell_pos(2) = cell_pos(2) & (grid_size_(2) - 1);

		return ((unsigned int)(cell_pos(2)))*grid_size_(1)*grid_size_(0) 
			+ ((unsigned int)(cell_pos(1)))*grid_size_(0) 
			+ (unsigned int)(cell_pos(0));
	}
}