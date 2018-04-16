#include "fluidsim_system.h"
#define PI 3.141592f
#define INF 1E-12f
#define BOUNDARY 0.0001f

namespace FluidSim {
	SimulateSystem::SimulateSystem(float w_x, float w_y, float w_z)
	{
		sys_running = false;
		num_particles_ = 0;
		minIteration = 3;
		maxIteration = 10000;

		max_particles_ = 30000;
		kernel_ = 0.04f;
		mass_ = 0.02f;

		world_size_(0) = w_x;
		world_size_(1) = w_y;
		world_size_(2) = w_z;
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
		timestep_ = 0.001f; // original timestep = 0.003f
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

		particle_radius = 0.01f;
		densityVarianceThreshold = rest_dens_ * 0.01 * 10;
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

		p->radius = particle_radius;

		p->pos = pos;
		p->vel = vel;

		p->acc(0) = 0.0f;
		p->acc(0) = 0.0f;
		p->acc(0) = 0.0f;
		p->ev(0) = 0.0f;
		p->ev(1) = 0.0f;
		p->ev(2) = 0.0f;

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
		/*init_dens();
		init_force();

		int iter = 0;
		while (iter < maxIteration) {
			pred_vel_pos();
			update_dens_var_scale();
			predDensVar_updatePres();
			update_presForce();
			if (++iter >= minIteration && maxDensityVariance < densityVarianceThreshold) {
				break;
			}
		}*/

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
							rel_pos = (np->pos - p->pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2<INF || r2 >= kernel2_)
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

			p->acc(0) = 0.0f;
			p->acc(1) = 0.0f;
			p->acc(2) = 0.0f;

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

								pres_kernel = spiky_value_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->acc -=rel_pos*temp_force / r;

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
			p->surf_norm = grad_color.norm();

			if (p->surf_norm > surf_norm_)
			{
				p->acc += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			}
		}
	}

	void SimulateSystem::integrate()
	{
		Particle *p;
		for (unsigned int i = 0; i<num_particles_; i++)
		{
			p = &(particles_[i]);
			
			
			p->vel += p->acc*timestep_ / p->dens + gravity_*timestep_;
			// change p->dens to mass_
			//p->vel += p->acc*timestep_ / mass_ + gravity_*timestep_;
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

			p->ev = (p->ev + p->vel) / 2;
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

	void SimulateSystem::init_dens()
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
			//p->pres = 0.0f;

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
							rel_pos = (np->pos - p->pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2<INF || r2 >= kernel2_)
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
		}
	}

	void SimulateSystem::init_force()
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

			p->acc(0) = 0.0f;
			p->acc(1) = 0.0f;
			p->acc(2) = 0.0f;

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
						near_pos = cell_pos + Vector3i(x, y, z);
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

								/*pres_kernel = spiky_value_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->acc -= rel_pos*temp_force / r;*/

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
			p->surf_norm = grad_color.norm();

			if (p->surf_norm > surf_norm_)
			{
				p->acc += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			}

			p->pres = 0.0f;
			p->pres_force(0) = 0.f;
			p->pres_force(1) = 0.f;
			p->pres_force(2) = 0.f;
		}
	}

	void SimulateSystem::pred_vel_pos()
	{
		Particle *p;
		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);

			p->pred_vel = p->vel + (p->acc + p->pres_force) * timestep_ / p->dens + gravity_ * timestep_;
			p->pred_pos = p->pos + p->pred_vel * timestep_;
		}

	}

	void SimulateSystem::update_dens_var_scale() 
	{
		Vector3f gradientKernelSum = Vector3f(0.f, 0.f, 0.f);
		float squaredGradientKernelSum = 0.f;

		for (float i = -kernel_ - particle_radius; i <= kernel_ + particle_radius; i += 2.f * particle_radius)
		{
			for (float j = -kernel_ - particle_radius; j <= kernel_ + particle_radius; j += 2.f * particle_radius)
			{
				for (float k = -kernel_ - particle_radius; k <= kernel_ + particle_radius; k += 2.f * particle_radius)
				{
					Vector3f p = Vector3f(i, j, k);
					float r2 = p.squaredNorm();
					float r = std::sqrt(r2);
					if (r2 < kernel2_)
					{
						Vector3f gradientKernel = grad_poly6_ * pow(kernel2_ - r2, 2) * p;
						gradientKernelSum += gradientKernel;
						squaredGradientKernelSum += gradientKernel.dot(gradientKernel);
					}
				}
			}
		}

		float sumGradientSquared = gradientKernelSum.dot(gradientKernelSum);
		float beta = 2.f * pow(timestep_ * mass_ / rest_dens_, 2);
		delta = -1.f / (beta * (-sumGradientSquared - squaredGradientKernelSum));

	}

	void SimulateSystem::predDensVar_updatePres()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		float r2;

		maxDensityVariance = -INF;
		aveDensityVariance = 0.f;
		float accDensityVariance = 0.f;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->pos);

			float density = 0.f;

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
							rel_pos = (np->pred_pos - p->pred_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < INF || r2 >= kernel2_)
							{
								np = np->next;
								continue;
							}

							density += mass_ * poly6_value_ * pow(kernel2_ - r2, 3);

							np = np->next;
						}
					}
				}
			}

			float dens_var = std::max(0.f, density - rest_dens_);
			maxDensityVariance = std::max(maxDensityVariance, dens_var);
			accDensityVariance += dens_var;

			p->pres += delta * dens_var;
		}
		aveDensityVariance = accDensityVariance / num_particles_;
	}
	
	void SimulateSystem::update_presForce()
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
		float temp_force;

		Vector3f grad_color;
		float lplc_color;

		for (unsigned int i = 0; i<num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->pos);

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
							rel_pos = p->pos - np->pos;
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > INF)
							{
								r = sqrt(r2);
								V = mass_ / np->dens / 2;
								kernel_r = kernel_ - r;

								pres_kernel = spiky_value_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->pres_force = rel_pos * temp_force / r;
								p->acc -= p->pres_force;

							}
							np = np->next;
						}
					}
				}
			}
		}
	}

}