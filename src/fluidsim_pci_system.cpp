#include <iostream>
#include "fluidsim_pci_system.h"
#define PI 3.141592f
#define LOWER_LIMIT 1E-12f
#define BOUNDARY 0.0001f

namespace FluidSimPCI {
	SimulateSystem::SimulateSystem()
	{
		min_iteration_ = 3;
		max_iteration_ = 1000;
		sys_running = false;
		num_particles_ = 0;
		num_boundary_particles_ = 0;

		max_particles_ = 30000;
		kernel_ = 0.04f;
		rest_dens_ = 1000.f;
		radius_ = 0.01f;
		mass_ = rest_dens_ * pow(radius_ * 2, 3) / 1.1f;
		mass2_ = pow(mass_, 2);

		world_size_(0) = 0.16f;
		world_size_(1) = 0.16f;
		world_size_(2) = 0.16f;
		cell_size_ = kernel_;
		grid_size_(0) = (int)ceil(world_size_(0) / cell_size_);
		grid_size_(1) = (int)ceil(world_size_(1) / cell_size_);
		grid_size_(2) = (int)ceil(world_size_(2) / cell_size_);
		total_cells_ = grid_size_(0)*grid_size_(1)*grid_size_(2);

		gravity_(0) = 0.f;
		gravity_(1) = -9.8f;
		gravity_(2) = 0.0f;
		bound_damping_ = -0.5f;
		gas_const_ = 1.0f;
		visc_ = 0.0f;
		timestep_ = 0.001f;
		dens_var_limit_ = 0.02f * rest_dens_ * 10.f;

		surf_coef_ = 1.0f;

		poly6_value_ = 315.0f / (64.0f * PI * pow(kernel_, 9));
		spiky_value_ = -45.0f / (PI * pow(kernel_, 6));
		visco_value_ = 45.0f / (PI * pow(kernel_, 6));
		surf_value_ = 32.0f / (PI * pow(kernel_, 9));
		surf_offset_ = -pow(kernel_, 6) / 64.0f;

		grad_poly6_ = -945 / (32 * PI * pow(kernel_, 9));
		lplc_poly6_ = -945 / (8 * PI * pow(kernel_, 9));

		kernel2_ = kernel_*kernel_;
		self_dens_ = mass_*poly6_value_*pow(kernel_, 6);
		self_lplc_color_ = lplc_poly6_*mass_*kernel2_*(0 - 3 / 4 * kernel2_);

		particles_ = (Particle *)malloc(sizeof(Particle)*max_particles_);
		boundary_particles_ = (Particle *)malloc(sizeof(Particle)*max_particles_);
		cells_ = (Particle **)malloc(sizeof(Particle *)*total_cells_);
		boundary_cells_ = (Particle **)malloc(sizeof(Particle *)*total_cells_);
	}

	SimulateSystem::~SimulateSystem() {
		free(particles_);
		free(boundary_particles_);
		free(cells_);
		free(boundary_cells_);
	}

	void SimulateSystem::start() {
		sys_running = true;
	}

	void SimulateSystem::add_cube_fluid(const Vector3f &pos_min,const Vector3f &pos_max)
	{
		Vector3f pos;
		Vector3f curr_vel;

		curr_vel(0) = 0.f;
		curr_vel(1) = 0.f;
		curr_vel(2) = 0.f;

		for (pos(0) = world_size_(0)*pos_min(0); pos(0)<world_size_(0)*pos_max(0); pos(0) += (kernel_*0.5f))
		{
			for (pos(1) = world_size_(1)*pos_min(1); pos(1)<world_size_(1)*pos_max(1); pos(1) += (kernel_*0.5f))
			{
				for (pos(2) = world_size_(2)*pos_min(2); pos(2)<world_size_(2)*pos_max(2); pos(2) += (kernel_*0.5f))
				{
					add_particle(pos, curr_vel);
				}
			}
		}
	}

	void SimulateSystem::add_boundary()
	{
		add_boundary_particles();
		build_boundary_table();
		massify_boundary();
	}

	void SimulateSystem::add_boundary_particles() {
		Vector3f pos;
		Vector3f curr_vel(0.f, 0.f, 0.f);
		float xl = world_size_(0) - BOUNDARY;
		float yl = world_size_(1) - BOUNDARY;
		float zl = world_size_(2) - BOUNDARY;

		//xy
		for (float x = kernel_*0.5f; x<world_size_(0); x += (kernel_*0.5f))
		{
			for (float y = kernel_*0.5f; y<world_size_(1); y += (kernel_*0.5f))
			{
				add_boundary_particle(Vector3f(x, y, 0), Vector3f(0.f, 0.f, 1.f).normalized());
				add_boundary_particle(Vector3f(x, y, zl), Vector3f(0.f, 0.f, -1.f).normalized());
			}
		}

		//xz
		for (float x = kernel_*0.5f; x<world_size_(0); x += (kernel_*0.5f))
		{
			for (float z = kernel_*0.5f; z<world_size_(2); z += (kernel_*0.5f))
			{
				add_boundary_particle(Vector3f(x, 0, z), Vector3f(0.f, 1.f, 0.f).normalized());
				add_boundary_particle(Vector3f(x, yl, z), Vector3f(0.f, -1.f, 0.f).normalized());
			}
		}

		//yz
		for (float y = kernel_*0.5f; y<world_size_(1); y += (kernel_*0.5f))
		{
			for (float z = kernel_*0.5f; z<world_size_(2); z += (kernel_*0.5f))
			{
				add_boundary_particle(Vector3f(0, y, z), Vector3f(1.f, 0.f, 0.f).normalized());
				add_boundary_particle(Vector3f(xl, y, z), Vector3f(-1.f, 0.f, 0.f).normalized());
			}
		}

		// x borders
		for (float x = kernel_*0.5f; x < world_size_(0); x += (kernel_*0.5f))
		{
			add_boundary_particle(Vector3f(x, 0, 0), Vector3f(0.f, 1.f, 1.f).normalized());
			add_boundary_particle(Vector3f(x, yl, 0), Vector3f(0.f, -1.f, 1.f).normalized());
			add_boundary_particle(Vector3f(x, 0, zl), Vector3f(0.f, 1.f, -1.f).normalized());
			add_boundary_particle(Vector3f(x, yl, zl), Vector3f(0.f, -1.f, -1.f).normalized());
		}

		// y borders
		for (float y = kernel_*0.5f; y < world_size_(0); y += (kernel_*0.5f))
		{
			add_boundary_particle(Vector3f(0, y, 0), Vector3f(1.f, 0.f, 1.f).normalized());
			add_boundary_particle(Vector3f(xl, y, 0), Vector3f(-1.f, 0.f, 1.f).normalized());
			add_boundary_particle(Vector3f(0, y, zl), Vector3f(1.f, 0.f, -1.f).normalized());
			add_boundary_particle(Vector3f(xl, y, zl), Vector3f(-1.f, 0.f, -1.f).normalized());
		}

		// z borders
		for (float z = kernel_*0.5f; z < world_size_(0); z += (kernel_*0.5f))
		{
			add_boundary_particle(Vector3f(0, 0, z), Vector3f(1.f, 1.f, 0.f).normalized());
			add_boundary_particle(Vector3f(xl, 0, z), Vector3f(-1.f, 1.f, 0.f).normalized());
			add_boundary_particle(Vector3f(0, yl, z), Vector3f(1.f, -1.f, 0.f).normalized());
			add_boundary_particle(Vector3f(xl, yl, z), Vector3f(-1.f, -1.f, 0.f).normalized());
		}

		// corners
		for (int i = 0; i < 8; ++i) {
			int x = (i)& 1;
			int y = (i >> 1) & 1;
			int z = (i >> 2) & 1;
			add_boundary_particle(Vector3f(x ? 0 : xl, y ? 0 : yl, z ? 0 : zl), Vector3f(x ? 1.f : -1.f, y ? 1.f : -1.f, z ? 1.f : -1.f).normalized());
		}
	}

	void SimulateSystem::add_boundary_particle(const Vector3f &pos, const Vector3f &normal)
	{
		Particle *p = &(boundary_particles_[num_boundary_particles_]);

		p->id = num_boundary_particles_;

		p->curr_pos = pos;

		p->force(0) = 0.0f;
		p->force(1) = 0.0f;
		p->force(2) = 0.0f;
		p->curr_vel(0) = 0.0f;
		p->curr_vel(1) = 0.0f;
		p->curr_vel(2) = 0.0f;

		p->dens = rest_dens_;
		p->pres = 0.0f;

		p->next = nullptr;

		num_boundary_particles_++;
	}

	void SimulateSystem::add_particle(const Vector3f &pos, const Vector3f &curr_vel)
	{
		Particle *p = &(particles_[num_particles_]);

		p->id = num_particles_;

		p->curr_pos = pos;
		p->curr_vel = curr_vel;

		p->force(0) = 0.0f;
		p->force(1) = 0.0f;
		p->force(2) = 0.0f;
		p->curr_vel(0) = 0.0f;
		p->curr_vel(1) = 0.0f;
		p->curr_vel(2) = 0.0f;

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
		init_densities();
		init_normals();
		comp_force();
		init_pres_pres_force();
		
		int iter = 0;
		while (max_dens_var_ > dens_var_limit_ || iter < min_iteration_)
		{
			if (max_dens_var_ > dens_var_limit_) {
				int i = 0;
			}
			pred_vel_pos();
			update_dens_var_scale();
			pred_dens_dens_var_update_pres();
			comp_pres_force();
			if (iter > max_iteration_)
			{
				break;
			}
			iter++;
		}

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
			hash = calc_cell_hash(calc_cell_pos(p->curr_pos));
			if (hash == (unsigned int)0xffffffff) {
				std::cout << p->curr_pos << std::endl;
			}
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

	void SimulateSystem::build_boundary_table()
	{
		Particle *p;
		unsigned int hash;

		for (unsigned int i = 0; i<total_cells_; i++)
		{
			boundary_cells_[i] = NULL;
		}

		for (unsigned int i = 0; i<num_boundary_particles_; i++)
		{
			p = &(boundary_particles_[i]);
			hash = calc_cell_hash(calc_cell_pos(p->curr_pos));
			if (boundary_cells_[hash] == NULL)
			{
				p->next = NULL;
				boundary_cells_[hash] = p;
			}
			else
			{
				p->next = boundary_cells_[hash];
				boundary_cells_[hash] = p;
			}
		}
	}

	void SimulateSystem::massify_boundary()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		float r2;

		for (unsigned int i = 0; i < num_boundary_particles_; i++)
		{
			float weight = 0.f;
			p = &(boundary_particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);

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

						np = boundary_cells_[hash];
						while (np != NULL)
						{
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								weight += pow(kernel2_ - r2, 3); // poly6 kernel
							}

							np = np->next;
						}
					}
				}
			}
			p->mass = rest_dens_ / (poly6_value_ * weight) / 1.17f;
		}
	}

	void SimulateSystem::init_densities() 
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		float r2;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			float fluid_term = 0.f;
			float boundary_term = 0.f;
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);

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
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								fluid_term += pow(kernel2_ - r2, 3) * mass_; // poly6 kernel
							}

							np = np->next;
						}

						// Consider boundary particles
						np = boundary_cells_[hash];
						while (np != NULL)
						{
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								boundary_term += pow(kernel2_ - r2, 3) * np->mass; // poly6 kernel
							}

							np = np->next;
						}
					}
				}
			}
			p->dens = poly6_value_ * (fluid_term + boundary_term);
			//std::cout << "fluid dens: " << p->dens << std::endl;
		}

		// init boundary particles density
		for (unsigned int i = 0; i < num_boundary_particles_; i++)
		{
			float fluid_term = 0.f;
			float boundary_term = 0.f;
			p = &(boundary_particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);

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
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								fluid_term += pow(kernel2_ - r2, 3) * mass_; // poly6 kernel
							}

							np = np->next;
						}

						// Consider boundary particles
						np = boundary_cells_[hash];
						while (np != NULL)
						{
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								boundary_term += pow(kernel2_ - r2, 3) * np->mass; // poly6 kernel
							}

							np = np->next;
						}
					}
				}
			}
			p->dens = poly6_value_ * (fluid_term + boundary_term);
			//std::cout << "boundary dens: " << p->dens << std::endl;
		}
	}

	void SimulateSystem::init_normals()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;
		float r2;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			Vector3f normal(0.f, 0.f, 0.f);
			cell_pos = calc_cell_pos(p->curr_pos);

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
							rel_pos = (p->curr_pos - np->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								normal += rel_pos * pow(kernel2_ - r2, 2) / np->dens; // poly6grad
							}
							np = np->next;
						}
					}
				}
			}
			normal *= kernel_ * mass_ * grad_poly6_;
			p->normal = normal;
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
		float half_V;

		float pres_kernel;
		float visc_kernel;

		float temp_force;

		Vector3f grad_color;
		float lplc_color;


		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);

			Vector3f viscosity(0.f, 0.f, 0.f);
			Vector3f cohesion(0.f, 0.f, 0.f);
			Vector3f curvature(0.f, 0.f, 0.f);

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
						near_pos = cell_pos + Vector3i(x, y, z);
						hash = calc_cell_hash(near_pos);

						if (hash == 0xffffffff)
						{
							continue;
						}

						np = cells_[hash];
						while (np != NULL)
						{
							rel_pos = (p->curr_pos - np->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2>LOWER_LIMIT)
							{
								r = sqrt(r2);
								half_V = mass_ / np->dens / 2;
								rel_vel = p->curr_vel - np->curr_vel;
								kernel_r = kernel_ - r; // Viscosity laplace

								viscosity -= rel_vel * kernel_r / np->dens;
								float kij = 2.f * rest_dens_ / (p->dens + np->dens);
								float surface_tension;
								if (r < kernel_ / 2) {
									surface_tension = 2.f * pow(kernel_ - r, 3) * pow(r, 3) + surf_offset_;
								}
								else {
									surface_tension = pow(r, 3) * pow(kernel_ - r, 3);
								}
								cohesion += kij * (rel_pos / r) * surface_tension;
								curvature += kij * (p->normal - np->normal);

								//float temp = (-1) * grad_poly6_ * half_V * pow(kernel2_ - r2, 2);
								//grad_color += temp * rel_pos;
								//lplc_color += lplc_poly6_ * half_V * (kernel2_ - r2) * (r2 - 3 / 4 * (kernel2_ - r2));
							}
							np = np->next;
						}
					}
				}
			}
			viscosity *=  visc_ * mass2_ * visco_value_ / p->dens;
			cohesion *= -surf_coef_ * mass2_ * surf_value_;
			curvature *= -surf_coef_ * mass_;

			Vector3f force = viscosity;// +cohesion + curvature; // Add cohesion and curvature
			force += mass_ * gravity_;

			p->force = force;

			//lplc_color += self_lplc_color_ / p->dens;
			//p->surf_norm = grad_color.norm();

			//if (p->surf_norm > surf_norm_)
			//{
			//	//p->force += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			//}
		}
	}

	void SimulateSystem::init_pres_pres_force()
	{
		Particle *p;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			p->pres_force = Vector3f(0.f, 0.f, 0.f);
			p->pres = 0.f;
		}
	}

	void SimulateSystem::pred_vel_pos()
	{
		Particle *p;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			Vector3f acc = 1 / mass_ * (p->force + p->pres_force);
			p->new_vel = p->curr_vel + acc * timestep_;
			p->new_pos = p->curr_pos + p->new_vel * timestep_;
		}
	}

	void SimulateSystem::pred_dens_dens_var_update_pres()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;

		float r2;
		max_dens_var_ = 0;
		avg_dens_var_ = 0;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);
			float fluid_dens = 0.f;
			float boundary_dens = 0.f;

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
							rel_pos = (p->new_pos - np->new_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								fluid_dens += pow(kernel2_ - r2, 3); // poly6 kernel
							}
							np = np->next;
						}

						// Consider boundary particles
						np = boundary_cells_[hash];
						while (np != NULL)
						{
							rel_pos = (p->curr_pos - np->new_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_) {
								//boundary_dens += pow(kernel2_ - r2, 3) * np->mass; // poly6 kernel
							}
							np = np->next;
						}
					}
				}
			}
			float dens = poly6_value_ * mass_ * fluid_dens + poly6_value_ * boundary_dens;

			//std::cout << dens << std::endl;
			float dens_var = std::max(0.f, dens - rest_dens_);
			max_dens_var_ = std::max(max_dens_var_, dens_var);
			avg_dens_var_ += dens_var;

			p->pres += dens_var_scale_ * dens_var;
			//std::cout << "p->pres: " << p->pres << std::endl;
		}
		avg_dens_var_ /= num_particles_;
	}

	void SimulateSystem::update_dens_var_scale()
	{
		Vector3f gradient_sum(0.f, 0.f, 0.f);
		float sum_squared_gradient = 0.f;
		for (float x = -kernel_ - radius_; x <= kernel_ + radius_; x += 2.f * radius_)
		{
			for (float y = -kernel_ - radius_; y <= kernel_ + radius_; y += 2.f * radius_)
			{
				for (float z = -kernel_ - radius_; z <= kernel_ + radius_; z += 2.f * radius_)
				{
					Vector3f r(x, y, z);
					float r2 = r.squaredNorm();
					if (r2 < kernel2_) {
						Vector3f gradient(grad_poly6_ * r * pow(kernel2_ - r2, 2)); // poly6grad
						gradient_sum += gradient;
						sum_squared_gradient += gradient.dot(gradient);
					}
				}
			}
		}
		float squared_sum_gradient = gradient_sum.dot(gradient_sum);
		float beta = 2.f * pow(mass_ * timestep_ / rest_dens_, 2);
		dens_var_scale_ = -1.f / (beta * (- squared_sum_gradient - sum_squared_gradient));
		//std::cout << "dens var scale: " << dens_var_scale_ << std::endl;
	}

	void SimulateSystem::comp_pres_force()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;

		float r2;

		//asdf
		float r;
		float V;
		float pres_kernel;
		float visc_kernel;
		float temp_force;
		float kernel_r;
		Vector3f rel_vel;
		Vector3f grad_color;
		float lplc_color;
		//asdf

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);
			Vector3f pres_force(0.f, 0.f, 0.f);

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
							rel_pos = (p->curr_pos - np->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > LOWER_LIMIT)
							{
								float r = sqrt(r2);
								pres_force -= mass2_ * (p->pres / pow(p->dens, 2) + np->pres / pow(np->dens, 2)) * spiky_value_ * rel_pos * (1.f / r) * pow(kernel_-r, 2); // spikygrad
							}
							np = np->next;
						}

						// Consider boundary particles
						np = boundary_cells_[hash];
						while (np != NULL)
						{
							rel_pos = (p->curr_pos - np->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > LOWER_LIMIT)
							{
								float r = sqrt(r2);
								//pres_force -= mass_ * np->mass * (p->pres / pow(p->dens, 2) + p->pres / pow(np->dens, 2)) * spiky_value_ * rel_pos * (1.f / r) * pow(kernel_ - r, 2); // spikygrad
							}
							np = np->next;
						}

					}
				}
			}
			p->pres_force = pres_force;
			//std::cout << "pres force: " << std::endl << p->pres_force << std::endl << "end pres force" << std::endl;
		}
	}

	/*void SimulateSystem::loop_through()
	{
		Particle *p;
		Particle *np;

		Vector3i cell_pos;
		Vector3i near_pos;
		unsigned int hash;

		Vector3f rel_pos;

		float r2;

		for (unsigned int i = 0; i < num_particles_; i++)
		{
			p = &(particles_[i]);
			cell_pos = calc_cell_pos(p->curr_pos);

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
							rel_pos = (np->curr_pos - p->curr_pos).cast<float>();
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > LOWER_LIMIT)
							{

							}
							np = np->next;
						}
					}
				}
			}
		}
	}*/

	/*void SimulateSystem::comp_force()
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
			cell_pos = calc_cell_pos(p->curr_pos);

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
							rel_pos = p->curr_pos - np->curr_pos;
							r2 = rel_pos.squaredNorm();

							if (r2 < kernel2_ && r2 > LOWER_LIMIT)
							{
								r = sqrt(r2);
								V = mass_ / np->dens / 2;
								kernel_r = kernel_ - r;

								pres_kernel = spiky_value_ * kernel_r * kernel_r;
								temp_force = V * (p->pres + np->pres) * pres_kernel;
								p->force -=rel_pos*temp_force / r;

								rel_vel = np->curr_vel - p->curr_vel;

								visc_kernel = visco_value_*(kernel_ - r);
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

			if (p->surf_norm > surf_norm_)
			{
				p->force += surf_coef_ * lplc_color * grad_color / p->surf_norm;
			}
		}
	}*/

	void SimulateSystem::integrate()
	{
		Particle *p;
		for (unsigned int i = 0; i<num_particles_; i++)
		{
			p = &(particles_[i]);
			//if (i == 83) {
			//	std::cout << p->pres_force << std::endl;
			//}

			p->curr_vel += (p->pres_force + p->force) * timestep_ / mass_;
			p->curr_pos += p->curr_vel * timestep_;

			if (p->curr_pos(0) >= world_size_(0) - BOUNDARY)
			{
				Vector3f n(-1.f, 0.f, 0.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(0) = world_size_(0) - BOUNDARY;
			}

			if (p->curr_pos(0) < 0.0f)
			{
				Vector3f n(1.f, 0.f, 0.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(0) = 0.0f;
			}

			if (p->curr_pos(1) >= world_size_(1) - BOUNDARY)
			{
				Vector3f n(0.f, -1.f, 0.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(1) = world_size_(1) - BOUNDARY;
			}

			if (p->curr_pos(1) < 0.0f)
			{
				Vector3f n(0.f, 1.f, 0.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(1) = 0.0f;
			}

			if (p->curr_pos(2) >= world_size_(2) - BOUNDARY)
			{
				Vector3f n(0.f, 0.f, -1.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(2) = world_size_(2) - BOUNDARY;
			}

			if (p->curr_pos(2) < 0.0f)
			{
				Vector3f n(0.f, 0.f, 1.f);
				p->curr_vel -= 1.5 * p->curr_vel.dot(n) * n;
				p->curr_pos(2) = 0.0f;
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