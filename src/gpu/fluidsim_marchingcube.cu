#include <gl/freeglut.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_math.h>

#include <thrust\copy.h>
#include <thrust\device_ptr.h>

#include "gpu/fluidsim_marchingcube.cuh"

namespace FluidSim {

	namespace gpu {

		__global__
			void init_grid_kernel(float *dev_scalar, float3* dev_pos, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			if (global_index < dev_param->tot_vox)
			{
				uint num_xy = global_index % (dev_param->dim_vox.x*dev_param->dim_vox.y);

				uint count_z = global_index / (dev_param->dim_vox.x*dev_param->dim_vox.y);
				uint count_y = num_xy / dev_param->dim_vox.x;
				uint count_x = num_xy % dev_param->dim_vox.x;

				dev_pos[global_index].x = count_x*dev_param->step;
				dev_pos[global_index].y = count_y*dev_param->step;
				dev_pos[global_index].z = count_z*dev_param->step;

				dev_scalar[global_index] = 0.f;
			}
		}

		__global__
			void calc_scalar_kernel(Particle *dev_particles, float *dev_scalar, float3* dev_pos, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			float radius = 0.02f;
			float vox_size = dev_param->step;

			if (global_index < dev_param->tot_particles)
			{
				int cell_pos_x = (dev_particles[global_index].pos.x) / vox_size;
				int cell_pos_y = (dev_particles[global_index].pos.y) / vox_size;
				int cell_pos_z = (dev_particles[global_index].pos.z) / vox_size;
				for (float x = -radius; x < radius; x+=vox_size)
				{
					for (float y = -radius; y < radius; y+=vox_size)
					{
						for (float z = -radius; z < radius; z+=vox_size)
						{
							int pos_x = cell_pos_x + x / vox_size;
							int pos_y = cell_pos_y + y / vox_size;
							int pos_z = cell_pos_z + z / vox_size;
							if (pos_x < 0 || pos_x >= dev_param->dim_vox.x ||
								pos_y < 0 || pos_y >= dev_param->dim_vox.y ||
								pos_z < 0 || pos_z >= dev_param->dim_vox.z)
								continue;
							else
							{
								int index = pos_z*dev_param->dim_vox.x*dev_param->dim_vox.y + pos_y*dev_param->dim_vox.x + pos_x;
								dev_scalar[index] = dev_particles[global_index].dens;
							}
						}
					}
				}
			}
		}

		__global__
			void calc_normal_kernel(float *dev_scalar, float3* dev_pos, float3 *dev_normal, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			if (global_index < dev_param->tot_vox)
			{
				uint num_xy = global_index % (dev_param->dim_vox.x*dev_param->dim_vox.y);

				uint count_z = global_index / (dev_param->dim_vox.x*dev_param->dim_vox.y);
				uint count_y = num_xy / dev_param->dim_vox.x;
				uint count_x = num_xy % dev_param->dim_vox.x;

				uint prev = 0;
				uint next = 0;
	

				if (count_x == 0)
				{
					prev = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x + 1;
					dev_normal[global_index].x = (dev_scalar[prev] - 0.0f) / dev_param->step;
				}

				if (count_x == dev_param->dim_vox.x - 1)
				{
					next = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x - 1;
					dev_normal[global_index].x = (0.0f - dev_scalar[next]) / dev_param->step;
				}

				if (count_x != 0 && count_x != dev_param->dim_vox.x - 1)
				{
					prev = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x + 1;
					next = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x - 1;
					dev_normal[global_index].x = (dev_scalar[prev] - dev_scalar[next]) / dev_param->step;
				}

				if (count_y == 0)
				{
					prev = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + (count_y + 1)*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].y = (dev_scalar[prev] - 0.0f) / dev_param->step;
				}

				if (count_y == dev_param->dim_vox.y - 1)
				{
					next = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + (count_y - 1)*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].y = (0.0f - dev_scalar[next]) / dev_param->step;
				}

				if (count_y != 0 && count_y != dev_param->dim_vox.y - 1)
				{
					prev = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + (count_y + 1)*dev_param->dim_vox.x + count_x;
					next = count_z*dev_param->dim_vox.x*dev_param->dim_vox.y + (count_y - 1)*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].y = (dev_scalar[prev] - dev_scalar[next]) / dev_param->step;
				}

				if (count_z == 0)
				{
					prev = (count_z + 1)*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].z = (dev_scalar[prev] - 0.0f) / dev_param->step;
				}

				if (count_z == dev_param->dim_vox.z - 1)
				{
					next = (count_z - 1)*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].z = (0.0f - dev_scalar[next]) / dev_param->step;
				}

				if (count_z != 0 && count_z != dev_param->dim_vox.z - 1)
				{
					prev = (count_z + 1)*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x;
					next = (count_z - 1)*dev_param->dim_vox.x*dev_param->dim_vox.y + count_y*dev_param->dim_vox.x + count_x;
					dev_normal[global_index].z = (dev_scalar[prev] - dev_scalar[next]) / dev_param->step;
				}

				float norm = -sqrt(dev_normal[global_index].x*dev_normal[global_index].x + dev_normal[global_index].y*dev_normal[global_index].y + dev_normal[global_index].z*dev_normal[global_index].z);
				if (norm == 0.0f)
				{
					dev_normal[global_index].x = 0.0f;
					dev_normal[global_index].y = 0.0f;
					dev_normal[global_index].z = 0.0f;
				}
				else
				{
					dev_normal[global_index].x = dev_normal[global_index].x / norm;
					dev_normal[global_index].y = dev_normal[global_index].y / norm;
					dev_normal[global_index].z = dev_normal[global_index].z / norm;
				}
			}
		}

		__global__
			void marching_cube_kernel(float *dev_scalar, float3* dev_pos, float3 *dev_normal, Triangle *dev_tri, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			if (global_index < dev_param->tot_vox)
			{
				uint num_xy = global_index % (dev_param->dim_vox.x*dev_param->dim_vox.y);

				uint count_z = global_index / (dev_param->dim_vox.x*dev_param->dim_vox.y);
				uint count_y = num_xy / dev_param->dim_vox.x;
				uint count_x = num_xy % dev_param->dim_vox.x;

				float cube_value[8];
				float3 cube_pos[8];
				float3 cube_norm[8];

				float3 edge_vertex[12];
				float3 edge_norm[12];

				uint index = 0;
				uint x = 0;
				uint y = 0;
				uint z = 0;

				if (count_x >= dev_param->dim_vox.x - 1 || count_y >= dev_param->dim_vox.y - 1 || count_z >= dev_param->dim_vox.z - 1)
					return;
				
				int flag_index;
				int edge_flags;
				for (uint count = 0; count < 8; count++)
				{
					x = count_x + dev_param->vertex_offset[count][0];
					y = count_y + dev_param->vertex_offset[count][1];
					z = count_z + dev_param->vertex_offset[count][2];

					index = z*dev_param->dim_vox.x*dev_param->dim_vox.y + y*dev_param->dim_vox.x + x;
					cube_value[count] = dev_scalar[index];
					cube_pos[count] = dev_pos[index];
					cube_norm[count] = dev_normal[index];
				}
				
				flag_index = 0;
				for (uint count = 0; count < 8; count++)
				{
					if (cube_value[count] < dev_param->isovalue)
					{
						flag_index |= 1 << count;
					}
				}
		
				edge_flags = dev_param->cube_edge_flags[flag_index];
				if (edge_flags == 0)
				{
					return;
				}
			
				for (uint count = 0; count < 12; count++)
				{
					if (edge_flags & (1 << count))
					{
						float diff = (dev_param->isovalue - cube_value[dev_param->edge_conn[count][0]]) / (cube_value[dev_param->edge_conn[count][1]] - cube_value[dev_param->edge_conn[count][0]]);

						edge_vertex[count].x = cube_pos[dev_param->edge_conn[count][0]].x + (cube_pos[dev_param->edge_conn[count][1]].x - cube_pos[dev_param->edge_conn[count][0]].x) * diff;
						edge_vertex[count].y = cube_pos[dev_param->edge_conn[count][0]].y + (cube_pos[dev_param->edge_conn[count][1]].y - cube_pos[dev_param->edge_conn[count][0]].y) * diff;
						edge_vertex[count].z = cube_pos[dev_param->edge_conn[count][0]].z + (cube_pos[dev_param->edge_conn[count][1]].z - cube_pos[dev_param->edge_conn[count][0]].z) * diff;

						edge_norm[count].x = cube_norm[dev_param->edge_conn[count][0]].x + (cube_norm[dev_param->edge_conn[count][1]].x - cube_norm[dev_param->edge_conn[count][0]].x) * diff;
						edge_norm[count].y = cube_norm[dev_param->edge_conn[count][0]].y + (cube_norm[dev_param->edge_conn[count][1]].y - cube_norm[dev_param->edge_conn[count][0]].y) * diff;
						edge_norm[count].z = cube_norm[dev_param->edge_conn[count][0]].z + (cube_norm[dev_param->edge_conn[count][1]].z - cube_norm[dev_param->edge_conn[count][0]].z) * diff;

					}
				}

				for (uint count_triangle = 0; count_triangle < 5; count_triangle++)
				{

					if (dev_param->triangle_table[flag_index][3 * count_triangle] < 0)
					{
						return;
					}

					uint idx = 5 * global_index+count_triangle;
					dev_tri[idx].valid = 1.f;

					for (uint count_point = 0; count_point < 3; count_point++)
					{
						index = dev_param->triangle_table[flag_index][3 * count_triangle + count_point];
						dev_tri[idx].n[count_point] = edge_norm[index];
						dev_tri[idx].v[count_point] = make_float3(edge_vertex[index].x*dev_param->sim_ratio.x + dev_param->origin.x,
							edge_vertex[index].y*dev_param->sim_ratio.y + dev_param->origin.y,
							edge_vertex[index].z*dev_param->sim_ratio.z + dev_param->origin.z);
					}
				}
			}
		}

		__host__
			MarchingCube::MarchingCube(uint3 dim_vox, float3 sim_ratio, float3 origin, float step, float isovalue)
		{
			param_ = new MarchingCubeParam();
			param_->dim_vox = dim_vox;
			param_->tot_vox = dim_vox.x*dim_vox.y*dim_vox.z;
			param_->origin = origin;
			param_->sim_ratio = sim_ratio;
			param_->step = step;
			param_->isovalue = isovalue;

			tri_ = (Triangle *)malloc(sizeof(Triangle) * param_->tot_vox*5);
			scalar_ = (float *)malloc(sizeof(float)*param_->tot_vox);
			normal_ = (float3 *)malloc(sizeof(float3)*param_->tot_vox);
			pos_ = (float3 *)malloc(sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_param_, sizeof(MarchingCubeParam));
			cudaMalloc(&dev_pos_, sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_scalar_, sizeof(float)*param_->tot_vox);
			cudaMalloc(&dev_normal_, sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_tri_, sizeof(Triangle)*param_->tot_vox*5);
			cudaMalloc(&dev_tri_non_empty, sizeof(Triangle)*param_->tot_vox*5);
		}

		__host__
			MarchingCube::~MarchingCube()
		{
			free(tri_);
			free(scalar_);
			free(normal_);
			free(pos_);
			delete param_;
			cudaFree(dev_param_);
			cudaFree(dev_pos_);
			cudaFree(dev_scalar_);
			cudaFree(dev_normal_);
			cudaFree(dev_tri_);
			cudaFree(dev_tri_non_empty);
		}

		__host__
			void MarchingCube::init(uint num_particles)
		{
			param_->tot_particles = num_particles;
			cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);

			uint num_threads;
			uint num_blocks;

			calc_grid_size(param_->tot_vox, 512, num_blocks, num_threads);

			init_grid_kernel << < num_blocks, num_threads >> > (dev_scalar_, dev_pos_, dev_param_);
		}

		__host__
			void MarchingCube::compute(Particle *dev_particles)
		{
			uint num_threads;
			uint num_blocks;

			calc_grid_size(param_->tot_vox, 512, num_blocks, num_threads);

			cudaMemset(dev_scalar_, 0, sizeof(float)*param_->tot_vox);
			cudaMemset(dev_normal_, 0, sizeof(float3)*param_->tot_vox);
			cudaMemset(dev_tri_, 0, sizeof(Triangle)*param_->tot_vox*5);

			calc_scalar_kernel << <num_blocks, num_threads >> > (dev_particles, dev_scalar_, dev_pos_, dev_param_);

			calc_normal_kernel << <num_blocks, num_threads >> > (dev_scalar_, dev_pos_, dev_normal_, dev_param_);

			marching_cube_kernel << <num_blocks, num_threads >> > (dev_scalar_, dev_pos_, dev_normal_, dev_tri_, dev_param_);
		}

		__host__
			void MarchingCube::render(RenderMode rm)
		{
			if (rm == RenderMode::TRI)
			{
				memset(tri_, 0, sizeof(Triangle)*param_->tot_vox * 5);
				cudaMemset(dev_tri_non_empty, 0, sizeof(Triangle)*param_->tot_vox * 5);
				thrust::copy_if(thrust::device_ptr<Triangle>(dev_tri_), thrust::device_ptr<Triangle>(dev_tri_ + param_->tot_vox * 5), thrust::device_ptr<Triangle>(dev_tri_non_empty), is_non_empty_tri());
				cudaMemcpy(tri_, dev_tri_non_empty, sizeof(Triangle)*param_->tot_vox * 5, cudaMemcpyDeviceToHost);
				is_non_empty_tri non_empty;

				for (int i = 0; i < param_->tot_vox * 5; i++)
				{
					if (!non_empty(tri_[i]))
						break;
					else
					{
						glBegin(GL_TRIANGLES);
						for (int j = 0; j < 3; ++j)
						{
							glNormal3f(tri_[i].n[j].x, tri_[i].n[j].y, tri_[i].n[j].z);
							glVertex3f(tri_[i].v[j].x, tri_[i].v[j].y, tri_[i].v[j].z);
						}
						glEnd();
					}
				}
			}
			else if (rm == RenderMode::SCALAR)
			{
				cudaMemcpy(scalar_, dev_scalar_, sizeof(float)*param_->tot_vox, cudaMemcpyDeviceToHost);
				cudaMemcpy(pos_, dev_pos_, sizeof(float3)*param_->tot_vox, cudaMemcpyDeviceToHost);
				glBegin(GL_POINTS);
				float c = 0.f;
				for (int i = 0; i < param_->tot_vox; i++)
				{
					if (scalar_[i] != 0.f)
					{
						c = scalar_[i] / 3000.f;
						glColor3f(c, c, c);
						glVertex3f(pos_[i].x*param_->sim_ratio.x + param_->origin.x,
							pos_[i].y*param_->sim_ratio.y + param_->origin.y,
							pos_[i].z*param_->sim_ratio.z + param_->origin.z);
					}
				}
				glEnd();
			}
			else if (rm == RenderMode::NORMAL)
			{
				is_non_zero_float3f non_zero;
				cudaMemcpy(normal_, dev_normal_, sizeof(float3)*param_->tot_vox, cudaMemcpyDeviceToHost);
				cudaMemcpy(pos_, dev_pos_, sizeof(float3)*param_->tot_vox, cudaMemcpyDeviceToHost);
				glBegin(GL_POINTS);
				float c = 0.f;
				for (int i = 0; i < param_->tot_vox; i++)
				{
					if (non_zero(normal_[i]))
					{
						auto tmp = (normal_[i] + 1.f) / 2.f;
						glColor3f(tmp.x,tmp.y,tmp.z);
						glVertex3f(pos_[i].x*param_->sim_ratio.x + param_->origin.x,
							pos_[i].y*param_->sim_ratio.y + param_->origin.y,
							pos_[i].z*param_->sim_ratio.z + param_->origin.z);
					}
				}
				glEnd();
			}
			else if (rm == RenderMode::POS)
			{
				cudaMemcpy(pos_, dev_pos_, sizeof(float3)*param_->tot_vox, cudaMemcpyDeviceToHost);
				glBegin(GL_POINTS);
				float c = 0.f;
				for (int i = 0; i < param_->tot_vox; i++)
				{
					glVertex3f(pos_[i].x*param_->sim_ratio.x + param_->origin.x,
						pos_[i].y*param_->sim_ratio.y + param_->origin.y,
						pos_[i].z*param_->sim_ratio.z + param_->origin.z);
				}
				glEnd();
			}
		}
	}
}