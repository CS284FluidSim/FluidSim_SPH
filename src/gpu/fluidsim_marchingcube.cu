#include <gl/freeglut.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_math.h>

#include "gpu/fluidsim_marchingcube.cuh"

namespace FluidSim {

	namespace gpu {

		__global__
			void init_grid_kernel(Particle *dev_particles, float *dev_scalar, float3* dev_pos, float3 *dev_normal, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			if (global_index < dev_param->tot_vox)
			{
				dev_pos[0].x = dev_param->isovalue;
				dev_pos[0].y = dev_param->isovalue;
				dev_pos[0].z = dev_param->isovalue;
			}

			//dev_pos[0].x = dev_param->edge_conn;
		}

		__global__
			void marching_cube_kernel(Particle *dev_particles, float *dev_scalar, float3* dev_pos, float3 *dev_normal, float3 *dev_vertex, float3  *dev_vertex_normal, MarchingCubeParam *dev_param)
		{
			uint global_index = blockIdx.x*blockDim.x + threadIdx.x;

			if (global_index < dev_param->tot_vox)
			{
				uint num_xy = global_index % (dev_param->dim_vox.x*dev_param->dim_vox.y);

				uint count_z = global_index / (dev_param->dim_vox.x*dev_param->dim_vox.y);
				uint count_y = num_xy / dev_param->dim_vox.x;
				uint count_x = num_xy%dev_param->dim_vox.x;

				float cube_value[8];
				float3 cube_pos[8];
				float3 cube_norm[8];

				float3 edge_vertex[12];
				float3 edge_norm[12];

				uint index = 0;
				uint prev = 0;
				uint next = 0;
				uint x = 0;
				uint y = 0;
				uint z = 0;

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

				__syncthreads();

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
						break;
					}

					for (uint count_point = 0; count_point < 3; count_point++)
					{
						index = dev_param->triangle_table[flag_index][3 * count_triangle + count_point];
						dev_vertex_normal[global_index] = make_float3(edge_norm[index].x, edge_norm[index].y, edge_norm[index].z);
						dev_vertex[global_index] = make_float3(edge_vertex[index].x*dev_param->sim_ratio.x + dev_param->origin.x,
							edge_vertex[index].y*dev_param->sim_ratio.y + dev_param->origin.y,
							edge_vertex[index].z*dev_param->sim_ratio.z + dev_param->origin.z);
					}
				}
			}
		}

		__host__
			MarchingCube::MarchingCube(Particle *dev_particles, uint3 dim_vox, float3 sim_ratio, float3 origin, float step, float isovalue)
		{
			param_ = new MarchingCubeParam();
			param_->dim_vox = dim_vox;
			param_->tot_vox = dim_vox.x*dim_vox.y*dim_vox.z;

			param_->origin = origin;
			param_->sim_ratio = sim_ratio;
			param_->step = step;

			param_->isovalue = isovalue;

			std::cout << sizeof(MarchingCubeParam) << std::endl;

			cudaMalloc(&dev_param_, sizeof(MarchingCubeParam));
			cudaMemcpy(dev_param_, param_, sizeof(MarchingCubeParam), cudaMemcpyHostToDevice);

			cudaMalloc(&dev_pos_, sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_scalar_, sizeof(float)*param_->tot_vox);
			cudaMalloc(&dev_normal_, sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_vertex_, sizeof(float3)*param_->tot_vox);
			cudaMalloc(&dev_vertex_normal_, sizeof(float3)*param_->tot_vox);
		}

		__host__
			MarchingCube::~MarchingCube()
		{
			delete param_;
			cudaFree(dev_param_);
			cudaFree(dev_pos_);
			cudaFree(dev_scalar_);
			cudaFree(dev_normal_);
			cudaFree(dev_vertex_);
			cudaFree(dev_vertex_normal_);
		}

		__host__
			void MarchingCube::run()
		{
			uint num_threads;
			uint num_blocks;

			calc_grid_size(param_->tot_vox, 512, num_blocks, num_threads);
			
			init_grid_kernel << <num_blocks, num_threads >> > (dev_particles_, dev_scalar_, dev_pos_, dev_normal_, dev_param_);

			//Debug
			float3 tmp[100];
			cudaMemcpy(tmp, dev_pos_, 100, cudaMemcpyDeviceToHost);
			for (int i = 0; i < 100; ++i)
				std::cout << tmp[i].x << ", " << tmp[i].y << ", " << tmp[i].z << std::endl;

			marching_cube_kernel << <num_blocks, num_threads >> > (dev_particles_, dev_scalar_, dev_pos_, dev_normal_, dev_vertex_, dev_vertex_normal_, dev_param_);
		}

		__host__
			void MarchingCube::render()
		{

		}
	}
}