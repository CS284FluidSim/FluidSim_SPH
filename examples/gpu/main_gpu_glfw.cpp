#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "gpu/fluidsim_system.cuh"
#include "gpu/fluidsim_marchingcube.cuh"
#include "fluidsim_gl_utils.h"
#include "fluidsim_shader.h"

/* choose pure reflection or pure refraction here. */

#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "glew32.lib") 

//FluidSim
FluidSim::gpu::SimulateSystem *simsystem;

FluidSim::Shader water_shader;
FluidSim::Shader skybox_shader;
FluidSim::Shader texcube_shader;
FluidSim::Shader diffuse_shader;
FluidSim::Shader particle_shader;
FluidSim::Shader refract_shader;

// camera matrices. it's easier if they are global
mat4 view_mat;
mat4 proj_mat;
vec3 cam_pos(1.0f, 2.0f, 3.0f);

// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 800;
int g_gl_height = 800;
GLFWwindow *g_window = NULL;


int primitive_mode = 0;

// render mode
int render_mode = 0;
bool render_mesh = false;
bool pause = false;
bool step = false;


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		if (primitive_mode == 0)
		{
			//simsystem->change_mass(0.01f);
			simsystem->add_fluid(make_float3(0.45f, 0.8f, 0.45f), make_float3(0.55f, 1.0f, 0.55f), make_float3(0.0f, -2.0f, 0.0f));  // drop cube
		}
		else if (primitive_mode == 1)
		{
			//simsystem->change_mass(0.01f);
			simsystem->add_fluid(make_float3(0.5f, 0.8f, 0.5f), 0.1f, make_float3(0.0f, -2.0f, 0.0f));  // drop sphere
		}
		else if (primitive_mode == 2)
		{
			//simsystem->change_mass(0.01f);
			simsystem->add_fluid(make_float3(1.3f, 1.3f, 1.3f));  // a bunny drop
		}
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
		render_mesh = !render_mesh;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
		render_mode = (render_mode + 1) % 4;
}

void init_sph_system(std::string config_path)
{

	if (config_path != "")
	{
		cv::FileStorage fs;
		fs.open(config_path, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cout << "Error: cannot open configuration file" << std::endl;
			exit(0);
		}

		float3 world_size, real_world_side, real_world_origin, sim_ratio;

		fs["world_size.x"] >> world_size.x;
		fs["world_size.y"] >> world_size.y;
		fs["world_size.z"] >> world_size.z;
		real_world_side = { world_size.x * 1.0f, world_size.y * 1.0f, world_size.z * 1.0f };
		cam_pos.v[0] = world_size.x / 2;
		cam_pos.v[1] = world_size.y;
		cam_pos.v[2] = world_size.z * 4;
		real_world_origin = { 0.f,0.f,0.f };
		sim_ratio = real_world_side / world_size;

		int max_particles = 500000;
		fs["max_particles"] >> max_particles;
		float h = 0.04f;
		fs["h"] >> h;
		float mass = 0.02f;
		fs["mass"] >> mass;
		float3 gravity = { 0.f,-9.8f,0.f };
		fs["gravity.x"] >> gravity.x;
		fs["gravity.y"] >> gravity.y;
		fs["gravity.z"] >> gravity.z;
		float bound_damping = -0.5f;
		fs["bound_damping"] >> bound_damping;
		float rest_dens = 1000.f;
		fs["rest_dens"] >> rest_dens;
		float gas_const = 1.f;
		fs["gas_const"] >> gas_const;
		float visc = 6.5f;
		fs["visc"] >> visc;
		float timestep = 0.002f;
		fs["timestep"] >> timestep;
		float surf_norm = 3.f;
		fs["surf_norm"] >> surf_norm;
		float surf_coef = 0.2f;
		fs["surf_coef"] >> surf_coef;
		simsystem = new FluidSim::gpu::SimulateSystem(world_size, sim_ratio, real_world_origin, max_particles,
			h, mass, gravity, bound_damping, rest_dens, gas_const, visc, timestep, surf_norm, surf_coef);

		float3 cube_min;
		fs["cube_min.x"] >> cube_min.x;
		fs["cube_min.y"] >> cube_min.y;
		fs["cube_min.z"] >> cube_min.z;
		float3 cube_max;
		fs["cube_max.x"] >> cube_max.x;
		fs["cube_max.y"] >> cube_max.y;
		fs["cube_max.z"] >> cube_max.z;

		float gap;
		fs["gap"] >> gap;
		//simsystem->add_cube_fluid(cube_min, cube_max, gap);

		//simsystem->add_cube_fluid(cube_min, cube_max);

		//simsystem->add_cube_fluid(make_float3(0.5f, 0.5f, 0.5f), make_float3(0.6f, 0.6f, 0.6f));

		simsystem->add_cube_fluid(make_float3(0.8f, 0.0f, 0.0f), make_float3(1.0f, 0.9f, 1.0f), gap);

		//simsystem->add_cube_fluid(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 0.2f, 1.0f), gap);

		//simsystem->add_fluid(make_float3(0.2f, 0.5f, 0.1f), make_float3(0.7f, 0.9f, 0.9f));  // a cube drop from the air

		//simsystem->add_fluid(make_float3(0.5f, 0.7f, 0.5f), 0.3f);  // a sphere drop from the air

		//simsystem->add_fluid(make_float3(0.6f, 0.8f, 0.6f), make_float3(0.7f, 1.0f, 0.7f), make_float3(0.0f, -2.0f, 0.0f));

		//simsystem->add_fluid(make_float3(0.5f, 0.8f, 0.5f), 0.1f, make_float3(0.0f, -2.0f, 0.0f));

		//simsystem->add_fluid(make_float3(1.3f, 1.3f, 1.3f));  // a bunny drop

		//simsystem->add_fluid(make_float3(1.5f, 1.5f, 1.5f));  // a bunny drop
		//FluidSim::Cube *cube = new FluidSim::Cube({ 0.5f*world_size.x,0.2f*world_size.y,0.5f*world_size.z },
		//{ 0.2f*world_size.x,0.4f*world_size.y,0.5f*world_size.z });
		//cube->set_shader(&diffuse_shader);
		//FluidSim::Cube *cube1 = new FluidSim::Cube({ 0.2f*world_size.x,0.4f*world_size.y,0.2f*world_size.z },
		//{ 0.2f*world_size.x,0.8f*world_size.y,0.2f*world_size.z });
		//cube1->set_shader(&diffuse_shader);
		//FluidSim::Sphere *sphere = new FluidSim::Sphere({ 0.5f*world_size.x,0.2f*world_size.y,0.5f*world_size.z }, 0.2f*world_size.z);
		//sphere->set_shader(&diffuse_shader);
		// texture cube
		//GLuint box_texture;
		GLuint box_texture;
		create_cube_map("../texture/cube/wood/", &box_texture);
		GLuint frame_texture;
		create_cube_map("../texture/cube/frame/", &frame_texture);
		FluidSim::TexturedCube *tex_cube = new FluidSim::TexturedCube({ 0.5f*world_size.x,0.2f*world_size.y,0.5f*world_size.z },
		{ 0.2f*world_size.x,0.4f*world_size.y,0.5f*world_size.z }, box_texture);
		tex_cube->set_shader(&texcube_shader);
		FluidSim::TexturedCube *bottomwall = new FluidSim::TexturedCube({ 0.5f*world_size.x,0.f,0.5f*world_size.z},
		{ world_size.x, 0.01f, world_size.z}, frame_texture);
		bottomwall->set_shader(&texcube_shader);
		FluidSim::TexturedCube *backwall = new FluidSim::TexturedCube({ 0.5f*world_size.x,0.5f*world_size.y,0.f},
		{ world_size.x, world_size.y, 0.01f }, frame_texture);
		backwall->set_shader(&texcube_shader);
		FluidSim::TexturedCube *leftwall = new FluidSim::TexturedCube({ 0.f,0.5f*world_size.y,0.5f*world_size.z },
		{ 0.01f, world_size.y, world_size.z }, frame_texture);
		leftwall->set_shader(&texcube_shader);
		FluidSim::TexturedCube *rightwall = new FluidSim::TexturedCube({ world_size.x,0.5f*world_size.y,0.5f*world_size.z },
		{ 0.01f, world_size.y, world_size.z }, frame_texture);
		rightwall->set_shader(&texcube_shader);
		//Model *model = new Model("../scene/bunny.txt", { 0.5f*world_size.x,0.5f*world_size.y,0.5f*world_size.z }, 0.1f);
		//simsystem->add_static_object(cube);
		//simsystem->add_static_object(cube);
		//simsystem->add_static_object(cube1);
		simsystem->add_static_object(tex_cube);
		simsystem->add_static_object(bottomwall, false);
		simsystem->add_static_object(backwall, false);
		simsystem->add_static_object(leftwall, false);
		simsystem->add_static_object(rightwall, false);
	}
}

int main() {
	/*--------------------------------START
	* OPENGL--------------------------------*/
	restart_gl_log();
	// start GL context and O/S window using the GLFW helper library
	start_gl();

	init_sph_system("config.yaml");

	/*---------------------------------CUBE
	* MAP-----------------------------------*/
	GLuint cube_vao = make_cube(60.f);
	GLuint cube_map_texture;
	create_cube_map("../texture/cube/bridge/", &cube_map_texture);

	/*-------------------------------CREATE
	* SHADERS-------------------------------*/
	// shaders for "Suzanne" mesh
	water_shader.create("../shader/water_vs.glsl", "../shader/water_fs.glsl");
	water_shader.add_uniform("M");
	water_shader.add_uniform("V");
	water_shader.add_uniform("P");

	diffuse_shader.create("../shader/diffuse_vs.glsl", "../shader/diffuse_fs.glsl");
	diffuse_shader.add_uniform("M");
	diffuse_shader.add_uniform("V");
	diffuse_shader.add_uniform("P");

	skybox_shader.create("../shader/cube_vs.glsl", "../shader/cube_fs.glsl");
	skybox_shader.add_uniform("V");
	skybox_shader.add_uniform("P");

	texcube_shader.create("../shader/texcube_vs.glsl", "../shader/texcube_fs.glsl");
	texcube_shader.add_uniform("M");
	texcube_shader.add_uniform("V");
	texcube_shader.add_uniform("P");

	refract_shader.create("../shader/refract_vs.glsl", "../shader/refract_fs.glsl");
	refract_shader.add_uniform("M");
	refract_shader.add_uniform("V");
	refract_shader.add_uniform("P");

	particle_shader.create("../shader/particle_vs.glsl", "../shader/particle_fs.glsl");
	particle_shader.add_uniform("M");
	particle_shader.add_uniform("V");
	particle_shader.add_uniform("P");

	/*-------------------------------CREATE CAMERA--------------------------------*/
#define ONE_DEG_IN_RAD ( 2.0 * M_PI ) / 360.0 // 0.017444444
	// input variables
	float fnear = 0.1f;																		 // clipping plane
	float ffar = 100.0f;																		 // clipping plane
	float fovy = 45.0f;																		 // 67 degrees
	float aspect = (float)g_gl_width / (float)g_gl_height; // aspect ratio
	proj_mat = perspective(fovy, aspect, fnear, ffar);

	float cam_speed = 3.0f;					 // 1 unit per second
	float cam_heading_speed = 50.0f; // 30 degrees per second
	float cam_heading = 0.0f;				 // y-rotation in degrees
	mat4 T = translate(identity_mat4(),
		vec3(-cam_pos.v[0], -cam_pos.v[1], -cam_pos.v[2]));
	mat4 R = rotate_x_deg(identity_mat4(), -cam_heading);
	versor q = quat_from_axis_deg(-cam_heading, 0.0f, 1.0f, 0.0f);
	view_mat = R * T;
	// keep track of some useful vectors that can be used for keyboard movement
	vec4 fwd(0.0f, 0.0f, -1.0f, 0.0f);
	vec4 rgt(1.0f, 0.0f, 0.0f, 0.0f);
	vec4 up(0.0f, 1.0f, 0.0f, 0.0f);

	/*---------------------------SET RENDERING
	* DEFAULTS---------------------------*/
	// unique model matrix for each sphere
	mat4 model_mat = identity_mat4();
	
	water_shader.set_uniform_matrix4fv("M", model_mat.m);
	water_shader.set_uniform_matrix4fv("V", view_mat.m);
	water_shader.set_uniform_matrix4fv("P", proj_mat.m);

	diffuse_shader.set_uniform_matrix4fv("M", model_mat.m);
	diffuse_shader.set_uniform_matrix4fv("V", view_mat.m);
	diffuse_shader.set_uniform_matrix4fv("P", proj_mat.m);

	texcube_shader.set_uniform_matrix4fv("M", model_mat.m);
	texcube_shader.set_uniform_matrix4fv("V", view_mat.m);
	texcube_shader.set_uniform_matrix4fv("P", proj_mat.m);

	refract_shader.set_uniform_matrix4fv("M", model_mat.m);
	refract_shader.set_uniform_matrix4fv("V", view_mat.m);
	refract_shader.set_uniform_matrix4fv("P", proj_mat.m);

	skybox_shader.set_uniform_matrix4fv("V", R.m);
	skybox_shader.set_uniform_matrix4fv("P", proj_mat.m);

	particle_shader.set_uniform_matrix4fv("M", model_mat.m);
	particle_shader.set_uniform_matrix4fv("V", view_mat.m);
	particle_shader.set_uniform_matrix4fv("P", proj_mat.m);

	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS);		 // depth-testing interprets a smaller value as "closer"
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.2, 0.2, 0.2, 1.0); // grey background to help spot mistakes
	glViewport(0, 0, g_gl_width, g_gl_height);

	/*-------------------------------RENDERING
	* LOOP-------------------------------*/
	while (!glfwWindowShouldClose(g_window)) {
		// update timers
		static double previous_seconds = glfwGetTime();
		double current_seconds = glfwGetTime();
		double elapsed_seconds = current_seconds - previous_seconds;
		previous_seconds = current_seconds;
		_update_fps_counter(g_window);

		// wipe the drawing surface clear
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// render skybox
		skybox_shader.use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map_texture);
		glBindVertexArray(cube_vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);

		// run simulation
		if (simsystem->is_running())
		{
			if (!pause||step)
			{
				simsystem->animation();
				step = !step;
			}
			if (!render_mesh)
			{
				glPointSize(6.f);
				particle_shader.use();
				simsystem->render_particles((FluidSim::gpu::SimulateSystem::RenderMode)render_mode);
			}
			else
			{
				water_shader.use();
				simsystem->render_surface(FluidSim::gpu::MarchingCube::RenderMode::TRI);
			}
			simsystem->render_static_object();
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map_texture);
		}

		glfwSetMouseButtonCallback(g_window, mouse_button_callback);

		if (glfwGetKey(g_window, GLFW_KEY_0)) {
			primitive_mode = 0;
		}
		if (glfwGetKey(g_window, GLFW_KEY_1)) {
			primitive_mode = 1;
		}
		if (glfwGetKey(g_window, GLFW_KEY_2)) {
			primitive_mode = 2;
		}

		// render skybox
		skybox_shader.use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map_texture);
		glBindVertexArray(cube_vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);

		// update other events like input handling
		glfwPollEvents();

		// control keys
		bool cam_moved = false;
		vec3 move(0.0, 0.0, 0.0);
		float cam_yaw = 0.0f; // y-rotation in degrees
		float cam_pitch = 0.0f;
		float cam_roll = 0.0;
		if (glfwGetKey(g_window, GLFW_KEY_A)) {
			move.v[0] -= cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_D)) {
			move.v[0] += cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_Q)) {
			move.v[1] += cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_E)) {
			move.v[1] -= cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_W)) {
			move.v[2] -= cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_S)) {
			move.v[2] += cam_speed * elapsed_seconds;
			cam_moved = true;
		}
		if (glfwGetKey(g_window, GLFW_KEY_LEFT)) {
			cam_yaw += cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_yaw = quat_from_axis_deg(cam_yaw, up.v[0], up.v[1], up.v[2]);
			q = q_yaw * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_RIGHT)) {
			cam_yaw -= cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_yaw = quat_from_axis_deg(cam_yaw, up.v[0], up.v[1], up.v[2]);
			q = q_yaw * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_UP)) {
			cam_pitch += cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_pitch =
				quat_from_axis_deg(cam_pitch, rgt.v[0], rgt.v[1], rgt.v[2]);
			q = q_pitch * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_DOWN)) {
			cam_pitch -= cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_pitch =
				quat_from_axis_deg(cam_pitch, rgt.v[0], rgt.v[1], rgt.v[2]);
			q = q_pitch * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_Z)) {
			cam_roll -= cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_roll = quat_from_axis_deg(cam_roll, fwd.v[0], fwd.v[1], fwd.v[2]);
			q = q_roll * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_C)) {
			cam_roll += cam_heading_speed * elapsed_seconds;
			cam_moved = true;
			versor q_roll = quat_from_axis_deg(cam_roll, fwd.v[0], fwd.v[1], fwd.v[2]);
			q = q_roll * q;
		}
		if (glfwGetKey(g_window, GLFW_KEY_SPACE)){
			simsystem->start();
		}
		if (glfwGetKey(g_window, GLFW_KEY_P))
		{
			pause = !pause;
		}
		if (glfwGetKey(g_window, GLFW_KEY_N))
		{
			step = !step;
		}
		if (glfwGetKey(g_window, GLFW_KEY_R))
		{
			simsystem->reset();
		}
		// update view matrix
		if (cam_moved) {
			cam_heading += cam_yaw;

			// re-calculate local axes so can move fwd in dir cam is pointing
			R = quat_to_mat4(q);
			fwd = R * vec4(0.0, 0.0, -1.0, 0.0);
			rgt = R * vec4(1.0, 0.0, 0.0, 0.0);
			up = R * vec4(0.0, 1.0, 0.0, 0.0);

			cam_pos = cam_pos + vec3(fwd) * -move.v[2];
			cam_pos = cam_pos + vec3(up) * move.v[1];
			cam_pos = cam_pos + vec3(rgt) * move.v[0];
			mat4 T = translate(identity_mat4(), vec3(cam_pos));

			view_mat = inverse(R) * inverse(T);

			water_shader.set_uniform_matrix4fv("V", view_mat.m);
			water_shader.set_uniform_matrix4fv("P", proj_mat.m);

			diffuse_shader.set_uniform_matrix4fv("V", view_mat.m);
			diffuse_shader.set_uniform_matrix4fv("P", proj_mat.m);

			texcube_shader.set_uniform_matrix4fv("V", view_mat.m);
			texcube_shader.set_uniform_matrix4fv("P", proj_mat.m);

			refract_shader.set_uniform_matrix4fv("V", view_mat.m);
			refract_shader.set_uniform_matrix4fv("P", proj_mat.m);

			skybox_shader.set_uniform_matrix4fv("V", inverse(R).m);
			skybox_shader.set_uniform_matrix4fv("P", proj_mat.m);

			particle_shader.set_uniform_matrix4fv("V", view_mat.m);
			particle_shader.set_uniform_matrix4fv("P", proj_mat.m);
		}

		water_shader.set_uniform_matrix4fv("P", proj_mat.m);
		diffuse_shader.set_uniform_matrix4fv("P", proj_mat.m);
		texcube_shader.set_uniform_matrix4fv("P", proj_mat.m);
		skybox_shader.set_uniform_matrix4fv("P", proj_mat.m);
		refract_shader.set_uniform_matrix4fv("P", proj_mat.m);
		particle_shader.set_uniform_matrix4fv("P", proj_mat.m);

		if (GLFW_PRESS == glfwGetKey(g_window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(g_window, 1);
		}
		// put the stuff we've been drawing onto the display
		glfwSwapBuffers(g_window);
	}

	// close GL context and any other GLFW resources
	glfwTerminate();
	return 0;
}
