/******************************************************************************\
| OpenGL 4 Example Code.                                                       |
| Accompanies written series "Anton's OpenGL 4 Tutorials"                      |
| Email: anton at antongerdelan dot net                                        |
| First version 27 Jan 2014                                                    |
| Copyright Dr Anton Gerdelan, Trinity College Dublin, Ireland.                |
| See individual libraries separate legal notices                              |
|******************************************************************************|
| Cube Maps                                                                    |
| You can swap the "reflect_vs.glsl" and "reflect_fs.glsl" for the refraction  |
| versions. Comment one set out and uncomment the other                        |
\******************************************************************************/
#include "gl_utils.h"		 // common opengl functions and small utilities like logs
#include "maths_funcs.h" // my maths functions
#include "obj_parser.h"	// my little Wavefront .obj mesh loader
#include "stb_image.h"	 // Sean Barrett's image loader - nothings.org
#include <GL/glew.h>		 // include GLEW and new version of GL on Windows
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>	// GLFW helper library
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <gpu/fluidsim_system.cuh>
#include <gpu/fluidsim_marchingcube.cuh>
#define MESH_FILE "../scene/suzanne.obj"

/* choose pure reflection or pure refraction here. */
#define MONKEY_VERT_FILE "../shader/water_vs.glsl"
#define MONKEY_FRAG_FILE "../shader/water_fs.glsl"
//#define MONKEY_VERT_FILE "refract_vs.glsl"
//#define MONKEY_FRAG_FILE "refract_fs.glsl"

#define CUBE_VERT_FILE "../shader/cube_vs.glsl"
#define CUBE_FRAG_FILE "../shader/cube_fs.glsl"

#define TEST_VERT_FILE "../shader/test_vs.glsl"
#define TEST_FRAG_FILE "../shader/test_fs.glsl"

#define FRONT "../texture/cube/bridge/negz.jpg"
#define BACK "../texture/cube/bridge/posz.jpg"
#define TOP "../texture/cube/bridge/posy.jpg"
#define BOTTOM "../texture/cube/bridge/negy.jpg"
#define LEFT "../texture/cube/bridge/negx.jpg"
#define RIGHT "../texture/cube/bridge/posx.jpg"

#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "glew32.lib") 

//FluidSim
FluidSim::gpu::SimulateSystem *simsystem;

// keep track of window size for things like the viewport and the mouse cursor
int g_gl_width = 960;
int g_gl_height = 640;
GLFWwindow *g_window = NULL;

/* big cube. returns Vertex Array Object */
GLuint make_cube(float size) {
	float points[] = {
		-size, size,	-size, -size, -size, -size, size,	-size, -size,
		size,	-size, -size, size,	size,	-size, -size, size,	-size,

		-size, -size, size,	-size, -size, -size, -size, size,	-size,
		-size, size,	-size, -size, size,	size,	-size, -size, size,

		size,	-size, -size, size,	-size, size,	size,	size,	size,
		size,	size,	size,	size,	size,	-size, size,	-size, -size,

		-size, -size, size,	-size, size,	size,	size,	size,	size,
		size,	size,	size,	size,	-size, size,	-size, -size, size,

		-size, size,	-size, size,	size,	-size, size,	size,	size,
		size,	size,	size,	-size, size,	size,	-size, size,	-size,

		-size, -size, -size, -size, -size, size,	size,	-size, -size,
		size,	-size, -size, -size, -size, size,	size,	-size, size
	};
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * 36 * sizeof(GLfloat), &points,
		GL_STATIC_DRAW);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	return vao;
}

GLuint make_triangle()
{
	/* OTHER STUFF GOES HERE NEXT */
	GLfloat points[] = { 0.0f, 1.0f, 0.2f, 0.5f, -0.5f, 0.2f, -0.5f, -0.5f, 0.2f };

	//GLfloat colours[] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };

	GLuint points_vbo;
	glGenBuffers(1, &points_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(GLfloat), points, GL_STATIC_DRAW);

	/* create the VAO.
	we bind each VBO in turn, and call glVertexAttribPointer() to indicate where
	the memory should be fetched for vertex shader input variables 0, and 1,
	respectively. we also have to explicitly enable both 'attribute' variables.
	'attribute' is the older name for vertex shader 'in' variables. */
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	return vao;
}

/* use stb_image to load an image file into memory, and then into one side of
a cube-map texture. */
bool load_cube_map_side(GLuint texture, GLenum side_target,
	const char *file_name) {
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

	int x, y, n;
	int force_channels = 4;
	unsigned char *image_data = stbi_load(file_name, &x, &y, &n, force_channels);
	if (!image_data) {
		fprintf(stderr, "ERROR: could not load %s\n", file_name);
		return false;
	}
	// non-power-of-2 dimensions check
	if ((x & (x - 1)) != 0 || (y & (y - 1)) != 0) {
		fprintf(stderr, "WARNING: image %s is not power-of-2 dimensions\n",
			file_name);
	}

	// copy image data into 'target' side of cube map
	glTexImage2D(side_target, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
		image_data);
	free(image_data);
	return true;
}

/* load all 6 sides of the cube-map from images, then apply formatting to the
final texture */
void create_cube_map(const char *front, const char *back, const char *top,
	const char *bottom, const char *left, const char *right,
	GLuint *tex_cube) {
	// generate a cube-map texture to hold all the sides
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, tex_cube);

	// load each image and copy into a side of the cube-map texture
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, front));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, back));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, top));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, bottom));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, left));
	(load_cube_map_side(*tex_cube, GL_TEXTURE_CUBE_MAP_POSITIVE_X, right));
	// format cube map texture
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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
		//real_world_origin = { -real_world_side.x / 2.f, -real_world_side.y / 2.f, -real_world_side.z / 2.f };
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

		//simsystem->add_fluid(make_float3(4.5f, 4.5f, 4.5f));  // a bunny drop

		//simsystem->add_fluid(make_float3(1.5f, 1.5f, 1.5f));  // a bunny drop
		FluidSim::Cube *cube = new FluidSim::Cube({ 0.5f,0.5f,0.5f},
		{ 1.f, 1.f, 1.f});
		//Sphere *sphere = new Sphere({ 0.5f*world_size.x,0.2f*world_size.y,0.5f*world_size.z }, 0.4f);
		//Model *model = new Model("../scene/bunny.txt", { 0.5f*world_size.x,0.5f*world_size.y,0.5f*world_size.z }, 0.1f);
		//simsystem->add_static_object(cube);
		//simsystem->add_static_object(cube);
	}
}

// camera matrices. it's easier if they are global
mat4 view_mat;
mat4 proj_mat;
vec3 cam_pos(0.0f, 0.0f, 5.0f);

int main() {
	/*--------------------------------START
	* OPENGL--------------------------------*/
	restart_gl_log();
	// start GL context and O/S window using the GLFW helper library
	start_gl();

	init_sph_system("config.yaml");

	/*---------------------------------CUBE
	* MAP-----------------------------------*/
	GLuint cube_vao = make_cube(10.f);
	GLuint cube_map_texture;
	create_cube_map(FRONT, BACK, TOP, BOTTOM, LEFT, RIGHT, &cube_map_texture);

	GLuint tri_vao = make_triangle();

	GLuint cube_1_vao = make_cube(1.f);
	/*------------------------------CREATE
	* GEOMETRY-------------------------------*/
	GLfloat *vp = NULL; // array of vertex points
	GLfloat *vn = NULL; // array of vertex normals
	GLfloat *vt = NULL; // array of texture coordinates
	int g_point_count = 0;
	(load_obj_file(MESH_FILE, vp, vt, vn, g_point_count));

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint points_vbo, normals_vbo;
	if (NULL != vp) {
		glGenBuffers(1, &points_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
		glBufferData(GL_ARRAY_BUFFER, 3 * g_point_count * sizeof(GLfloat), vp,
			GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(0);
	}
	if (NULL != vn) {
		glGenBuffers(1, &normals_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
		glBufferData(GL_ARRAY_BUFFER, 3 * g_point_count * sizeof(GLfloat), vn,
			GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(1);
	}

	/*-------------------------------CREATE
	* SHADERS-------------------------------*/
	// shaders for "Suzanne" mesh
	GLuint monkey_sp =
		create_programme_from_files(MONKEY_VERT_FILE, MONKEY_FRAG_FILE);
	int monkey_M_location = glGetUniformLocation(monkey_sp, "M");
	int monkey_V_location = glGetUniformLocation(monkey_sp, "V");
	int monkey_P_location = glGetUniformLocation(monkey_sp, "P");

	// cube-map shaders
	GLuint cube_sp = create_programme_from_files(CUBE_VERT_FILE, CUBE_FRAG_FILE);
	// note that this view matrix should NOT contain camera translation.
	int cube_V_location = glGetUniformLocation(cube_sp, "V");
	int cube_P_location = glGetUniformLocation(cube_sp, "P");

	GLuint test_sp = create_programme_from_files(TEST_VERT_FILE, TEST_FRAG_FILE);
	int test_V_location = glGetUniformLocation(test_sp, "V");
	int test_P_location = glGetUniformLocation(test_sp, "P");

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
	mat4 R = rotate_y_deg(identity_mat4(), -cam_heading);
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
	glUseProgram(monkey_sp);
	glUniformMatrix4fv(monkey_M_location, 1, GL_FALSE, model_mat.m);
	glUniformMatrix4fv(monkey_V_location, 1, GL_FALSE, view_mat.m);
	glUniformMatrix4fv(monkey_P_location, 1, GL_FALSE, proj_mat.m);
	glUseProgram(cube_sp);
	glUniformMatrix4fv(cube_V_location, 1, GL_FALSE, R.m);
	glUniformMatrix4fv(cube_P_location, 1, GL_FALSE, proj_mat.m);
	glUseProgram(test_sp);
	glUniformMatrix4fv(test_V_location, 1, GL_FALSE, view_mat.m);
	glUniformMatrix4fv(test_P_location, 1, GL_FALSE, proj_mat.m);


	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS);		 // depth-testing interprets a smaller value as "closer"
	//glEnable(GL_CULL_FACE);	// cull face
	//glCullFace(GL_BACK);		 // cull back face
	//glFrontFace(GL_CCW); // set counter-clock-wise vertex order to mean the front
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

		// run simulation
		if (simsystem->is_running())
		{
			simsystem->animation();
			glUseProgram(test_sp);
			simsystem->render_static_object();
			//simsystem->render_particles();
			glUseProgram(monkey_sp);
			simsystem->render_surface(FluidSim::gpu::MarchingCube::RenderMode::TRI);
		}

		glUseProgram(cube_sp);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map_texture);
		glBindVertexArray(cube_vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);

		//glBindVertexArray(cube_1_vao);
		//glDrawArrays(GL_TRIANGLES, 0, 36);

		// render a sky-box using the cube-map texture
		//glDepthMask(GL_FALSE);
		//glUseProgram(cube_sp);
		//glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map_texture);
		//glBindVertexArray(cube_vao);
		//glDrawArrays(GL_TRIANGLES, 0, 36);
		//glDepthMask(GL_TRUE);

		//glUseProgram(monkey_sp);
		//glBindVertexArray(vao);
		//glUniformMatrix4fv(monkey_M_location, 1, GL_FALSE, model_mat.m);
		//glDrawArrays(GL_TRIANGLES, 0, g_point_count);
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
			glUseProgram(monkey_sp);
			glUniformMatrix4fv(monkey_V_location, 1, GL_FALSE, view_mat.m);
			glUniformMatrix4fv(monkey_P_location, 1, GL_FALSE, proj_mat.m);

			// cube-map view matrix has rotation, but not translation
			glUseProgram(cube_sp);
			glUniformMatrix4fv(cube_V_location, 1, GL_FALSE, inverse(R).m);
			glUniformMatrix4fv(cube_P_location, 1, GL_FALSE, proj_mat.m);

			glUseProgram(test_sp);
			glUniformMatrix4fv(test_V_location, 1, GL_FALSE, view_mat.m);
			glUniformMatrix4fv(test_P_location, 1, GL_FALSE, proj_mat.m);
		}
		glUseProgram(monkey_sp);
		glUniformMatrix4fv(monkey_P_location, 1, GL_FALSE, proj_mat.m);

		// cube-map view matrix has rotation, but not translation
		glUseProgram(cube_sp);
		glUniformMatrix4fv(cube_P_location, 1, GL_FALSE, proj_mat.m);

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
