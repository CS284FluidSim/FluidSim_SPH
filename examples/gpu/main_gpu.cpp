#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <string>

#include "fluidsim_timer.h"
#include "fluidsim_gl_utils.h"
#include "gpu/fluidsim_system.cuh"
#include "gpu/fluidsim_marchingcube.cuh"

#include "fluidsim_marchingcube.h"

#include "json.hpp"
#include <string>
#include <vector>



#pragma comment(lib, "glew32.lib") 
#define GPU_MC

using json = nlohmann::json;

//Somulation system global variable
FluidSim::gpu::SimulateSystem *simsystem;
FluidSim::Timer *timer;

//OpenGL global variable
//light
float light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_ambient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_diffuse[] = { 0.0f, 0.5f, 1.0f, 1.0f };
float light_position[] = { 0.0f, 50.0f, 0.0f, 1.0f };
//material
float mat_specular[] = { 0.6f, 0.6f, 0.6f, 1.0f };

char *window_title;
float window_width = 500;
float window_height = 500;
float xRot = 15.0f;
float yRot = 0.0f;
float xTrans = 0.0;
float yTrans = 0;
float zTrans = -35.0;
int render_mode = 0;
int mc_render_mode = 0;
int collision_render_mode = 0;
int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;
bool pause = false;
bool wireframe = false;
bool step = false;

//Simulation Parameters
float3 world_size = { 1.28f, 1.28f, 1.28f };
float3 real_world_side = { world_size.x * 10, world_size.y * 10, world_size.z * 10 };
float3 real_world_origin = { -real_world_side.x / 2.f, -real_world_side.y / 2.f, -real_world_side.z / 2.f };
float3 sim_ratio;

//Shaders
GLuint phong_shader;
GLuint particle_shader;
GLuint refractive_shader;
GLuint skybox_shader;

//VAO
GLuint skyboxVAO;

//Textures
unsigned int cube_tex_id;

using namespace std;
using namespace FluidSim;

void draw_skybox()
{
	glDepthMask(GL_FALSE);
	glUseProgram(skybox_shader);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cube_tex_id);
	glBindVertexArray(skyboxVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glDepthMask(GL_TRUE);
}

void draw_box(float ox, float oy, float oz, float width, float height, float length)
{
	glLineWidth(1.0f);
	glColor3f(1.0f, 1.0f, 1.0f);

	glBegin(GL_LINES);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox + width, oy, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy, oz);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy + height, oz);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox, oy + height, oz);

	glVertex3f(ox + width, oy, oz);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox + width, oy + height, oz);
	glVertex3f(ox + width, oy + height, oz + length);

	glVertex3f(ox + width, oy + height, oz + length);
	glVertex3f(ox + width, oy, oz + length);

	glVertex3f(ox, oy + height, oz + length);
	glVertex3f(ox + width, oy + height, oz + length);

	glEnd();
}

void init_sph_system(std::string config_path)
{

	if(config_path!="")
	{
		cv::FileStorage fs;
		fs.open(config_path, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			std::cout << "Error: cannot open configuration file" << std::endl;
			exit(0);
		}

		fs["world_size.x"] >> world_size.x;
		fs["world_size.y"] >> world_size.y;
		fs["world_size.z"] >> world_size.z;
		real_world_side = { world_size.x * 10, world_size.y * 10, world_size.z * 10 };
		real_world_origin = { -real_world_side.x / 2.f, -real_world_side.y / 2.f, -real_world_side.z / 2.f };
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

		simsystem->add_cube_fluid(make_float3(0.9f, 0.0f, 0.0f), make_float3(1.0f, 0.9f, 1.0f), gap);

		//simsystem->add_cube_fluid(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 0.2f, 1.0f), gap);

		//simsystem->add_fluid(make_float3(0.2f, 0.5f, 0.1f), make_float3(0.7f, 0.9f, 0.9f));  // a cube drop from the air

		//simsystem->add_fluid(make_float3(0.5f, 0.5f, 0.5f), 0.4f);  // a sphere drop from the air

		//simsystem->add_fluid(make_float3(4.5f, 4.5f, 4.5f));  // a bunny drop
		
		//simsystem->add_fluid(make_float3(1.5f, 1.5f, 1.5f));  // a bunny drop

		simsystem->add_static_object({ 0.3f,0.0f,0.3f }, { 0.7f,0.7f,0.7f });
	}
	else
	{
		simsystem = new FluidSim::gpu::SimulateSystem(world_size, sim_ratio, real_world_origin);
		simsystem->add_cube_fluid(make_float3(0.7f, 0.0f, 0.0f), make_float3(1.0f, 0.4f, 1.0f), 0.5f);
	}

	timer = new FluidSim::Timer();
	window_title = (char *)malloc(sizeof(char) * 50);
}

void init()
{
	glewInit();

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)window_width / window_height, 0.1f, 1000.f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void init_cube_map(string cube_path)
{
	vector<string> texture_path;
	texture_path.push_back(cube_path + "/posx.jpg");
	texture_path.push_back(cube_path + "/negx.jpg");
	texture_path.push_back(cube_path + "/posy.jpg");
	texture_path.push_back(cube_path + "/negy.jpg");
	texture_path.push_back(cube_path + "/posz.jpg");
	texture_path.push_back(cube_path + "/negz.jpg");

	skyboxVAO = create_cube_vao(10.f);
	cube_tex_id = create_cube_map_tex(texture_path[0].c_str(), texture_path[1].c_str(), 
		texture_path[2].c_str(), texture_path[3].c_str(), 
		texture_path[4].c_str(), texture_path[5].c_str());
}

void render_simulation()
{
	if (wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (render_mode != 3)
	{
		glUseProgram(0);
		glPointSize(1.0f);
		glColor3f(0.2f, 0.2f, 1.0f);

		float3 color = make_float3(0.2f, 0.2f, 1.0f);
		float3 pos;

		FluidSim::gpu::Particle *particles = simsystem->get_particles();

		for (unsigned int i = 0; i < simsystem->get_num_particles(); i++)
		{
			if (render_mode == 0)  // particle color represent surface normal
			{
				if (particles[i].surf_norm > simsystem->get_sys_pararm()->surf_norm)
				{
					glColor3f(1.0f, 0.2f, 1.0f);
				}
				else
				{
					glColor3f(0.2f, 2.0f, 1.0f);
				}
			}
			else if (render_mode == 1)  // particle color represent particle density
			{
				float color = 1.0f - particles[i].dens / 5000;
				glColor3f(color*0.0, color*0.3, color*0.6);
			}
			else  // particle color represent particle velocity
			{
				float3 vel = particles[i].force/100.f;
				glColor3f((vel.x+1)/2.f, (vel.y + 1) / 2.f, (vel.z + 1) / 2.f);
			}
			glBegin(GL_POINTS);
			pos.x = particles[i].pos.x*sim_ratio.x + real_world_origin.x;
			pos.y = particles[i].pos.y*sim_ratio.y + real_world_origin.y;
			pos.z = particles[i].pos.z*sim_ratio.z + real_world_origin.z;
			glVertex3f(pos.x, pos.y, pos.z);
			glEnd();
		}
		glUseProgram(0);
	}
	else
	{
		glUseProgram(phong_shader);
		if (simsystem->is_running())
		{
			if (mc_render_mode == 0)
			{
				glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
				glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
				glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
				glLightfv(GL_LIGHT0, GL_POSITION, light_position);
				glEnable(GL_LIGHT0);
				glEnable(GL_LIGHTING);
			}
			FluidSim::gpu::MarchingCube::RenderMode rm;
			switch (mc_render_mode)
			{
				case 0:rm = FluidSim::gpu::MarchingCube::RenderMode::TRI; break;
				case 1:rm = FluidSim::gpu::MarchingCube::RenderMode::SCALAR; break;
				case 2:rm = FluidSim::gpu::MarchingCube::RenderMode::NORMAL; break;
				case 3:rm = FluidSim::gpu::MarchingCube::RenderMode::POS; break;
			}
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			simsystem->render(rm);
			if (mc_render_mode == 0)
			{
				glDisable(GL_LIGHTING);
			}
		}
		glUseProgram(0);
	}
}

void display_func()
{
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	if (buttonState == 1)
	{
		xRot += (xRotLength - xRot)*0.1f;
		yRot += (yRotLength - yRot)*0.1f;
	}

	glTranslatef(xTrans, yTrans, zTrans);
	glRotatef(xRot, 1.0f, 0.0f, 0.0f);
	glRotatef(yRot, 0.0f, 1.0f, 0.0f);

	if (!pause || step)
	{
		simsystem->animation();
		step = !step;
	}

	render_simulation();

	draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);

	//draw_skybox();

	glPopMatrix();
	glutSwapBuffers();

	timer->update();
	memset(window_title, 0, 50);
	sprintf(window_title, "SPH System 3D. FPS: %f", timer->get_fps());
	glutSetWindowTitle(window_title);
}

void idle_func()
{
	glutPostRedisplay();
}

void reshape_func(GLint width, GLint height)
{
	window_width = width;
	window_height = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (float)width / height, 0.001, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y)
{
	if (key == ' ')
	{
		simsystem->start();
	}

	if (key == 'w')
	{
		zTrans += 0.3f;
	}

	if (key == 's')
	{
		zTrans -= 0.3f;
	}

	if (key == 'a')
	{
		xTrans += 0.3f;
	}

	if (key == 'd')
	{
		xTrans -= 0.3f;
	}

	if (key == 'q')
	{
		yTrans -= 0.3f;
	}

	if (key == 'e')
	{
		yTrans += 0.3f;
	}

	if (key == 'c')
	{
		render_mode = (render_mode + 1) % 4;
	}

	if (key == 'm')
	{
		mc_render_mode = (mc_render_mode + 1) % 4;
	}

	if (key == 'k')
	{
		if (simsystem->is_running())
		{
			simsystem->add_cube_fluid({ 0.4f,0.4f,0.4f }, { 0.5f,0.5f,0.5f }, 0.5);
		}
	}

	if (key == 'v')
		wireframe = !wireframe;

	if (key == 'p')
		pause = !pause;

	if (key == 'n')
		step = !step;

	glutPostRedisplay();
}

void mouse_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void motion_func(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 1)
	{
		xRotLength += dy / 5.0f;
		yRotLength += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

int main(int argc, char **argv)
{
	if (argc != 3)
		exit(0);

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	(void)glutCreateWindow("GLUT Program");

	init();
	init_cube_map(argv[2]);
	init_sph_system(argv[1]);

	phong_shader = create_shader_program("../shader/phong.vs", "../shader/phong.fs");
	particle_shader = create_shader_program("../shader/particle.vs", "../shader/particle.fs");
	skybox_shader = create_shader_program("../shader/skybox.vs", "../shader/skybox.fs");

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	//glDepthMask(GL_TRUE);
	//glEnable(GL_DEPTH_TEST);	// enable depth-testing
	//glDepthFunc(GL_LESS);		 // depth-testing interprets a smaller value as "closer"
	//glEnable(GL_CULL_FACE);	// cull face
	//glCullFace(GL_BACK);		 // cull back face
	//glFrontFace(GL_CCW); // set counter-clock-wise vertex order to mean the front

	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutIdleFunc(idle_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutKeyboardFunc(keyboard_func);
	glutMainLoop();

	delete simsystem;

	return EXIT_SUCCESS;
}
