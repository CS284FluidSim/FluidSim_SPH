#include <GL/glew.h>
#include <GL/freeglut.h>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

#include "fluidsim_timer.h"
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
float light_position[] = { 0.0f, 20.0f, 0.0f, 1.0f };
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
GLuint v;
GLuint f;
GLuint p;

//json object name
const std::string SPHERE = "sphere";


void set_shaders()
{
	char *vs = NULL;
	char *fs = NULL;

	vs = (char *)malloc(sizeof(char) * 10000);
	fs = (char *)malloc(sizeof(char) * 10000);
	memset(vs, 0, sizeof(char) * 10000);
	memset(fs, 0, sizeof(char) * 10000);

	FILE *fp;
	char c;
	int count;

	fp = fopen("../shader/shader.vs", "r");
	count = 0;
	while ((c = fgetc(fp)) != EOF)
	{
		vs[count] = c;
		count++;
	}
	fclose(fp);

	fp = fopen("../shader/shader.fs", "r");
	count = 0;
	while ((c = fgetc(fp)) != EOF)
	{
		fs[count] = c;
		count++;
	}
	fclose(fp);

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	const char *vv;
	const char *ff;
	vv = vs;
	ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	int success;

	glCompileShader(v);
	glGetShaderiv(v, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		char info_log[5000];
		glGetShaderInfoLog(v, 5000, NULL, info_log);
		printf("Error in vertex shader compilation!\n");
		printf("Info Log: %s\n", info_log);
	}

	glCompileShader(f);
	glGetShaderiv(f, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		char info_log[5000];
		glGetShaderInfoLog(f, 5000, NULL, info_log);
		printf("Error in fragment shader compilation!\n");
		printf("Info Log: %s\n", info_log);
	}

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);
	glLinkProgram(p);
	glUseProgram(p);

	free(vs);
	free(fs);
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

		//simsystem->add_cube_fluid(cube_min, cube_max);

		//simsystem->add_cube_fluid(make_float3(0.5f, 0.5f, 0.5f), make_float3(0.6f, 0.6f, 0.6f));

		//simsystem->add_cube_fluid(make_float3(0.7f, 0.0f, 0.0f), make_float3(1.0f, 0.9f, 1.0f));

		simsystem->add_cube_fluid(make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 0.2f, 1.0f));

		//simsystem->add_fluid(make_float3(0.2f, 0.5f, 0.1f), make_float3(0.7f, 0.9f, 0.9f));  // a cube drop from the air

		//simsystem->add_fluid(make_float3(0.5f, 0.5f, 0.5f), 0.4f);  // a sphere drop from the air

		//simsystem->add_fluid(make_float3(4.5f, 4.5f, 4.5f));  // a bunny drop
		
		simsystem->add_fluid(make_float3(1.5f, 1.5f, 1.5f));  // a bunny drop
	}
	else
	{
		simsystem = new FluidSim::gpu::SimulateSystem(world_size, sim_ratio, real_world_origin);
		simsystem->add_cube_fluid(make_float3(0.7f, 0.0f, 0.0f), make_float3(1.0f, 0.4f, 1.0f));
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

	gluPerspective(45.0, (float)window_width / window_height, 10.0f, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0f, 0.0f, -3.0f);
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
	}
	else
	{
		glUseProgram(p);
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

	// visualize object point cloud
	/*glPointSize(2.0f);
	glColor3f(0.2f, 0.2f, 1.0f);

	float3 pos;

	for (int i = 0; i < object_coord.size(); i++)
	{
		glBegin(GL_POINTS);
		pos.x = 4.5f * object_coord[i][0] * sim_ratio.x + real_world_origin.x + 7.5f;
		pos.y = 4.5f * object_coord[i][1] * sim_ratio.y + real_world_origin.y - 1.5f;
		pos.z = 4.5f * object_coord[i][2] * sim_ratio.z + real_world_origin.z + 4.0f;
		glVertex3f(pos.x, pos.y, pos.z);
		glEnd();
	}*/
	
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

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	(void)glutCreateWindow("GLUT Program");

	init();
	if (argc == 2)
		init_sph_system(argv[1]);
	else
		init_sph_system("");

	set_shaders();
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

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
