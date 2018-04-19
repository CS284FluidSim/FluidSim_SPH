#include "fluidsim_timer.h"
#include "fluidsim_system.h"
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <fstream>
#include <iostream>

#pragma comment(lib, "glew32.lib") 

using namespace FluidSim;

namespace FluidSim
{
#ifndef BUILD_CUDA
	struct float3
	{
		float x;
		float y;
		float z;
	};
#endif
}


//Somulation system global variable
SimulateSystem *simsystem;
Timer *timer;

//OpenGL global variable
char *window_title;
float window_width = 800;
float window_height = 800;
float xRot = 15.0f;
float yRot = 0.0f;
float xTrans = 0.0;
float yTrans = 0;
float zTrans = -35.0;
int render_mode = 0;
int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;
bool pause = false;
bool wireframe = false;
bool step = false;

//Simulation Parameters
float world_size = 0.16f;
float vox_size = 0.01f;
int row_vox = world_size / vox_size;
int col_vox = world_size / vox_size;
int len_vox = world_size / vox_size;
float3* model_vox;
float *model_scalar;
float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;


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

void init_sph_system()
{
	simsystem = new FluidSim::SimulateSystem(world_size, world_size, world_size);
	simsystem->add_cube_fluid({ 0.f, 0.f, 0.f }, { 1.0f, 0.9f, 0.3f });

	sim_ratio.x = real_world_side.x / world_size;
	sim_ratio.y = real_world_side.y / world_size;
	sim_ratio.z = real_world_side.z / world_size;

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

	real_world_origin = { -10.f, -10.f, -10.f };
	real_world_side = { 20.f, 20.f, 20.f };
}

void render_simulation()
{
	if (render_mode != 3)
	{
		glPointSize(1.0f);
		glColor3f(0.2f, 0.2f, 1.0f);

		float3 color = { 0.2f, 0.2f, 1.0f };
		float3 pos;

		FluidSim::Particle *particles = simsystem->get_particles();

		for (unsigned int i = 0; i < simsystem->get_num_particles(); i++)
		{
			if (render_mode == 0)
			{
				if (particles[i].surf_norm > simsystem->get_surf_norm())
				{
					glColor3f(1.0f, 0.0f, 0.0f);
				}
				else
				{
					glColor3f(0.2f, 1.0f, 0.2f);
				}
			}
			else if (render_mode == 1)
			{
				glColor3f(0.2f, 0.2f, 1.0f);
			}
			else
			{
				float3 vel;
				vel.x = particles[i].vel(0);
				vel.y = particles[i].vel(1);
				vel.z = particles[i].vel(2);
				glColor3f(vel.x*10.f, vel.y*10.f, vel.z*10.f);
			}
			glBegin(GL_POINTS);
			pos.x = particles[i].pos(0)*sim_ratio.x + real_world_origin.x;
			pos.y = particles[i].pos(1)*sim_ratio.y + real_world_origin.y;
			pos.z = particles[i].pos(2)*sim_ratio.z + real_world_origin.z;
			glVertex3f(pos.x, pos.y, pos.z);
			glEnd();
		}
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
		xTrans -= 0.3f;
	}

	if (key == 'd')
	{
		xTrans += 0.3f;
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

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);

	init();
	init_sph_system();

	(void)glutCreateWindow("GLUT Program");
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