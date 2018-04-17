#include "fluidsim_timer.h"
#include "fluidsim_system_gpu.cuh"
#include "fluidsim_marchingcube.h"
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <fstream>
#include <iostream>

#pragma comment(lib, "glew32.lib") 

//Somulation system global variable
FluidSim::SimulateSystem *simsystem;
FluidSim::MarchingCube *marchingcube;
FluidSim::Timer *timer;

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
float world_size = 0.32f;
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
	simsystem = new FluidSim::SimulateSystem(world_size,world_size,world_size);
	simsystem->add_cube_fluid(make_float3(0.f, 0.f, 0.f), make_float3(1.0f, 0.9f, 0.3f));

	sim_ratio = real_world_side / world_size;

	timer = new FluidSim::Timer();
	window_title = (char *)malloc(sizeof(char) * 50);
}

void init_marching_cube()
{
	int tot_vox = row_vox*col_vox*len_vox;

	model_vox = (float3 *)malloc(sizeof(float3)*row_vox*col_vox*len_vox);
	model_scalar = (float *)malloc(sizeof(float)*row_vox*col_vox*len_vox);

	marchingcube = new FluidSim::MarchingCube(row_vox, col_vox, len_vox, model_scalar, model_vox, sim_ratio, real_world_origin, vox_size, simsystem->get_sys_pararm()->surf_norm);
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

	real_world_origin = make_float3(-10.f, -10.f, -10.f);
	real_world_side = make_float3(20.f, 20.f, 20.f);
}

void render_simulation()
{
	if (render_mode != 3)
	{
		glPointSize(1.0f);
		glColor3f(0.2f, 0.2f, 1.0f);

		float3 color = make_float3(0.2f, 0.2f, 1.0f);
		float3 pos;

		FluidSim::Particle *particles = simsystem->get_particles();

		for (unsigned int i = 0; i < simsystem->get_num_particles(); i++)
		{
			if (render_mode == 0)
			{
				if (particles[i].surf_norm > simsystem->get_sys_pararm()->surf_norm)
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
				float3 vel = particles[i].vel;
				glColor3f(vel.x*10.f, vel.y*10.f, vel.z*10.f);
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

		if (simsystem->is_running())
		{

			GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
			GLfloat mat_shininess[] = { 50.0 };
			GLfloat light_position[] = { 10.0, 10.0, 10.0, 0.0 };
			glShadeModel(GL_SMOOTH);
			glPushAttrib(GL_ALL_ATTRIB_BITS);
			if (wireframe)
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
			glLightfv(GL_LIGHT0, GL_POSITION, light_position);

			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_COLOR_MATERIAL);
			glColorMaterial(GL_FRONT, GL_DIFFUSE);
			glColor3f(0.2, 0.5, 0.8);
			glColorMaterial(GL_FRONT, GL_SPECULAR);
			glColor3f(0.9, 0.9, 0.9);
			glColorMaterial(GL_FRONT, GL_AMBIENT);
			glColor3f(0.2, 0.5, 0.8);

			FluidSim::Particle *particles = simsystem->get_particles();

			for (int count_x = 0; count_x < row_vox; count_x++)
			{
				for (int count_y = 0; count_y < col_vox; count_y++)
				{
					for (int count_z = 0; count_z < len_vox; count_z++)
					{
						int index = count_z*row_vox*col_vox + count_y*row_vox + count_x;

						model_vox[index].x = count_x*vox_size;
						model_vox[index].y = count_y*vox_size;
						model_vox[index].z = count_z*vox_size;

						model_scalar[index] = 0;
					}
				}
			}

			for (int i = 0; i < simsystem->get_num_particles(); ++i)
			{
				//for (float x = -0.01; x <= 0.01; x += vox_size)
				//{
				//	for (float y = -0.01; y <= 0.01; y += vox_size)
				//	{
				//		for (float z = -0.01; z <= 0.01; z += vox_size)
				//		{
							/*if (particles[i].pos.x + x < 0 || particles[i].pos.y + y < 0 || particles[i].pos.z + z < 0
								|| particles[i].pos.x + x > world_size || particles[i].pos.y + y > world_size || particles[i].pos.z + z > world_size)
								continue;*/
							int cell_pos_x = (particles[i].pos.x) / vox_size;
							int cell_pos_y = (particles[i].pos.y) / vox_size;
							int cell_pos_z = (particles[i].pos.z) / vox_size;
							int index = cell_pos_z*row_vox*col_vox + cell_pos_y*row_vox + cell_pos_x;
							model_vox[index].x = cell_pos_x*vox_size;
							model_vox[index].y = cell_pos_y*vox_size;
							model_vox[index].z = cell_pos_z*vox_size;
							if (particles[i].surf_norm>simsystem->get_sys_pararm()->surf_norm)
								model_scalar[index] = particles[i].surf_norm;
				//		}
				//	}
				//}

			}
			marchingcube->run();
			glPopAttrib();
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
	init_marching_cube();

	(void)glutCreateWindow("GLUT Program");
	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutIdleFunc(idle_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutKeyboardFunc(keyboard_func);
	glutMainLoop();

	delete simsystem;
	delete marchingcube;

	return EXIT_SUCCESS;
}
