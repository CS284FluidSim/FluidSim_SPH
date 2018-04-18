#include <GL\glew.h>
#include <GL\freeglut.h>

#include <fstream>
#include <iostream>

#include "fluidsim_timer.h"
#include "gpu/fluidsim_system.cuh"
//#include "gpu/fluidsim_marchingcube.cuh"
#include "fluidsim_marchingcube.h"

#pragma comment(lib, "glew32.lib") 
#define GPU_MC

//Somulation system global variable
FluidSim::gpu::SimulateSystem *simsystem;
FluidSim::MarchingCube *marchingcube;
FluidSim::Timer *timer;

//OpenGL global variable
float light_ambient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
float light_diffuse[] = { 0.0f, 0.5f, 1.0f, 1.0f };
float light_position[] = { 0.0f, 20.0f, 0.0f, 1.0f };
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
float3 world_size = { 1.28f , 1.28f, 1.28f };
float3 real_world_side = { world_size.x * 10, world_size.y * 10, world_size.z * 10 };
float3 real_world_origin = { -real_world_side.x / 2.f, -real_world_side.y / 2.f, -real_world_side.z / 2.f };
float3 sim_ratio;

//Marching Cubes Parameters
float vox_size = 0.02f;
int row_vox = world_size.x / vox_size;
int col_vox = world_size.y / vox_size;
int len_vox = world_size.z / vox_size;
int tot_vox = row_vox*col_vox*len_vox;
float3* model_vox_init;
float *model_scalar_init;
float3* model_vox;
float *model_scalar;

//Shaders
GLuint v;
GLuint f;
GLuint p;

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

void init_sph_system()
{
	sim_ratio = real_world_side / world_size;

	simsystem = new FluidSim::gpu::SimulateSystem(world_size, sim_ratio, real_world_origin);
	//simsystem->add_cube_fluid(make_float3(0.5f, 0.5f, 0.5f), make_float3(0.6f, 0.6f, 0.6f));

	simsystem->add_cube_fluid(make_float3(0.7f, 0.0f, 0.0f), make_float3(1.0f, 0.9f, 1.0f));

	timer = new FluidSim::Timer();
	window_title = (char *)malloc(sizeof(char) * 50);
}

void init_marching_cube()
{
	int tot_vox = row_vox*col_vox*len_vox;

	model_vox = (float3 *)malloc(sizeof(float3)*row_vox*col_vox*len_vox);
	model_scalar = (float *)malloc(sizeof(float)*row_vox*col_vox*len_vox);

	model_vox_init = (float3 *)malloc(sizeof(float3)*row_vox*col_vox*len_vox);
	model_scalar_init = (float *)malloc(sizeof(float)*row_vox*col_vox*len_vox);

	for (int count_x = 0; count_x < row_vox; count_x++)
	{
		for (int count_y = 0; count_y < col_vox; count_y++)
		{
			for (int count_z = 0; count_z < len_vox; count_z++)
			{
				int index = count_z*row_vox*col_vox + count_y*row_vox + count_x;

				model_vox_init[index].x = count_x*vox_size;
				model_vox_init[index].y = count_y*vox_size;
				model_vox_init[index].z = count_z*vox_size;

				model_scalar_init[index] = 0;
			}
		}
	}

	marchingcube = new FluidSim::MarchingCube(row_vox, col_vox, len_vox, model_scalar, model_vox, sim_ratio, real_world_origin, vox_size, simsystem->get_sys_pararm()->rest_dens);
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
		glPointSize(1.0f);
		glColor3f(0.2f, 0.2f, 1.0f);

		float3 color = make_float3(0.2f, 0.2f, 1.0f);
		float3 pos;

		FluidSim::gpu::Particle *particles = simsystem->get_particles();

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
				float color = 1.0f - particles[i].dens / 5000;
				glColor3f(color*0.0, color*0.3, color*0.6);
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
#ifndef GPU_MC
			FluidSim::gpu::Particle *particles = simsystem->get_particles();

			memcpy(model_vox, model_vox_init, sizeof(float3)*tot_vox);
			memcpy(model_scalar, model_scalar_init, sizeof(float)*tot_vox);

			float radius = 0.02f;

			for (int i = 0; i < simsystem->get_num_particles(); ++i)
			{
				int cell_pos_x = (particles[i].pos.x) / vox_size;
				int cell_pos_y = (particles[i].pos.y) / vox_size;
				int cell_pos_z = (particles[i].pos.z) / vox_size;
				for (float x = -radius; x < radius; x+=vox_size)
				{
					for (float y = -radius; y < radius; y+=vox_size)
					{
						for (float z = -radius; z < radius; z+=vox_size)
						{
								int pos_x = cell_pos_x + x / vox_size;
								int pos_y = cell_pos_y + y / vox_size;
								int pos_z = cell_pos_z + z / vox_size;
								if (pos_x < 0 || pos_x >= row_vox ||
									pos_y < 0 || pos_y >= col_vox ||
									pos_z < 0 || pos_z >= len_vox)
									continue;
								else
								{
									int index = pos_z*row_vox*col_vox + pos_y*row_vox + pos_x;
									model_scalar[index] = 1000;
								}
						}
					}
				}
			}
#endif
			glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
			glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
			glLightfv(GL_LIGHT1, GL_POSITION, light_position);
			glEnable(GL_LIGHT1);
			glEnable(GL_LIGHTING);
#ifndef GPU_MC
			marchingcube->run();
#else
			simsystem->render();
#endif
			glDisable(GL_LIGHTING);
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

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	(void)glutCreateWindow("GLUT Program");

	init();
	init_sph_system();
	init_marching_cube();

	//set_shaders();
	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	//glEnable(GL_POINT_SPRITE_ARB);
	//glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	//glDepthMask(GL_TRUE);
	//glEnable(GL_DEPTH_TEST);

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