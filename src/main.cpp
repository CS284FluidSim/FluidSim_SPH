#include "fluidsim_timer.h"
#ifdef BUILD_CUDA
#include "fluidsim_system_gpu.cuh"
#else
#include "fluidsim_system.h"
#include <eigen3\Eigen\Dense>
#endif
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <fstream>
#include <iostream>

#pragma comment(lib, "glew32.lib") 

FluidSim::SimulateSystem *simsystem;

FluidSim::Timer *timer;
char *window_title;

GLuint v;
GLuint f;
GLuint p;

float window_width = 500;
float window_height = 500;

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

#ifdef BUILD_CUDA
float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;
#else
Vector3f real_world_origin;
Vector3f real_world_side;
Vector3f sim_ratio;
#endif
float world_width;
float world_height;
float world_length;

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

	fp = fopen("../Shader/shader.vs", "r");
	count = 0;
	while ((c = fgetc(fp)) != EOF)
	{
		vs[count] = c;
		count++;
	}
	fclose(fp);

	fp = fopen("../Shader/shader.fs", "r");
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
	simsystem = new FluidSim::SimulateSystem();
#ifdef BUILD_CUDA
	real_world_origin.x = -10.0f;
	real_world_origin.y = -10.0f;
	real_world_origin.z = -10.0f;

	real_world_side.x = 20.0f;
	real_world_side.y = 20.0f;
	real_world_side.z = 20.0f;

	simsystem->add_cube_fluid(make_float3(0.f, 0.f, 0.f), make_float3(0.6f, 0.9f, 0.6f));
#else
	real_world_origin(0) = -10.0f;
	real_world_origin(1) = -10.0f;
	real_world_origin(2) = -10.0f;

	real_world_side(0) = 20.0f;
	real_world_side(1) = 20.0f;
	real_world_side(2) = 20.0f;

	simsystem->add_cube_fluid(Vector3f(0.f,0.f,0.f), Vector3f(0.6f, 0.9f, 0.6f));
#endif
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

void init_ratio()
{
#ifdef BUILD_CUDA
	float3 world_size = simsystem->get_sys_pararm()->world_size;
	sim_ratio.x = real_world_side.x / world_size.x;
	sim_ratio.y = real_world_side.y / world_size.y;
	sim_ratio.z = real_world_side.z / world_size.z;
#else
	Vector3f world_size = simsystem->get_world_size();
	sim_ratio(0) = real_world_side(0) / world_size(0);
	sim_ratio(1) = real_world_side(1) / world_size(1);
	sim_ratio(2) = real_world_side(2) / world_size(2);
#endif
}

void render_particles()
{
	glPointSize(1.0f);
	glColor3f(0.2f, 0.2f, 1.0f);

#ifdef BUILD_CUDA
	float3 color = make_float3(0.2f,0.2f,1.0f);
	float3 pos;
#else
	Vector3f color;
	Vector3f pos;
#endif

	FluidSim::Particle *particles = simsystem->get_particles();

	for (unsigned int i = 0; i<simsystem->get_num_particles(); i++)
	{
#ifdef BUILD_CUDA
		if (render_mode == 0)
		{
			if (particles[i].surf_norm <= simsystem->get_sys_pararm()->surf_norm)
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
#else
		//color = particles[i].vel*10.f;
		//glColor3f(color(0), color(1), color(2));
		if (particles[i].surf_norm > simsystem->get_surf_norm())
		{
			glColor3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			glColor3f(0.2f, 1.0f, 0.2f);
		}
		glBegin(GL_POINTS);
		pos = particles[i].pos.cwiseProduct(sim_ratio) + real_world_origin;
		glVertex3f(pos(0), pos(1), pos(2));
#endif
		glEnd();
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

	simsystem->animation();

	glUseProgram(p);
	render_particles();

	glUseProgram(0);
#if BUILD_CUDA
	draw_box(real_world_origin.x, real_world_origin.y, real_world_origin.z, real_world_side.x, real_world_side.y, real_world_side.z);
#else
	draw_box(real_world_origin(0), real_world_origin(1), real_world_origin(2), real_world_side(0), real_world_side(1), real_world_side(2));
#endif
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
		render_mode = (render_mode + 1) % 3;
	}

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
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("SPH Fluid 3D");

	init_sph_system();
	init();
	init_ratio();
	set_shaders();
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glutDisplayFunc(display_func);
	glutReshapeFunc(reshape_func);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutMotionFunc(motion_func);
	glutIdleFunc(idle_func);

	glutMainLoop();

	return 0;
}
