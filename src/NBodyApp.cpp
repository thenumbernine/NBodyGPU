#include "CLApp/CLApp.h"
#include "TensorMath/Vector.h"
#include "Profiler/Profiler.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "nbody.h"
#include "Quat.h"

struct NBodyApp : public ::CLApp::CLApp {
	typedef ::CLApp::CLApp Super;
	
	GLuint posVBO;
	cl::Memory posMem;

	cl::Buffer objsMem;	//current contents
	cl::Buffer objsMemPrev;	//previous contents
	cl::Program program;
	cl::Kernel updateKernel;
	cl::Kernel copyToGLKernel;

	int count;

	cl::NDRange globalSize;
	cl::NDRange localSize;
	
	Quat viewAngle;
	float dist;
	bool leftShiftDown;
	bool rightShiftDown;
	bool leftButtonDown;
	bool rightButtonDown;

	NBodyApp();
	virtual void init();
	virtual void update();
	virtual void resize(int width, int height);
	virtual void sdlEvent(SDL_Event &event);
};

NBodyApp::NBodyApp()
: Super()
, posVBO(0)
, count(8192)
, dist(1.f)
, leftShiftDown(false)
, rightShiftDown(false)
, leftButtonDown(false)
, rightButtonDown(false)
{}

void NBodyApp::init() {
	Super::init();

	size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	std::cout << "max work group size " << maxWorkGroupSize << std::endl;
	std::cout << "max work item sizes";
	for (size_t size : maxWorkItemSizes) { std::cout << " " << size; }
	std::cout << std::endl;

	size_t localSizeValue = 16;//std::min<size_t>(maxWorkItemSizes[0], count);
	globalSize = cl::NDRange(count);
	localSize = cl::NDRange(localSizeValue);

	glGenBuffers(1, &posVBO);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);	
	glBufferData(GL_ARRAY_BUFFER, sizeof(cl_float4) * count, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);	
	
	posMem = cl::BufferGL(context, CL_MEM_WRITE_ONLY, posVBO);

	objsMem = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Object) * count);
	objsMemPrev = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(Object) * count);

	/*
	radial density profile: rho(r) = 3(v(r))^2 / (4 pi G r^2)	<- by Kepler's 3rd law
	*/
	std::vector<Object> objs(count);
	for (Object &obj : objs) {
		obj.pos.s[0] = crand();
		obj.pos.s[1] = crand();
		obj.pos.s[2] = crand();
		obj.vel.s[0] = crand();
		obj.vel.s[1] = crand();
		obj.vel.s[2] = crand();
		obj.mass = exp(frand() * 3.);
	}
	commands.enqueueWriteBuffer(objsMem, CL_TRUE, 0, sizeof(Object) * count, &objs[0]);
	commands.enqueueWriteBuffer(objsMemPrev, CL_TRUE, 0, sizeof(Object) * count, &objs[0]);

	try {
		std::string source = Common::File::read("nbody.cl");
		std::vector<std::pair<const char *, size_t>> sources = {
			std::pair<const char *, size_t>(source.c_str(), source.length())
		};
		program = cl::Program(context, sources);
		program.build({device}, "-I include");
	} catch (cl::Error &err) {
		std::cout << "failed to build program executable!" << std::endl;
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		throw;
	}

	updateKernel = cl::Kernel(program, "update");
	updateKernel.setArg(2, count);

	copyToGLKernel = cl::Kernel(program, "copyToGL");
	copyToGLKernel.setArg(0, posMem);
	copyToGLKernel.setArg(2, count);
}

void NBodyApp::update() {
PROFILE_BEGIN_FRAME()
	glFinish();

	std::vector<cl::Memory> glObjects = {posMem};
	commands.enqueueAcquireGLObjects(&glObjects);

	//copy CL to GL
	copyToGLKernel.setArg(1, objsMem);
	commands.enqueueNDRangeKernel(copyToGLKernel, cl::NDRange(0), globalSize, localSize);
	
	commands.enqueueReleaseGLObjects(&glObjects);
	commands.finish();

	//update CL state
	updateKernel.setArg(0, objsMemPrev);	//write new state over old state
	updateKernel.setArg(1, objsMem);
	commands.enqueueNDRangeKernel(updateKernel, cl::NDRange(0), globalSize, localSize);
	std::swap<cl::Memory>(objsMem, objsMemPrev);

	//render GL
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0,0,-dist);
	Quat angleAxis = viewAngle.toAngleAxis();
	glRotatef(angleAxis(3) * 180. / M_PI, angleAxis(0), angleAxis(1), angleAxis(2));
	

	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, count);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

PROFILE_END_FRAME()
}

void NBodyApp::resize(int width, int height) {
	Super::resize(width, height);
	const float zNear = .01;
	const float zFar = 10;
	float aspectRatio = (float)width / (float)height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-aspectRatio * zNear, aspectRatio * zNear, -zNear, zNear, zNear, zFar);
}
	
void NBodyApp::sdlEvent(SDL_Event &event) {
	bool shiftDown = leftShiftDown || rightShiftDown;

	switch (event.type) {
	case SDL_MOUSEMOTION:
		{
			int dx = event.motion.xrel;
			int dy = event.motion.yrel;
			if (leftButtonDown) {
				if (shiftDown) {
					if (dy) {
						dist *= (float)exp((float)dy * -.03f);
					} 
				} else {
					if (dx || dy) {
						float magn = sqrt(dx * dx + dy * dy);
						float fdx = (float)dx / magn;
						float fdy = (float)dy / magn;
						Quat rotation = Quat(fdy, fdx, 0, magn * M_PI / 180.).fromAngleAxis();
						viewAngle = rotation * viewAngle;
						viewAngle /= Quat::length(viewAngle);
					}
				}
			}
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = true;
		}
		if (event.button.button == SDL_BUTTON_RIGHT) {
			rightButtonDown = true;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = false;
		}
		if (event.button.button == SDL_BUTTON_RIGHT) {
			rightButtonDown = false;
		}
		break;
	case SDL_KEYDOWN:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = true;
		}
		break;
	case SDL_KEYUP:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = false;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = false;
		}
		break;
	}

}

GLAPP_MAIN(NBodyApp)

