#include "CLCommon/CLCommon.h"
#include "GLApp/GLApp.h"
#include "Tensor/Vector.h"
#include "Profiler/Profiler.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include "Shader/Program.h"
#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "nbody.h"
#include "Quat.h"

struct NBodyApp : public ::GLApp::GLApp {
	typedef ::GLApp::GLApp Super;

	std::shared_ptr<CLCommon::CLCommon> clCommon;

	GLuint positionVBO;	//position vertex buffer object -- for rendering
	GLuint postTex;		//post-processing texture
	GLuint gradientTex;	//gradient texture
	GLuint particleTex;	//particle texture
	GLuint fbo;

	cl::Memory posMem;
	
	cl::Buffer objsMem;	//current contents
	cl::Buffer objsMemPrev;	//previous contents
	cl::Program program;
	cl::Kernel updateKernel;
	cl::Kernel copyToGLKernel;
	cl::Kernel initDataKernel;

	Shader::Program particleShader;

	int count;

	cl::NDRange globalSize;
	cl::NDRange localSize;

	Tensor::Vector<int,2> screenBufferSize;
	Tensor::Vector<int,2> viewportSize;
	Quat viewAngle;
	float dist;
	bool leftShiftDown;
	bool rightShiftDown;
	bool leftButtonDown;
	bool rightButtonDown;

	NBodyApp();

	virtual void init();
	virtual void shutdown();
	virtual void update();
	virtual void resize(int width, int height);
	virtual void sdlEvent(SDL_Event &event);
};

NBodyApp::NBodyApp()
: Super()
, positionVBO(0)
, postTex(0)
, gradientTex(0)
, particleTex(0)
, fbo(0)
, count(COUNT)
, screenBufferSize(1024, 1024)
, dist(1.f)
, leftShiftDown(false)
, rightShiftDown(false)
, leftButtonDown(false)
, rightButtonDown(false)
{}

void NBodyApp::init() {
	Super::init();

	clCommon = std::make_shared<CLCommon::CLCommon>(true);

	size_t maxWorkGroupSize = clCommon->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = clCommon->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	std::cout << "max work group size " << maxWorkGroupSize << std::endl;
	std::cout << "max work item sizes";
	for (size_t size : maxWorkItemSizes) { std::cout << " " << size; }
	std::cout << std::endl;

	size_t localSizeValue = 16;//std::min<size_t>(maxWorkItemSizes[0], count);
	globalSize = cl::NDRange(count);
	localSize = cl::NDRange(localSizeValue);

	glGenBuffers(1, &positionVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);	
	glBufferData(GL_ARRAY_BUFFER, sizeof(cl_float4) * count, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);	
	
	posMem = cl::BufferGL(clCommon->context, CL_MEM_WRITE_ONLY, positionVBO);

	objsMem = cl::Buffer(clCommon->context, CL_MEM_READ_WRITE, sizeof(Object) * count);
	objsMemPrev = cl::Buffer(clCommon->context, CL_MEM_READ_WRITE, sizeof(Object) * count);
	
	try {
		std::string source = Common::File::read("nbody.cl");
		std::vector<std::pair<const char *, size_t>> sources = {
			std::pair<const char *, size_t>(source.c_str(), source.length())
		};
		program = cl::Program(clCommon->context, sources);
		program.build({clCommon->device}, "-I include");
	} catch (cl::Error &err) {
		std::cout << "failed to build program executable!" << std::endl;
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clCommon->device) << std::endl;
		throw;
	}

	std::vector<float> randBuffer(count);
	for (float &f : randBuffer) { f = frand(); };
	cl::Buffer randMem = cl::Buffer(clCommon->context, CL_MEM_READ_ONLY, sizeof(float) * count);
	clCommon->commands.enqueueWriteBuffer(randMem, CL_TRUE, 0, sizeof(float) * count, &randBuffer[0]);	
	
	initDataKernel = cl::Kernel(program, "initData");
	initDataKernel.setArg(0, objsMem);
	initDataKernel.setArg(1, randMem);
	initDataKernel.setArg(2, count);
	clCommon->commands.enqueueNDRangeKernel(initDataKernel, cl::NDRange(0), globalSize, localSize);
	clCommon->commands.finish();
	
	updateKernel = cl::Kernel(program, "update");
	updateKernel.setArg(2, count);

	copyToGLKernel = cl::Kernel(program, "copyToGL");
	copyToGLKernel.setArg(0, posMem);
	copyToGLKernel.setArg(2, count);

	glGenTextures(1, &postTex);
	glBindTexture(GL_TEXTURE_2D, postTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenBufferSize(0), screenBufferSize(1), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	
	glGenTextures(1, &particleTex);
	glBindTexture(GL_TEXTURE_2D, particleTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	Tensor::Vector<int,2> particleTexSize(256, 256);
	std::vector<char> particleData(particleTexSize(0) * particleTexSize(1) * 4);
	{
		char *p = &particleData[0];
		for (int y = 0; y < particleTexSize(1); ++y) {
			for (int x = 0; x < particleTexSize(0); ++x) {
				float dx = (float)(x + .5f) / (float)particleTexSize(0) - .5f;
				float dy = (float)(y + .5f) / (float)particleTexSize(1) - .5f;
				float dr2 = dx*dx + dy*dy;
				float lum = exp(-100.f * dr2);
				*p++ = (char)(255.f * std::min(.5f, lum));
				*p++ = (char)(255.f * std::min(.5f, lum));
				*p++ = (char)(255.f * std::min(1.f, lum));
				*p++ = 255;//(char)(255.f * lum);
			}
		}
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, particleTexSize(0), particleTexSize(1), 0, GL_RGBA, GL_UNSIGNED_BYTE, &particleData[0]);
	glGenerateMipmap(GL_TEXTURE_2D);

	glGenTextures(1, &gradientTex);
	glBindTexture(GL_TEXTURE_2D, gradientTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	int gradientSize = 1024;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gradientSize, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);

#if 0
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif

	std::string particleShaderSource = Common::File::read("particle.shader");
	std::vector<Shader::Shader> shaders = {
		Shader::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", particleShaderSource}),
		Shader::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", particleShaderSource}),
	};
	particleShader = Shader::Program(shaders)
		.link()
		.setUniform<int>("tex", 0)
		.done();
	
	int err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;
}

void NBodyApp::shutdown() {
#if 0
	glDeleteFramebuffers(1, &fbo);
#endif
	glDeleteTextures(1, &postTex);
	glDeleteTextures(1, &gradientTex);
	glDeleteTextures(1, &particleTex);
	glDeleteBuffers(1, &positionVBO);
}

void NBodyApp::update() {
PROFILE_BEGIN_FRAME()
	glFinish();

	std::vector<cl::Memory> glObjects = {posMem};
	clCommon->commands.enqueueAcquireGLObjects(&glObjects);

	//copy CL to GL
	copyToGLKernel.setArg(1, objsMem);
	clCommon->commands.enqueueNDRangeKernel(copyToGLKernel, cl::NDRange(0), globalSize, localSize);
	
	clCommon->commands.enqueueReleaseGLObjects(&glObjects);
	clCommon->commands.finish();

	//update CL state
	updateKernel.setArg(0, objsMemPrev);	//write new state over old state
	updateKernel.setArg(1, objsMem);
	clCommon->commands.enqueueNDRangeKernel(updateKernel, cl::NDRange(0), globalSize, localSize);
	std::swap<cl::Memory>(objsMem, objsMemPrev);

#if 0
	//render to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postTex, 0);
	int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) throw Common::Exception() << "check framebuffer status " << status;

	//render GL
	glViewport(0, 0, screenBufferSize(0), screenBufferSize(1));
#endif
	glClear(GL_COLOR_BUFFER_BIT);
	
	const float zNear = .01;
	const float zFar = 10;
	float aspectRatio = (float)viewportSize(0) / (float)viewportSize(1);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-aspectRatio * zNear, aspectRatio * zNear, -zNear, zNear, zNear, zFar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0,0,-dist);
	Quat angleAxis = viewAngle.toAngleAxis();
	glRotatef(angleAxis(3) * 180. / M_PI, angleAxis(0), angleAxis(1), angleAxis(2));

	glPointSize(20.f);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, Tensor::Vector<float,3>(0.f, 1.f, 0.f).v);
	glPointParameterf(GL_POINT_SIZE_MIN, 1.f);
	glPointParameterf(GL_POINT_SIZE_MAX, 128.f);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBindTexture(GL_TEXTURE_2D, particleTex);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	
	particleShader.use();
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, count);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	particleShader.done();
	
	glDisable(GL_BLEND);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);

	glBindTexture(GL_TEXTURE_2D, postTex);
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
#if 0
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//render to screen
	glViewport(0, 0, viewportSize(0), viewportSize(1));
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, postTex);
	glGenerateMipmap(GL_TEXTURE_2D);
#if 0
	//first layer...
	glBegin(GL_QUADS);
	glTexCoord2f(0,0);	glVertex2f(0,0);
	glTexCoord2f(1,0);	glVertex2f(1,0);
	glTexCoord2f(1,1);	glVertex2f(1,1);
	glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
#endif
	//2nd layer... isn't blurring like it should be ...
	glTexEnvf(GL_TEXTURE_FILTER_CONTROL, GL_TEXTURE_LOD_BIAS, -1000.f);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0);	glVertex2f(0,0);
	glTexCoord2f(1,0);	glVertex2f(1,0);
	glTexCoord2f(1,1);	glVertex2f(1,1);
	glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	glTexEnvf(GL_TEXTURE_FILTER_CONTROL, GL_TEXTURE_LOD_BIAS, 0.f);

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
#endif
PROFILE_END_FRAME()
}

void NBodyApp::resize(int width, int height) {
	Super::resize(width, height);
	viewportSize(0) = width;
	viewportSize(1) = height;
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

