#include "CLCommon/CLCommon.h"
#include "CLCommon/cl.hpp"
#include "GLApp/ViewBehavior.h"
#include "GLApp/GLApp.h"
#include "GLCxx/Program.h"
#include "GLCxx/Texture.h"
#include "GLCxx/gl.h"
#include "Profiler/Profiler.h"
#include "Tensor/Vector.h"
#include "Tensor/Quat.h"
#include "Common/Macros.h"
#include "Common/File.h"
#include "Common/Exception.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "nbody.h"

//stupid windows
#ifndef min
#define min std::min
#endif

using Quat = Tensor::quat<float>;

struct NBodyApp : public ::GLApp::ViewBehavior<::GLApp::GLApp> {
	using Super = ::GLApp::ViewBehavior<::GLApp::GLApp>;

	std::shared_ptr<CLCommon::CLCommon> clCommon;
	
	bool hasGLSharing = {};
	bool hasFP64 = {};

	GLuint positionVBO = {};			//position vertex buffer object -- for rendering
	GLCxx::Texture postTex;				//post-processing texture
	GLCxx::Texture gradientTex = {};	//gradient texture
	GLCxx::Texture particleTex = {};	//particle texture
	GLuint fbo = {};

	//with GL sharing
	cl::Memory posMem;
	//without GL sharing
	std::vector<real4> posCPUMem;
	
	cl::Buffer objsMem;	//current contents
	cl::Buffer objsMemPrev;	//previous contents
	cl::Program program;
	cl::Kernel updateKernel;
	cl::Kernel copyToGLKernel;
	cl::Kernel initDataKernel;

	GLCxx::Program particleShader;

	int count = 16384;

	cl::NDRange globalSize;
	cl::NDRange localSize;

	Tensor::int2 screenBufferSize = {1024, 1024};
	Quat viewAngle;
	float dist = 1;
	bool leftShiftDown = {};
	bool rightShiftDown = {};
	bool leftButtonDown = {};
	bool rightButtonDown = {};

	virtual std::string getTitle() { return "N-Body GPU"; }

	virtual void init(const Init& args);
	~NBodyApp();

	virtual void onUpdate();
	virtual void onSDLEvent(SDL_Event &event);
};

//TODO put this in CLCommon, where a similar function operating on vectors exists
static auto checkHasGLSharing = [](const cl::Device& device)-> bool {
	std::vector<std::string> extensions = CLCommon::getExtensions(device);
	return std::find(extensions.begin(), extensions.end(), "cl_khr_gl_sharing") != extensions.end()
		|| std::find(extensions.begin(), extensions.end(), "cl_APPLE_gl_sharing") != extensions.end();
};

static auto checkHasFP64 = [](const cl::Device& device)-> bool {
	std::vector<std::string> extensions = CLCommon::getExtensions(device);
	return std::find(extensions.begin(), extensions.end(), "cl_khr_fp64") != extensions.end();
};

void NBodyApp::init(const Init& args) {
	Super::init(args);

	clCommon = std::make_shared<CLCommon::CLCommon>(
		/*useGPU=*/true,	
		/*verbose=*/true,	
		/*pickDevice=*/[&](const std::vector<cl::Device>& devices_) -> std::vector<cl::Device>::const_iterator {
			
			//sort with a preference to sharing
			std::vector<cl::Device> devices = devices_;
			std::sort(
				devices.begin(),
				devices.end(),
				[&](const cl::Device& a, const cl::Device& b) -> bool {
					return (checkHasGLSharing(a) + checkHasFP64(a)) 
						> (checkHasGLSharing(b) + checkHasFP64(b));
				});

			cl::Device best = devices[0];
			//return std::find<std::vector<cl::Device>::const_iterator, cl::Device>(devices_.begin(), devices_.end(), best);
			for (std::vector<cl::Device>::const_iterator iter = devices_.begin(); iter != devices_.end(); ++iter) {
				if ((*iter)() == best()) return iter;
			}
			throw Common::Exception() << "couldn't find a device";
		});
	
	hasGLSharing = checkHasGLSharing(clCommon->device);
std::cout << "hasGLSharing " << hasGLSharing << std::endl; 
	hasFP64 = checkHasFP64(clCommon->device);
std::cout << "hasFP64 " << hasFP64 << std::endl;


	size_t maxWorkGroupSize = clCommon->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::vector<size_t> maxWorkItemSizes = clCommon->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	std::cout << "max work group size " << maxWorkGroupSize << std::endl;
	std::cout << "max work item sizes";
	for (size_t size : maxWorkItemSizes) { std::cout << " " << size; }
	std::cout << std::endl;

	size_t localSizeValue = 16;//min(maxWorkItemSizes[0], count);
	globalSize = cl::NDRange(count);
	localSize = cl::NDRange(localSizeValue);

	glGenBuffers(1, &positionVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionVBO);	
	glBufferData(GL_ARRAY_BUFFER, sizeof(cl_float4) * count, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);	

	if (hasGLSharing) {
		posMem = cl::BufferGL(clCommon->context, CL_MEM_WRITE_ONLY, positionVBO);
	} else {
		posCPUMem.resize(count);
	}

	objsMem = cl::Buffer(clCommon->context, CL_MEM_READ_WRITE, sizeof(Object) * count);
	objsMemPrev = cl::Buffer(clCommon->context, CL_MEM_READ_WRITE, sizeof(Object) * count);
	
	std::stringstream ss;
	ss	
		<< (hasGLSharing ? "#pragma OPENCL EXTENSION cl_khr_gl_sharing : enable\n" : "")
		<< (hasFP64 ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" : "")
		<< "#define USE_GL_SHARING " << (hasGLSharing ? "1" : "0") << "\n"
		<< "#define COUNT " << count << "\n"
		<< "#define INITIAL_RADIUS 50000\n"	//radius of the milky way, in light years
		<< "#include \"nbody.cl\"\n";
	std::string source = ss.str();
	try {
#if defined(CL_HPP_TARGET_OPENCL_VERSION) && CL_HPP_TARGET_OPENCL_VERSION>=200
		program = cl::Program(clCommon->context, std::vector<std::string>{source});
#else
		std::vector<std::pair<const char *, size_t>> sources = {
			std::pair<const char *, size_t>(source.c_str(), source.length())
		};
		program = cl::Program(clCommon->context, sources);
#endif	//CL_HPP_TARGET_OPENCL_VERSION
		program.build({clCommon->device}, "-I include");
	} catch (cl::Error &err) {
		std::cout << source << std::endl;
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
	clCommon->commands.enqueueNDRangeKernel(initDataKernel, cl::NDRange(0), globalSize, localSize);
	clCommon->commands.finish();
	
	updateKernel = cl::Kernel(program, "update");

	copyToGLKernel = cl::Kernel(program, "copyToGL");
	if (hasGLSharing) {
		copyToGLKernel.setArg(0, posMem);
	}

	postTex = GLCxx::Texture2D()
		.bind()
		.setParam<GL_TEXTURE_MIN_FILTER>(GL_LINEAR)
		.setParam<GL_TEXTURE_MAG_FILTER>(GL_LINEAR)
		.create2D(screenBufferSize(0), screenBufferSize(1), GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
		.unbind();
	
	Tensor::int2 particleTexSize(256, 256);
	std::vector<char> particleData(particleTexSize(0) * particleTexSize(1) * 4);
	{
		char *p = &particleData[0];
		for (int y = 0; y < particleTexSize(1); ++y) {
			for (int x = 0; x < particleTexSize(0); ++x) {
				float dx = (float)(x + .5f) / (float)particleTexSize(0) - .5f;
				float dy = (float)(y + .5f) / (float)particleTexSize(1) - .5f;
				float dr2 = dx*dx + dy*dy;
				float lum = exp(-100.f * dr2);
				*p++ = (char)(255.f * min(.5f, lum));
				*p++ = (char)(255.f * min(.5f, lum));
				*p++ = (char)(255.f * min(1.f, lum));
				*p++ = 255;//(char)(255.f * lum);
			}
		}
	}
	particleTex = GLCxx::Texture2D()
		.bind()
		.setParam<GL_TEXTURE_MIN_FILTER>(GL_LINEAR)
		.setParam<GL_TEXTURE_MAG_FILTER>(GL_LINEAR)
		.create2D(particleTexSize(0), particleTexSize(1), GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, particleData.data())
		.generateMipmap()
		.unbind();

	int gradientSize = 1024;
	gradientTex = GLCxx::Texture2D()
		.bind()
		.setParam<GL_TEXTURE_MIN_FILTER>(GL_NEAREST)
		.setParam<GL_TEXTURE_MAG_FILTER>(GL_LINEAR)
		.create2D(gradientSize, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
		.unbind();

#if 0
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#endif

	std::string particleShaderSource = Common::File::read("particle.shader");
	std::vector<GLCxx::Shader> shaders = {
		GLCxx::VertexShader(std::vector<std::string>{"#define VERTEX_SHADER\n", particleShaderSource}),
		GLCxx::FragmentShader(std::vector<std::string>{"#define FRAGMENT_SHADER\n", particleShaderSource}),
	};
	particleShader = GLCxx::Program(shaders)
		.setUniform<int>("tex", 0)
		.done();
	
	int err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;
}

NBodyApp::~NBodyApp() {
#if 0
	glDeleteFramebuffers(1, &fbo);
#endif
}

void NBodyApp::onUpdate() {
PROFILE_BEGIN_FRAME()
	copyToGLKernel.setArg(1, objsMem);
	if (hasGLSharing) {
		glFinish();

		std::vector<cl::Memory> glObjects = {posMem};
		clCommon->commands.enqueueAcquireGLObjects(&glObjects);

		//copy CL to GL
		clCommon->commands.enqueueNDRangeKernel(copyToGLKernel, cl::NDRange(0), globalSize, localSize);
		
		clCommon->commands.enqueueReleaseGLObjects(&glObjects);
		clCommon->commands.finish();
	} else {
		copyToGLKernel.setArg(0, objsMemPrev);	//use prev as a temp buffer for packing data
		clCommon->commands.enqueueNDRangeKernel(copyToGLKernel, cl::NDRange(0), globalSize, localSize);
		
		clCommon->commands.enqueueReadBuffer(objsMemPrev, GL_TRUE, 0, sizeof(Object) * count, posCPUMem.data());
		clCommon->commands.finish();

		//now upload to the VBO
		glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
		glBufferSubData(GL_ARRAY_BUFFER_ARB, 0, sizeof(cl_float4) * count, posCPUMem.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glFinish();
	}

	//update CL state
	updateKernel.setArg(0, objsMemPrev);	//write new state over old state
	updateKernel.setArg(1, objsMem);
	clCommon->commands.enqueueNDRangeKernel(updateKernel, cl::NDRange(0), globalSize, localSize);
	
#if PLATFORM_MSVC
	{
		cl::Buffer tmp = objsMem;
		objsMem = objsMemPrev;
		objsMemPrev = tmp;
	}
#else
	std::swap<cl::Memory>(objsMem, objsMemPrev);
#endif

#if 0
	//render to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postTex(), 0);
	int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE) throw Common::Exception() << "check framebuffer status " << status;

	//render GL
	glViewport(0, 0, screenBufferSize(0), screenBufferSize(1));
#endif
	glClear(GL_COLOR_BUFFER_BIT);
	
	const float zNear = .01;
	const float zFar = 10;
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
	glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, Tensor::float3(0.f, 1.f, 0.f).s.data());
	glPointParameterf(GL_POINT_SIZE_MIN, 1.f);
	glPointParameterf(GL_POINT_SIZE_MAX, 128.f);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	particleTex.bind();
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

	postTex
		.bind()
		.generateMipmap()
		.unbind();
#if 0
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//render to screen
	glViewport(0, 0, screenSize(0), screenSize(1));
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	postTex.enable().bind().generateMipmap();
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

	postTex.unbind().disable();
#endif
PROFILE_END_FRAME()
}

void NBodyApp::onSDLEvent(SDL_Event &event) {
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
						viewAngle = (rotation * viewAngle).normalize();
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
