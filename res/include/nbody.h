#pragma once

//radius of the milky way, in light years
#define INITIAL_RADIUS	50000
#define COUNT			8192

typedef float real;
#ifdef __OPENCL_VERSION__
typedef float3 real3;
#else
typedef cl_float3 real3;
#endif

typedef struct {
	real3 pos;
	real3 vel;
	real mass;
} Object;

