#pragma once

typedef float real;
#ifdef __OPENCL_VERSION__
typedef float3 real3;
typedef float4 real4;
#else
typedef cl_float3 real3;
typedef cl_float4 real4;
#endif

typedef struct {
	real3 pos;
	real3 vel;
	real mass;
} Object;
