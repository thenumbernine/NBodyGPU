#include "nbody.h"

#define GRAVITY_CONSTANT	.0005f
#define DT	.001f
#define EPS	.01f

real3 gravity(real3 posA, real3 posB) {
	real3 dx = posA - posB;
	real invLen = rsqrt(dot(dx,dx) + EPS);
	real invLen3 = invLen * invLen * invLen;
	return -dx * invLen3;
}

__kernel void update(
	__global Object *newObjs,
	__global const Object *oldObjs,
	int count)
{
	int i = get_global_id(0);
	if (i >= count) return;

	__global Object *newObj = newObjs + i;
	*newObj = oldObjs[i];
	
	newObj->pos += newObj->vel * DT;
	
	__global const Object *oldObj = oldObjs;
	for (int j = 0; j < count; ++j, ++oldObj) {
		if (i != j) {
			newObj->vel += gravity(newObj->pos, oldObj->pos) * (oldObj->mass * GRAVITY_CONSTANT * DT);
		}
	}
}

__kernel void copyToGL(
	__global float4 *dsts,
	__global const Object *srcObjs,
	int count)
{
	int i = get_global_id(0);
	if (i >= count) return;

	__global float4 *dst = dsts + i;
	__global const Object *srcObj = srcObjs + i;
	dst->xyz = srcObj->pos.xyz;
	dst->w = srcObj->mass;
}
