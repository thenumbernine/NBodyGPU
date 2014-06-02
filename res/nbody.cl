#include "nbody.h"

/*
x'_i = v_i
v'_i = a_i = F_i / m_i = sum_j~=i G m_j * (x_j - x_i) / |x_j - x_i|^3
... versus G [sum_j m_j] [[sum_j (m_j x_j) / sum_j m_j] - x_i] / [[sum_j (m_j x_j) / sum_j m_j] - x_i|^3
*/

/*
const float gravityConstantInM3PerKgS2 = 6.67384e-11;	//G in terms of m^e / kg s^2
const float lyrPerM = 1.0570234110814e-16;				//light years per meter
const float sPerYr = 3.15569e7;							//seconds per year
const float yrPerGyr = 2.25e8;							//galactic years per year
const float sPerGyr = sPerYr * yrPerGyr;				//seconds per galactic year
const float kgPerSm = 1.9891e30;						//kilograms per solar mass
const float gravityConstantInLyr3PerSmGyr2 = gravityConstantInM3PerKgS2 * lyrPerM * lyrPerM * lyrPerM * sPerGyr * sPerGyr * kgPerSm;
*/
#define GRAVITY_CONSTANT	7903.8725760201f			//in case the above is beyond the precision of the compiler 
#define DT	.1f										//in galactic years
#define EPS	1e+8f										//in light years

real3 gravity(real3 posA, real3 posB) {
	real3 dx = posA - posB;
	real invLen = rsqrt(dot(dx,dx) + EPS);
	real invLen3 = invLen * invLen * invLen;
	return -dx * invLen3;
}

__kernel void initData(
	__global Object *objs,
	__global float *randBuffer,
	int count)
{
	int i = get_global_id(0);
	if (i >= count) return;
	__global Object *obj = objs + i;
	
	int j = i;
#define FRAND()		(randBuffer[j=(j+104729)%count])
#define CRAND()		(FRAND() * 2. - 1.)
#define M_PI		3.1415926535898
#define TOTAL_MASS		10000000
	obj->mass = mix(100., 10000., FRAND());//pow(10., FRAND() * 4.);
	float density = .5 * sqrt(-log(1 - FRAND()));
	float cbrtMm = cbrt(TOTAL_MASS / FRAND());
	float a = 100. * INITIAL_RADIUS / sqrt(sqrt(2.) - 1.);
	float radius = a / sqrt(cbrtMm * cbrtMm - 1.);
	float phi = 2. * M_PI * FRAND();
	float theta = asin(2. * FRAND() - 1.) + M_PI*.5;
	float3 dir = (float3)(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
	obj->pos = dir * radius;

/*
 radial density profile: rho(r) = 3(v(r))^2 / (4 pi G r^2)	<- by Kepler's 3rd law
 v(r) = 4 pi G r^2 rho(r) / 3
*/

	obj->vel = (real3)(-dir.y, dir.x, 0.) * sqrt(2. * GRAVITY_CONSTANT * TOTAL_MASS / radius) * FRAND();
#undef FRAND
}
/*
F1 = G m1 m2 / r^2 = m1 a1
a1 = G m2 / r^2

for orbits
x = (r cos wt, r sin wt)
x' = (-rw sin wt, rw cos wt)
x'' = (-rw^2 cos wt, -rw^2 sin wt) = -w^2 x
|v| = |x'| = rw
|a| = w^2 |x| = w^2 r = G M / r^2		<- for 'M' the mass of all other bodies
G M = w^2 r^3							<- 1-2-3 law
w = sqrt(G M / r^3)
|v| = r w = sqrt(G M / r)
*/

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
	dst->xyz = srcObj->pos.xyz / INITIAL_RADIUS;
	dst->w = srcObj->mass;
}

