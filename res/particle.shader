varying float mass;
varying float distSq;

#ifdef VERTEX_SHADER
void main() {
	const float spriteWidth = 10.;
	mass = gl_Vertex.w;
	vec4 vertex = vec4(gl_Vertex.xyz, 1.);
	vec4 eyeVertex = gl_ModelViewMatrix * vertex;
	distSq = dot(eyeVertex.xyz, eyeVertex.xyz);
	gl_Position = gl_ModelViewProjectionMatrix * eyeVertex;
	gl_PointSize = spriteWidth / gl_Position.w;
}
#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER
uniform sampler2D tex;
#define LOG_10 2.302585092994
void main() {
	//float lumForMass = (log(mass) / LOG_10 + 1.) / 5.;	//mass is 10^0 to 10^4
	//float lum = lumForMass * .3 / distSq;

	float lum = 1.;//mass * .0001;// / distSq;

	gl_FragColor = texture2D(tex, gl_TexCoord[0].xy) * lum;
}
#endif	//FRAGMENT_SHADER

