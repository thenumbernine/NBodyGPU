#ifdef VERTEX_SHADER
varying vec2 pos;
void main() {
	pos.xy = gl_Vertex.xy;
	gl_Position = ftransform();
}
#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER
varying vec2 pos;
uniform vec2 dx;
uniform sampler2D tex;
void main() {
	gl_FragColor = 
		(2. * texture2D(tex, pos + vec2(1.5 * dx.x, 0.)) +
		2. * texture2D(tex, pos + vec2(-1.5 * dx.x, 0.)) +
		2. * texture2D(tex, pos + vec2(0., 1.5 * dx.y)) +
		2. * texture2D(tex, pos + vec2(0., -1.5 * dx.y)) +
		4. * texture2D(tex, pos)) / 12.;
}
#endif	//FRAGMENT_SHADER

