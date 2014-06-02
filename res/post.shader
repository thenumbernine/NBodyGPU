#ifdef VERTEX_SHADER
varying vec2 pos;
void main() {
	pos.xy = gl_Vertex.xy;
	gl_Position = ftransform();
}
#endif	//VERTEX_SHADER

#ifdef FRAGMENT_SHADER
varying vec2 pos;
uniform sampler2D tex;
void main() {
	gl_FragColor = texture2D(tex, pos);
}
#endif	//FRAGMENT_SHADER

