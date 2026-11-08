// Animated sky with clouds, drawn behind the scene in place of the black background.
//
// Cloud shader: https://www.shadertoy.com/view/4tdSWr by drift.
// Shadertoy shaders are licensed CC BY-NC-SA 3.0 unless the author says otherwise:
// https://creativecommons.org/licenses/by-nc-sa/3.0/
// Changes: wrapped for WebGL2 (version and precision lines, the iResolution and iTime uniforms
// Shadertoy normally provides, and a main() that calls mainImage), and the two sky colours get a
// small random tint on each page load.

const CLOUDS_VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
void main() {
  // The plane spans -0.5..0.5; doubled, it covers the whole canvas (-1..1 in clip space)
  gl_Position = vec4(aPosition.xy * 2.0, 0.0, 1.0);
}
`;

// GLSL has no Math.random, so the random sky tint is worked out here in JavaScript and written
// into the shader text with \${...}. toFixed(3) always gives a decimal point (e.g. 0.247),
// which GLSL needs for a float.
const skyTint = base => (base + Math.random() * 0.1).toFixed(3);

const CLOUDS_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec3 iResolution;   // canvas size in pixels (Shadertoy's name for it)
uniform float iTime;        // seconds since the page started (Shadertoy's name for it)
out vec4 outColor;

const float cloudscale = 1.1;
const float speed = 0.03;
const float clouddark = 0.5;
const float cloudlight = 0.3;
const float cloudcover = 0.2;
const float cloudalpha = 8.0;
const float skytint = 0.5;
const vec3 skycolour1 = vec3(${skyTint(0.2)}, ${skyTint(0.4)}, ${skyTint(0.6)});   // random tint, picked in JavaScript on each page load
const vec3 skycolour2 = vec3(${skyTint(0.4)}, ${skyTint(0.7)}, 1.0);

const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );

vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

float fbm(vec2 n) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
		total += noise(n) * amplitude;
		n = m * n;
		amplitude *= 0.4;
	}
	return total;
}

// -----------------------------------------------

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 p = fragCoord.xy / iResolution.xy;
	vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0);    
    float time = iTime * speed;
    float q = fbm(uv * cloudscale * 0.5);
    
    //ridged noise shape
	float r = 0.0;
	uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i=0; i<8; i++){
		r += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.7;
    }
    
    //noise shape
	float f = 0.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i=0; i<8; i++){
		f += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    f *= r + f;
    
    //noise colour
    float c = 0.0;
    time = iTime * speed * 2.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale*2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c += weight*noise( uv );
        uv = m*uv + time;
		weight *= 0.6;
    }
    
    //noise ridge colour
    float c1 = 0.0;
    time = iTime * speed * 3.0;
    uv = p*vec2(iResolution.x/iResolution.y,1.0);
	uv *= cloudscale*3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i=0; i<7; i++){
		c1 += abs(weight*noise( uv ));
        uv = m*uv + time;
		weight *= 0.6;
    }
	
    c += c1;
    
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * clamp((clouddark + cloudlight*c), 0.0, 1.0);
   
    f = cloudcover + cloudalpha*f*r;
    
    vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), clamp(f + c, 0.0, 1.0));
    
	fragColor = vec4( result, 1.0 );
}

void main() {
  mainImage(outColor, gl_FragCoord.xy);
}
`;

// Build the cloud program. quad is a mesh from createMesh (the plane), drawn over the whole
// canvas. If the shader fails to build, the error goes to the console and draw does nothing,
// so the rest of the scene still runs.
function createClouds(gl, quad) {
  try {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, CLOUDS_VERTEX_SHADER));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, CLOUDS_FRAGMENT_SHADER));
    gl.bindAttribLocation(program, 0, "aPosition");   // where the plane's positions are
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));

    const uResolution = gl.getUniformLocation(program, "iResolution");
    const uTime = gl.getUniformLocation(program, "iTime");

    return {
      // Draw the sky over the whole canvas. With the depth test off, it also writes no depth,
      // so everything drawn afterwards appears in front of it.
      draw(seconds) {
        gl.useProgram(program);
        gl.uniform3f(uResolution, gl.canvas.width, gl.canvas.height, 1);
        gl.uniform1f(uTime, seconds);
        gl.disable(gl.DEPTH_TEST);
        gl.bindVertexArray(quad.vao);
        gl.drawElements(gl.TRIANGLES, quad.count, gl.UNSIGNED_SHORT, 0);
        gl.enable(gl.DEPTH_TEST);
      }
    };
  } catch (e) {
    console.error("Clouds shader failed:", e.message);
    return { draw() {} };
  }
}
