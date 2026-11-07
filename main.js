const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aTexCoord;

uniform mat4 uModelView;
uniform mat4 uProjection;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vTexCoord;

void main() {
  vec4 p = uModelView * vec4(aPosition, 1.0);
  vPosition = p.xyz;

  // Inverse transpose for potential non-uniform scalings
  mat3 mv = mat3(uModelView);
  vNormal = transpose(inverse(mv)) * aNormal;


  vTangent = mv * aTangent;
  vTexCoord = aTexCoord;
  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vTexCoord;

uniform vec3 uLightPos;
uniform sampler2D uTexture;
uniform int uMapping;  // 0 = bump mapping, 1 = normal mapping

out vec4 fragColor;

const float bumpScale = 3.0;

// convert RGB to luminance (Rec._601 standard, https://en.wikipedia.org/wiki/Rec._601)
// or you can use a height map texture
float height(vec2 uv) {
  vec3 c = texture(uTexture, uv).rgb;
  return 1.0 - dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
  // Tangent frame (T, B, N)
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent - dot(vTangent, N) * N);
  vec3 B = cross(N, T);
  mat3 TBN = mat3(T, B, N);

  vec3 baseColor;
  vec3 n;

  if (uMapping == 0) {
    // Bump mapping: tilt the normal against the gradient of the height field
    vec2 du = vec2(1.0 / float(textureSize(uTexture, 0).x), 0.0);
    vec2 dv = vec2(0.0, 1.0 / float(textureSize(uTexture, 0).y));
    float dhdu = (height(vTexCoord + du) - height(vTexCoord - du)) * 0.5;
    float dhdv = (height(vTexCoord + dv) - height(vTexCoord - dv)) * 0.5;
    n = normalize(vec3(-bumpScale * dhdu, -bumpScale * dhdv, 1.0));
    baseColor = texture(uTexture, vTexCoord).rgb;
  } else {
    // Normal mapping: the texture stores the tangent-space normal in [0, 1]
    n = normalize(texture(uTexture, vTexCoord).rgb * 2.0 - 1.0);
    baseColor = vec3(0.85, 0.65, 0.3);
  }

  vec3 Ka = 0.2 * baseColor;
  vec3 Kd = 0.8 * baseColor;
  vec3 Ks = vec3(0.5);
  float shininess = 32.0;

  N = normalize(TBN * n);
  vec3 L = normalize(uLightPos - vPosition);
  vec3 V = normalize(-vPosition);
  vec3 R = reflect(-L, N);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(V, R), 0.0), shininess);

  fragColor = vec4(Ka + Kd * diff + Ks * spec, 1.0);
}
`;

// WebGL setup
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl2");

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

const ATTRIBUTES = { aPosition: 0, aNormal: 1, aTexCoord: 2, aTangent: 3 };

function createProgram(vsSource, fsSource) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
  for (const name in ATTRIBUTES) gl.bindAttribLocation(prog, ATTRIBUTES[name], name);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  return prog;
}

// Shader editors
const vertEditor = document.getElementById("vertEditor");
const fragEditor = document.getElementById("fragEditor");
const shaderError = document.getElementById("shaderError");
vertEditor.value = VERTEX_SHADER;
fragEditor.value = FRAGMENT_SHADER;

let program, uModelView, uProjection, uLightPos, uMapping;

function updateProgram() {
  try {
    const prog = createProgram(vertEditor.value, fragEditor.value);
    gl.deleteProgram(program);
    program = prog;
    gl.useProgram(program);
    uModelView = gl.getUniformLocation(program, "uModelView");
    uProjection = gl.getUniformLocation(program, "uProjection");
    uLightPos = gl.getUniformLocation(program, "uLightPos");
    uMapping = gl.getUniformLocation(program, "uMapping");
    gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
    shaderError.textContent = "";
  } catch (e) {
    shaderError.textContent = e.message.replace(/\0/g, "");
  }
}
updateProgram();
vertEditor.addEventListener("input", updateProgram);
fragEditor.addEventListener("input", updateProgram);

function bindAttribute(name, data, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.vertexAttribPointer(ATTRIBUTES[name], size, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(ATTRIBUTES[name]);
}

function createMesh({ positions, normals, tangents, texCoords, indices }) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  bindAttribute("aPosition", positions, 3);
  bindAttribute("aNormal", normals, 3);
  bindAttribute("aTangent", tangents, 3);
  bindAttribute("aTexCoord", texCoords, 2);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vao, count: indices.length };
}

const cubeMesh = createMesh(createCube());
const cylinderMesh = createMesh(createCylinder());

function loadTexture(url) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 255, 255]));

  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);  // images start at the top row, texture coordinates at the bottom
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  };
  image.src = url;
  return texture;
}

const brickTexture = loadTexture("brick.png");
const normalTexture = loadTexture("texture2.jpg");

const BUMP_MAPPING = 0, NORMAL_MAPPING = 1;
const objects = [
  { mesh: cubeMesh, texture: brickTexture, mapping: BUMP_MAPPING, offset: [-1.8, 0, 0] },
  { mesh: cylinderMesh, texture: normalTexture, mapping: NORMAL_MAPPING, offset: [1.8, 0, 0] }
];

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0, 0, 0, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// Interactive controls
let dragRotX = 0, dragRotY = -0.5;      // mouse drag
let camX = 0, camY = 0, camZ = -7;      // arrow keys, w, s
let sliderAngle = 0;                    // degrees
let spinning = false, spinAngle = 0;    // Start/Stop button

let mouseDown = false, lastX, lastY;
canvas.addEventListener("mousedown", e => { mouseDown = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => mouseDown = false);
window.addEventListener("mousemove", e => {
  if (!mouseDown) return;
  dragRotY += (e.clientX - lastX) * 0.01;
  dragRotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener("keydown", e => {
  if (e.target === vertEditor || e.target === fragEditor) return;
  const step = 0.2;
  switch (e.key) {
    case "ArrowUp": camY -= step; break;
    case "ArrowDown": camY += step; break;
    case "ArrowLeft": camX += step; break;
    case "ArrowRight": camX -= step; break;
    case "w": camZ += step; break;
    case "s": camZ -= step; break;
    default: return;
  }
  e.preventDefault();
});

const axisSelect = document.getElementById("axisSelect");
document.getElementById("slider").addEventListener("input", e => {
  sliderAngle = Number(e.target.value);
  document.getElementById("sliderValue").textContent = sliderAngle;
});
document.getElementById("spinButton").addEventListener("click", () => spinning = !spinning);

// Render loop
let lastTime = 0;
function render(time) {
  const t = time / 1000;
  if (spinning) spinAngle += t - lastTime;
  lastTime = t;
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const view = translate(camX, camY, camZ);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);

  gl.activeTexture(gl.TEXTURE0);
  for (const { mesh, texture, mapping, offset } of objects) {
    gl.uniformMatrix4fv(uModelView, false, multiply(view, multiply(translate(...offset), rotation)));
    gl.uniform1i(uMapping, mapping);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
