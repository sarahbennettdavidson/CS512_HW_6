// Show any error on the page, in the red box under the shader editors, so a black canvas
// always comes with a reason (no need to open the console)
window.addEventListener("error", e => {
  const box = document.getElementById("shaderError");
  if (box) box.textContent += `${e.message} (${(e.filename || "").split("/").pop()}, line ${e.lineno})\n`;
});

const VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec2 aTexCoord;

uniform mat4 uParentView;  // view * the space the object sits in (rotation + translation only)
uniform mat4 uLocal;       // the object's transform inside that space; identity for top-level objects
uniform mat4 uProjection;

out vec3 vPosition;
out vec3 vNormal;
out vec3 vTangent;
out vec2 vTexCoord;
out vec3 vObjPos;      // position in the parent's space, used by triplanar mapping
out vec3 vObjNormal;   // normal in the parent's space

void main() {
  vec4 objPos = uLocal * vec4(aPosition, 1.0);
  vec4 p = uParentView * objPos;
  vPosition = p.xyz;

  // Inverse transpose for potential non-uniform scalings
  mat3 mv = mat3(uParentView * uLocal);
  vNormal = transpose(inverse(mv)) * aNormal;
  vTangent = mv * aTangent;
  vTexCoord = aTexCoord;

  vObjPos = objPos.xyz;
  vObjNormal = transpose(inverse(mat3(uLocal))) * aNormal;

  gl_Position = uProjection * p;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec3 vPosition;
in vec3 vNormal;
in vec3 vTangent;
in vec2 vTexCoord;
in vec3 vObjPos;
in vec3 vObjNormal;

uniform vec3 uLightPos;
uniform sampler2D uTexture;
uniform highp mat4 uParentView;  // highp to match the vertex shader, or the program won't link
uniform int uMapping;  // 0 = bump mapping, 1 = normal mapping, 2 = video (unlit), 3 = triplanar material
uniform vec4 uVideoCrop;  // part of the video frame shown on a screen: (u0, v0, u1, v1)
uniform ivec3 uScreenOrient;  // x: 1 = upside down, y: 1 = mirrored left to right, z: quarter turns (0 to 3)

// Material maps for triplanar mode (uTexture is the color map). uHasNormal and uHasHeight say
// which of the other two the material has.
uniform sampler2D uNormalMap;     // tangent-space normals, OpenGL style (green points up); flat if missing
uniform sampler2D uHeightMap;     // heights in red; if missing (and no normal map), the color's darkness is used
uniform int uHasNormal, uHasHeight;
uniform float uNormalStrength;    // 1 = the normal map as it is
uniform float uHeightDepth;       // how deep the height map's full range is, in world units

out vec4 fragColor;

const float bumpScale = 3.0;
const float triScale = 0.5;   // texture copies per unit; 0.5 = one copy per 2-unit cube face
const vec3 LUMA = vec3(0.299, 0.587, 0.114);

// convert RGB to luminance (Rec._601 standard, https://en.wikipedia.org/wiki/Rec._601)
// or you can use a height map texture
float height(vec2 uv) {
  vec3 c = texture(uTexture, uv).rgb;
  return 1.0 - dot(c, vec3(0.299, 0.587, 0.114));
}

// Slope of the height field at uv (dh/du, dh/dv), by central differences one texel each way
vec2 heightSlope(vec2 uv) {
  vec2 texel = 1.0 / vec2(textureSize(uTexture, 0));
  float dhdu = (height(uv + vec2(texel.x, 0.0)) - height(uv - vec2(texel.x, 0.0))) * 0.5;
  float dhdv = (height(uv + vec2(0.0, texel.y)) - height(uv - vec2(0.0, texel.y))) * 0.5;
  return vec2(dhdu, dhdv);
}

// Height map slope per unit of uv. Displacement maps are often 16-bit with a narrow range, and the
// browser reduces them to 8 bits, leaving only a few levels; one-texel differences of those would
// show as steps. So this reads a blurrier mip level (bias 2) and measures 4 texels each way.
vec2 heightMapSlope(vec2 uv) {
  vec2 d = 4.0 / vec2(textureSize(uHeightMap, 0));
  float hu = texture(uHeightMap, uv + vec2(d.x, 0.0), 2.0).r - texture(uHeightMap, uv - vec2(d.x, 0.0), 2.0).r;
  float hv = texture(uHeightMap, uv + vec2(0.0, d.y), 2.0).r - texture(uHeightMap, uv - vec2(0.0, d.y), 2.0).r;
  return vec2(hu / (2.0 * d.x), hv / (2.0 * d.y));
}

// Slope of the color map's darkness per unit of uv, for materials with no height or normal map
vec2 luminanceSlope(vec2 uv) {
  vec2 t = 1.0 / vec2(textureSize(uTexture, 0));
  float hu = dot(texture(uTexture, uv - vec2(t.x, 0.0)).rgb - texture(uTexture, uv + vec2(t.x, 0.0)).rgb, LUMA);
  float hv = dot(texture(uTexture, uv - vec2(0.0, t.y)).rgb - texture(uTexture, uv + vec2(0.0, t.y)).rgb, LUMA);
  return vec2(hu / (2.0 * t.x), hv / (2.0 * t.y));
}

// Add one projector's share (weight w) of the material at uv. T and B are the directions in the
// parent's space that the texture's u and v run along for this projector. The surface's slope is
// collected as a 3D vector along T and B, which blends correctly between projectors.
void project(vec2 uv, vec3 T, vec3 B, float w, inout vec3 color, inout vec3 slope) {
  color += w * texture(uTexture, uv).rgb;

  vec2 s = vec2(0.0);   // (dh/du, dh/dv): how much the surface rises along u and along v
  if (uHasNormal == 1) {
    vec3 n = texture(uNormalMap, uv).rgb * 2.0 - 1.0;
    s += uNormalStrength * -n.xy / max(n.z, 0.2);   // a normal (-dh/du, -dh/dv, 1), back to slopes
  }
  if (uHasHeight == 1) s += uHeightDepth * triScale * heightMapSlope(uv);
  else if (uHasNormal == 0) s += uHeightDepth * triScale * luminanceSlope(uv);
  slope += w * (s.x * T + s.y * B);
}

void main() {
  // Video: the screen gives off its own light, so show the frame as is with no shading
  if (uMapping == 2) {
    // Orient the picture: a flip runs a coordinate the other way (0.2..0.6 becomes 0.6..0.2),
    // and a quarter turn swaps the two coordinates
    vec2 t = vTexCoord;
    if (uScreenOrient.y == 1) t.x = 1.0 - t.x;
    if (uScreenOrient.x == 1) t.y = 1.0 - t.y;
    if (uScreenOrient.z == 1) t = vec2(t.y, 1.0 - t.x);
    else if (uScreenOrient.z == 2) t = 1.0 - t;
    else if (uScreenOrient.z == 3) t = vec2(1.0 - t.y, t.x);
    vec2 uv = mix(uVideoCrop.xy, uVideoCrop.zw, t);
    fragColor = vec4(texture(uTexture, uv).rgb, 1.0);
    return;
  }

  vec3 baseColor;
  vec3 N;

  if (uMapping == 3) {
    // Triplanar: project the texture along x, y and z in the parent's space,
    // weighted by how much the surface faces each axis
    vec3 No = normalize(vObjNormal);
    vec3 w = abs(No);
    w /= (w.x + w.y + w.z);

    vec3 p = vObjPos * triScale + 0.5;
    vec2 uvX = p.zy;   // T = +z, B = +y
    vec2 uvY = p.xz;   // T = +x, B = +z
    vec2 uvZ = p.xy;   // T = +x, B = +y

    // Axis-aligned faces have one weight of 1 and two of 0, so only one projector is sampled
    vec3 color = vec3(0.0), slope = vec3(0.0);
    if (w.x > 0.001) project(uvX, vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), w.x, color, slope);
    if (w.y > 0.001) project(uvY, vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), w.y, color, slope);
    if (w.z > 0.001) project(uvZ, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), w.z, color, slope);

    baseColor = color;
    // Tilt the normal against the slope, then bring it from the parent's space to eye space
    N = normalize(mat3(uParentView) * (No - slope));
  } else {
    // Tangent frame (T, B, N) from the mesh
    vec3 Ng = normalize(vNormal);
    vec3 T = normalize(vTangent - dot(vTangent, Ng) * Ng);
    vec3 B = cross(Ng, T);
    mat3 TBN = mat3(T, B, Ng);
    vec3 n;

    if (uMapping == 0) {
      // Bump mapping: tilt the normal against the gradient of the height field
      vec2 s = heightSlope(vTexCoord);
      n = normalize(vec3(-bumpScale * s.x, -bumpScale * s.y, 1.0));
      baseColor = texture(uTexture, vTexCoord).rgb;
    } else {
      // Normal mapping: the texture stores the tangent-space normal in [0, 1]
      n = normalize(texture(uTexture, vTexCoord).rgb * 2.0 - 1.0);
      baseColor = vec3(0.85, 0.65, 0.3);
    }
    N = normalize(TBN * n);
  }

  vec3 Ka = 0.2 * baseColor;
  vec3 Kd = 0.8 * baseColor;
  vec3 Ks = vec3(0.5);
  float shininess = 32.0;

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

let program, uParentView, uLocal, uProjection, uLightPos, uMapping, uVideoCrop, uScreenOrient, uMaterial;

// Texture unit for each material map; the color map shares unit 0 with the video and cylinder
const MAP_UNITS = { color: 0, normal: 1, height: 2 };
const MATERIAL_UNIFORMS = ["uHasNormal", "uHasHeight", "uNormalStrength", "uHeightDepth"];

function updateProgram() {
  try {
    const prog = createProgram(vertEditor.value, fragEditor.value);
    gl.deleteProgram(program);
    program = prog;
    gl.useProgram(program);
    uParentView = gl.getUniformLocation(program, "uParentView");
    uLocal = gl.getUniformLocation(program, "uLocal");
    uProjection = gl.getUniformLocation(program, "uProjection");
    uLightPos = gl.getUniformLocation(program, "uLightPos");
    uMapping = gl.getUniformLocation(program, "uMapping");
    uVideoCrop = gl.getUniformLocation(program, "uVideoCrop");
    uScreenOrient = gl.getUniformLocation(program, "uScreenOrient");
    uMaterial = Object.fromEntries(MATERIAL_UNIFORMS.map(name => [name, gl.getUniformLocation(program, name)]));
    gl.uniform1i(gl.getUniformLocation(program, "uTexture"), 0);
    gl.uniform1i(gl.getUniformLocation(program, "uNormalMap"), MAP_UNITS.normal);
    gl.uniform1i(gl.getUniformLocation(program, "uHeightMap"), MAP_UNITS.height);
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

function scaleMatrix(sx, sy, sz) {
  return new Float32Array([
    sx, 0,  0,  0,
    0,  sy, 0,  0,
    0,  0,  sz, 0,
    0,  0,  0,  1
  ]);
}

const openCubeMesh = createMesh(createOpenCube());
const planeMesh = createMesh(createPlane());
// The sky behind the scene, from clouds.js. If clouds.js didn't load (missing, misnamed, or not in
// index.html), the scene still draws, on black, and the console says why.
const clouds = typeof createClouds === "function"
  ? createClouds(gl, planeMesh)
  : (console.error("clouds.js didn't load: check it's next to index.html and has a <script> line before main.js"), { draw() {} });
const cylinderMesh = createMesh(createCylinder());

// makeSphere (models.js) gives positions, colors, UVs and indices. createMesh also needs normals
// and tangents: on a unit sphere the normal is the position itself, and the tangent points
// around the ring, the direction u grows in.
function sphereMeshData(latRes, lonRes) {
  const { sphPositions, sphUVs, sphIndices } = makeSphere(latRes, lonRes);
  const tangents = new Float32Array(sphPositions.length);
  for (let i = 0; i < sphUVs.length / 2; i++) {
    const lonAngle = sphUVs[2 * i] * 2 * Math.PI;
    tangents.set([-Math.sin(lonAngle), 0, Math.cos(lonAngle)], 3 * i);
  }
  return { positions: sphPositions, normals: sphPositions, tangents, texCoords: sphUVs, indices: sphIndices };
}
const sphereMesh = createMesh(sphereMeshData(64, 64));
const TORUS_MAJOR = 1.2, TORUS_MINOR = 0.38;   // ring radius and tube radius
// createTorus is in models.js. If that file is an older copy without it, the rest of the scene
// still draws and the console says why.
const torusMesh = typeof createTorus === "function"
  ? createMesh(createTorus(64, 32, TORUS_MAJOR, TORUS_MINOR))
  : (console.error("createTorus isn't defined: upload the models.js that has it"), null);

// Upload an image into an existing texture. Set data for maps that hold numbers rather than
// colors (normal and height maps), so the browser doesn't color-correct them. onLoad (which
// receives the image) and onError are optional callbacks.
function setTextureImage(texture, url, data = false, onLoad = () => {}, onError = () => {}) {
  const image = new Image();
  image.onerror = onError;
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);  // images start at the top row, texture coordinates at the bottom
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, data ? gl.NONE : gl.BROWSER_DEFAULT_WEBGL);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    if (url.startsWith("blob:")) URL.revokeObjectURL(url);
    onLoad(image);
  };
  image.src = url;
}

// A 1x1 texture of one color, shown until an image loads and used for missing maps
function solidTexture(rgba) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
  return texture;
}

function loadTexture(url, { data = false, placeholder = [128, 128, 255, 255] } = {}) {
  const texture = solidTexture(placeholder);
  setTextureImage(texture, url, data);
  return texture;
}

const normalTexture = loadTexture("texture2.jpg", { data: true });

// A material is a set of maps: color, normal and height. Any of them can be missing;
// DEFAULT_MAPS stand in for those.
const DEFAULT_MAPS = {
  color: solidTexture([255, 255, 255, 255]),
  normal: solidTexture([128, 128, 255, 255]),
  height: solidTexture([0, 0, 0, 255])
};
const LUMINANCE_DEPTH = 0.012;   // bump depth when it comes from the color map's darkness
const HEIGHT_MAP_DEPTH = 0.15;   // bump depth for a real height map's full range

function createMaterial(maps = {}) {
  return {
    maps,                      // textures by name: color, normal, height
    normalStrength: 1.0,       // these two can be changed at any time
    heightDepth: LUMINANCE_DEPTH
  };
}

// The brick image, with bumps from its darkness; last in the Material button's list
const brickMaterial = createMaterial({ color: loadTexture("brick.png", { placeholder: [128, 128, 128, 255] }) });

// Load a texture that might not exist. Resolves to the texture, or to null if the file is missing.
function loadOptionalTexture(url, { data = false, placeholder = [128, 128, 128, 255] } = {}) {
  return new Promise(resolve => {
    const texture = solidTexture(placeholder);
    setTextureImage(texture, url, data, () => resolve(texture), () => {
      gl.deleteTexture(texture);
      console.warn("Couldn't load", url);
      resolve(null);
    });
  });
}

// The materials the Material button goes through, in order. Each folder sits next to index.html
// with its files named as they came from the texture site; only the color, OpenGL normal and
// height files are used. To add one, copy an entry and change the folder and file names
// (leave folder out for files that sit next to index.html).
const MATERIAL_PRESETS = [
  {
    name: "plastic",
    folder: "plastic_0010_1k_5m9Axh",
    color: "plastic_0010_color_1k.jpg",
    normal: "plastic_0010_normal_opengl_1k.png",
    height: "plastic_0010_height_1k.png"
  },
  {
    name: "orange",
    folder: "food_0022_1k_Z1rbdX",
    color: "food_0022_color_1k.jpg",
    normal: "food_0022_normal_opengl_1k.png",
    height: "food_0022_height_1k.png"
  },
  {
    name: "knitted",
    folder: "fabric_0019_1k_Z7MxvD (1)",
    color: "fabric_0019_color_1k.jpg",
    normal: "fabric_0019_normal_opengl_1k.png",
    height: "fabric_0019_height_1k.png"
  },
  {
    name: "fur",
    folder: "fabrics_0059_1k_SkuReA",
    color: "fabrics_0059_color_1k.jpg",
    normal: "fabrics_0059_normal_opengl_1k.png",
    height: "fabrics_0059_height_1k.png"
  }
];

function loadPresetMaterial(preset) {
  const m = createMaterial();
  // encodeURIComponent turns spaces and brackets in a name into a valid URL
  const url = file => preset.folder
    ? `${encodeURIComponent(preset.folder)}/${encodeURIComponent(file)}`
    : encodeURIComponent(file);
  const maps = [
    ["color", { placeholder: [128, 128, 128, 255] }],
    ["normal", { data: true, placeholder: [128, 128, 255, 255] }],
    ["height", { data: true, placeholder: [0, 0, 0, 255] }]
  ];
  for (const [name, options] of maps) {
    if (!preset[name]) continue;   // the entry leaves this map out
    loadOptionalTexture(url(preset[name]), options).then(texture => {
      if (!texture) return;
      m.maps[name] = texture;
      if (name === "height") m.heightDepth = HEIGHT_MAP_DEPTH;
    });
  }
  return m;
}

// All the folders load at the start, so switching is instant
const presetMaterials = MATERIAL_PRESETS.map(preset => ({ name: preset.name, material: loadPresetMaterial(preset) }));
presetMaterials.push({ name: "brick", material: brickMaterial });
const orangeMaterial = presetMaterials.find(p => p.name === "orange").material;
const furMaterial = presetMaterials.find(p => p.name === "fur").material;

// Noise texture: TV snow rendered on the GPU into a 512x512 texture, new every frame
// Render to texture: https://webglfundamentals.org/webgl/lessons/webgl-render-to-texture.html
// Data textures:     https://webglfundamentals.org/webgl/lessons/webgl-data-textures.html
// Hash: pcg3d from Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering",
//       Journal of Computer Graphics Techniques 9(3), 2020, http://jcgt.org/published/0009/03/02/
const NOISE_SIZE = 512;

const NOISE_VERTEX_SHADER = `#version 300 es
in vec3 aPosition;
void main() {
  // The plane spans -0.5..0.5; doubled, it covers the whole texture (-1..1 in clip space)
  gl_Position = vec4(aPosition.xy * 2.0, 0.0, 1.0);
}
`;

const NOISE_FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;    // 32-bit integers for the hash
uniform uint uSeed;     // new every frame, so the snow moves
uniform bool uColor;    // false: gray snow (one value in r, g and b); true: each channel random
out vec4 fragColor;

// Three numbers in, three unrelated random numbers out
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}

void main() {
  // Hash this pixel's position with the seed, scaled to 0..1
  vec3 r = vec3(pcg3d(uvec3(uvec2(gl_FragCoord.xy), uSeed))) / 4294967295.0;
  fragColor = vec4(uColor ? r : vec3(r.x), 1.0);
}
`;

// The texture to render into, allocated with no data (null)
const noiseTexture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, NOISE_SIZE, NOISE_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);   // hard-edged specks, no mipmaps
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
// Wrapping stays REPEAT (the default), so the snow tiles across a TV body

// A framebuffer with the texture as its only attachment. No depth buffer: the noise pass
// draws one flat plane, so there's nothing to sort.
const noiseFramebuffer = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, noiseFramebuffer);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, noiseTexture, 0);
if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
  console.error("The noise framebuffer can't be rendered to on this system");
}
gl.bindFramebuffer(gl.FRAMEBUFFER, null);

const noiseProgram = createProgram(NOISE_VERTEX_SHADER, NOISE_FRAGMENT_SHADER);
const uNoiseSeed = gl.getUniformLocation(noiseProgram, "uSeed");
const uNoiseColor = gl.getUniformLocation(noiseProgram, "uColor");
let noiseFrame = 0;      // the seed: counts frames, so every frame gets new snow
let colorSnow = false;   // gray or colored; the channel button picks it at random each time ch 2 comes on

// Render a new frame of snow into noiseTexture, then point drawing back at the canvas.
// The viewport has to match whatever is being drawn into, each time it changes.
function renderNoise() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, noiseFramebuffer);
  gl.viewport(0, 0, NOISE_SIZE, NOISE_SIZE);
  gl.useProgram(noiseProgram);
  gl.uniform1ui(uNoiseSeed, noiseFrame++);
  gl.uniform1i(uNoiseColor, colorSnow ? 1 : 0);
  gl.bindVertexArray(planeMesh.vao);
  gl.drawElements(gl.TRIANGLES, planeMesh.count, gl.UNSIGNED_SHORT, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  gl.useProgram(program);   // back to the scene program
}


function setMaterial(m) {
  for (const tv of tvs) tv.material = m;
}

// Material button: <button id="materialButton"></button> in index.html.
// The TVs start on the first material (plastic); each click moves to the next, wrapping around.
let presetIndex = 0;
const materialButton = document.getElementById("materialButton");
if (materialButton) materialButton.textContent = `Material: ${presetMaterials[0].name}`;
materialButton?.addEventListener("click", () => {
  presetIndex = (presetIndex + 1) % presetMaterials.length;
  setMaterial(presetMaterials[presetIndex].material);
  materialButton.textContent = `Material: ${presetMaterials[presetIndex].name}`;
});

// Which map an image is, from its file name. Texture libraries put the map type in the name,
// e.g. Plastic010_1K-PNG_NormalGL.png (ambientCG) or orange_peel_nor_gl_1k.png (Poly Haven).
// The name is split into words and read from the end, so the material's own name doesn't count.
function mapKind(fileName) {
  const words = fileName.toLowerCase().replace(/\.[^.]*$/, "").split(/[^a-z0-9]+/);
  for (let i = words.length - 1; i >= 0; i--) {
    const w = words[i], next = words[i + 1];
    // DirectX normal maps (green pointing down) are skipped; WebGL uses the OpenGL kind
    if (["normaldx", "nordx", "nrmdx"].includes(w)) return "other";
    if (["normal", "normals", "normalgl", "nor", "norgl", "nrm", "norm"].includes(w)) {
      return ["dx", "directx", "direct"].includes(next) ? "other" : "normal";
    }
    if (["height", "heightmap", "disp", "displacement", "bump"].includes(w)) return "height";
    if (["color", "colour", "col", "albedo", "diff", "diffuse", "basecolor"].includes(w)) return "color";
    // Other maps in the same folders: recognized so they're skipped, not used
    if (["rough", "roughness", "gloss", "glossiness", "ao", "ambientocclusion", "occlusion", "metal",
         "metalness", "metallic", "specular", "opacity", "emission", "emissive", "arm"].includes(w)) return "other";
  }
  return null;
}

// Replace a material's maps with images from a list of files: a whole folder, several picked
// files, or a single image (an unlabeled single image is used as the color map).
// Returns false if none of the files is a map it recognizes.
function loadMaterialFiles(material, fileList) {
  const images = [...fileList].filter(f => /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
  const found = {};
  for (const file of images) {
    const kind = mapKind(file.name);
    if (kind && !found[kind]) found[kind] = file;
  }
  if (images.length === 1 && !mapKind(images[0].name)) found.color = images[0];

  const normalFile = found.normal;
  if (!found.color && !normalFile && !found.height) {
    console.warn("No texture maps recognized in:", images.map(f => f.name));
    return false;
  }

  for (const texture of Object.values(material.maps)) gl.deleteTexture(texture);
  const load = (file, options) => loadTexture(URL.createObjectURL(file), options);
  const maps = {};
  if (found.color) maps.color = load(found.color, { placeholder: [128, 128, 128, 255] });
  if (normalFile) maps.normal = load(normalFile, { data: true });
  if (found.height) maps.height = load(found.height, { data: true, placeholder: [0, 0, 0, 255] });

  material.maps = maps;
  material.heightDepth = found.height ? HEIGHT_MAP_DEPTH : LUMINANCE_DEPTH;
  console.log("Material maps:", { color: found.color?.name, normal: normalFile?.name, height: found.height?.name });
  return true;
}

// Material pickers in index.html, for trying maps from your computer:
//   <input type="file" id="materialInput" webkitdirectory>             a folder of maps
//   <input type="file" id="brickInput" accept="image/*" multiple>      one image, or several maps
const customMaterial = createMaterial();
for (const id of ["materialInput", "brickInput"]) {
  document.getElementById(id)?.addEventListener("change", e => {
    if (!loadMaterialFiles(customMaterial, e.target.files)) return;
    setMaterial(customMaterial);
    if (materialButton) materialButton.textContent = "Material: custom";
  });
}

// Bind a material's maps to their texture units and set its uniforms
function bindMaterial(m) {
  for (const [name, unit] of Object.entries(MAP_UNITS)) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, m.maps[name] || DEFAULT_MAPS[name]);
  }
  gl.activeTexture(gl.TEXTURE0);
  gl.uniform1i(uMaterial.uHasNormal, m.maps.normal ? 1 : 0);
  gl.uniform1i(uMaterial.uHasHeight, m.maps.height ? 1 : 0);
  gl.uniform1f(uMaterial.uNormalStrength, m.normalStrength);
  gl.uniform1f(uMaterial.uHeightDepth, m.heightDepth);
}

// url can be a file next to index.html or a direct link to a video file on another site.
// Another site has to allow cross-origin use (CORS), or WebGL isn't allowed to read its pixels.
function loadVideoTexture(url) {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";   // ask for CORS permission; must be set before src
  video.addEventListener("error", () =>
    console.error(`Couldn't load video ${url}: check the link, and that the site allows cross-origin use (CORS)`));
  video.src = url;
  video.muted = true;        // browsers only autoplay muted video
  video.loop = true;
  video.playsInline = true;
  video.play().catch(() => document.addEventListener("click", () => video.play(), { once: true }));

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  // No mipmaps: they would have to be rebuilt every frame
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { texture, video };
}

// Copy the video's current frame into its texture; called once per render
function updateVideoTexture(v) {
  if (v.blocked || v.video.readyState < v.video.HAVE_CURRENT_DATA) return;
  gl.bindTexture(gl.TEXTURE_2D, v.texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
  try {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, v.video);
  } catch (e) {
    // WebGL refuses video from another site that doesn't allow it. Stop trying, so the
    // error doesn't end the render loop; the screens stay black and the rest still draws.
    v.blocked = true;
    console.error("Can't use this video as a texture (the site doesn't allow cross-origin use):", e.message);
  }
}

const tvVideo = loadVideoTexture("video2_2.mp4");   // shared by every TV

// Test pattern images for channel 3, in the tc folder next to index.html. Each time the channel
// button reaches ch 3, one of them is picked at random.
const TEST_PATTERN_FOLDER = "tc";
const TEST_PATTERNS = ["FuBK_STANDARD.png", "PM5544_with_non-PAL_signals.png", "Testcard_F.jpg"];

const blackTexture = solidTexture([0, 0, 0, 255]);   // shown on ch 3 if no test pattern loaded
const testPatterns = [];                              // the ones that loaded: { texture, aspect }
for (const file of TEST_PATTERNS) {
  const texture = solidTexture([0, 0, 0, 255]);
  setTextureImage(texture, `${TEST_PATTERN_FOLDER}/${encodeURIComponent(file)}`, false,
    image => testPatterns.push({ texture, aspect: image.naturalWidth / image.naturalHeight }),
    () => { gl.deleteTexture(texture); console.warn("Couldn't load test pattern", file); });
}

// Channels: ch 1 the video, ch 2 snow, ch 3 a test pattern, ch 4 a mix of all three
const SOURCES = ["video", "snow", "test"];   // what ch 1, 2 and 3 show
let channel = 0;            // 0 to 3 for ch 1 to 4
let testPattern = null;     // the test pattern picked for ch 3

// A random test pattern, a different one from `previous` when there's more than one
function pickTestPattern(previous) {
  if (!testPatterns.length) return null;
  const others = testPatterns.filter(p => p !== previous);
  const choices = others.length ? others : testPatterns;
  return choices[Math.floor(Math.random() * choices.length)];
}

// The picture for one source: its texture and its aspect (width / height; null while the
// video's size isn't known yet). pattern is the test pattern to use for "test".
function picture(source, pattern) {
  if (source === "video") {
    const v = tvVideo.video;
    return { texture: tvVideo.texture, aspect: v.videoWidth ? v.videoWidth / v.videoHeight : null };
  }
  if (source === "snow") return { texture: noiseTexture, aspect: 1 };
  return pattern || { texture: blackTexture, aspect: 1 };
}

// Ch 4, the mix: every screen gets its own random source, its own test pattern if it's
// showing one, and its own choice of spread (its part of the picture across all the screens)
// or repeat (the whole picture). Picked again each time ch 4 comes on.
function mixScreens() {
  for (const tv of tvs) {
    const source = SOURCES[Math.floor(Math.random() * SOURCES.length)];
    tv.mix = { source, pattern: pickTestPattern(null), spread: Math.random() < 0.5 };
  }
}

// Channel button: <button id="channelButton">Ch 1</button> in index.html
const channelButton = document.getElementById("channelButton");
channelButton?.addEventListener("click", () => {
  channel = (channel + 1) % 4;
  if (channel === 1 || channel === 3) colorSnow = Math.random() < 0.5;   // gray or colored snow, picked each time
  if (channel === 2) testPattern = pickTestPattern(testPattern);
  if (channel === 3) mixScreens();
  channelButton.textContent = `Ch ${channel + 1}`;
  updateSound();
});

// Screens button: <button id="screensButton">Screens: spread</button> in index.html.
// Goes through these, for ch 1 to 3 (on ch 4 every screen makes its own choice):
//   spread:         one picture across all the screens
//   repeat:         the whole picture on every screen
//   mix:            some screens spread, some repeat, picked at random
//   spread shuffle: spread, but the pieces of the picture swapped between screens at random
const SCREEN_MODES = ["spread", "repeat", "mix", "spread shuffle"];
let screenMode = 0;

// A random order of 0..n-1 in which nothing stays in its own place (when n is at least 2)
function derangement(n) {
  const order = [...Array(n).keys()];
  if (n < 2) return order;
  do {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  } while (order.some((k, i) => k === i));
  return order;
}

// Pick the random parts of mix and spread shuffle; done each time one of them comes on
function arrangeScreens() {
  for (const tv of tvs) tv.screenSpread = Math.random() < 0.5;

  // Pieces are swapped only between screens of the same shape and size (the 16:9s among
  // themselves, the squares among themselves). Each piece then keeps exactly its place in the
  // picture, corners and all, and just moves to another monitor: no stretching, no overlaps,
  // no repeats.
  const groups = new Map();
  for (const tv of tvs) {
    const key = tv.screenHalf.map(h => h.toFixed(4)).join();
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(tv);
  }
  for (const group of groups.values()) {
    const order = derangement(group.length);
    group.forEach((tv, i) => tv.pieceOf = group[order[i]]);
  }
}

const screensButton = document.getElementById("screensButton");
screensButton?.addEventListener("click", () => {
  screenMode = (screenMode + 1) % SCREEN_MODES.length;
  if (SCREEN_MODES[screenMode] === "mix" || SCREEN_MODES[screenMode] === "spread shuffle") arrangeScreens();
  screensButton.textContent = `Screens: ${SCREEN_MODES[screenMode]}`;
});

// Orientation shuffles for the pictures on the screens, each switched on or off by itself:
//   x: some screens upside down (flipped around the x axis)
//   y: some screens mirrored left to right (flipped around the y axis)
//   z: every screen turned a random number of quarter turns (around the z axis)
// Which screens get what is picked at random each time a shuffle is switched on.
// In index.html: three toggle buttons, shuffleXButton, shuffleYButton and shuffleZButton.
const shuffleOn = { x: false, y: false, z: false };

function pickOrientations(axis) {
  for (const tv of tvs) {
    if (axis === "x") tv.flipX = Math.random() < 0.5;
    if (axis === "y") tv.flipY = Math.random() < 0.5;
    if (axis === "z") tv.quarterTurns = Math.floor(Math.random() * 4);
  }
}

for (const axis of ["x", "y", "z"]) {
  const button = document.getElementById(`shuffle${axis.toUpperCase()}Button`);
  const show = () => button.textContent = `Shuffle ${axis.toUpperCase()}: ${shuffleOn[axis] ? "on" : "off"}`;
  if (!button) continue;
  show();
  button.addEventListener("click", () => {
    shuffleOn[axis] = !shuffleOn[axis];
    if (shuffleOn[axis]) pickOrientations(axis);
    show();
  });
}

// Channel sound with the browser's Web Audio API: ch 1 plays the video's own sound, ch 2 white
// noise, ch 3 a 1000 Hz test tone. The tone and noise run all the time; their volumes are
// turned up or down. The audio is created on the first click, because browsers only start
// sound after the person has done something on the page.
const TONE_VOLUME = 0.1;     // 1 kHz at -20 dBFS (amplitude 0.1), the usual test-tone level
const NOISE_VOLUME = 0.08;
let soundOn = false;
let audio = null;

function setupAudio() {
  if (audio) return audio;
  const ctx = new AudioContext();

  const tone = new OscillatorNode(ctx, { type: "sine", frequency: 1000 });
  const toneGain = new GainNode(ctx, { gain: 0 });
  tone.connect(toneGain).connect(ctx.destination);
  tone.start();

  // White noise: two seconds of random samples between -1 and 1, played on a loop
  const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const samples = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;
  const noise = new AudioBufferSourceNode(ctx, { buffer, loop: true });
  const noiseGain = new GainNode(ctx, { gain: 0 });
  noise.connect(noiseGain).connect(ctx.destination);
  noise.start();

  audio = { ctx, toneGain, noiseGain };
  return audio;
}

// Play the sound that goes with the channel (on ch 4, all three together), or nothing when the
// sound is off
function updateSound() {
  const showing = new Set(channel === 3 ? SOURCES : [SOURCES[channel]]);
  const video = tvVideo.video;
  video.muted = !(soundOn && showing.has("video"));
  if (!video.muted) video.play();   // in case autoplay was blocked
  if (!soundOn && !audio) return;   // nothing to turn down yet

  const { ctx, toneGain, noiseGain } = setupAudio();
  if (ctx.state === "suspended") ctx.resume();
  // setTargetAtTime fades over about 10 ms instead of jumping, which would click
  noiseGain.gain.setTargetAtTime(soundOn && showing.has("snow") ? NOISE_VOLUME : 0, ctx.currentTime, 0.01);
  toneGain.gain.setTargetAtTime(soundOn && showing.has("test") ? TONE_VOLUME : 0, ctx.currentTime, 0.01);
}

// Sound toggle: <button id="soundButton">Sound on</button> in index.html
const soundButton = document.getElementById("soundButton");
soundButton?.addEventListener("click", () => {
  soundOn = !soundOn;
  soundButton.textContent = soundOn ? "Sound off" : "Sound on";
  updateSound();
});

const BUMP_MAPPING = 0, NORMAL_MAPPING = 1, VIDEO = 2, TRIPLANAR = 3;
const IDENTITY = translate(0, 0, 0);

// Draw one mesh. uParentView (the space it sits in) and the texture are set by the caller.
function drawMesh(mesh, mapping, local) {
  gl.uniformMatrix4fv(uLocal, false, local);
  gl.uniform1i(uMapping, mapping);
  gl.bindVertexArray(mesh.vao);
  gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
}

// A TV: an open brick box with a border around the opening, inner walls running back
// from the border, and a curved video screen just behind the opening.
//
// size is the box's half-height and aspect is the screen opening's width / height;
// the box is as wide as the opening plus the border on each side. border and depth
// are relative to size. The TV's own space is translate(position) * rotateY(turn), and every
// part is built in real units inside it, so triplanar bricks stay the same size on all TVs.
class TV {
  constructor({ position = [0, 0, 0], size = 1, aspect = 1, border = 0.2, depth = 0.5 } = {}) {
    this.position = position;
    this.size = size;
    this.aspect = aspect;
    this.border = border;            // width of the border strips, relative to size
    this.depth = depth;              // how far the inner walls run back, relative to size
    this.material = presetMaterials[0].material;   // body material (starts on plastic); reassign at any time
    this.turn = 0;                   // turn about its vertical axis, in radians (set by stackTVs)
    this.flipX = false; this.flipY = false; this.quarterTurns = 0;   // picked by the orientation shuffles
    this.orient = [0, 0, 0];          // the orientation the screen uses this frame
    this.videoCrop = [0, 0, 1, 1];   // part of the picture shown on the screen: u0, v0, u1, v1 (set each frame in render)
    this.build();
  }

  // Work out every part; call again after changing size, aspect, border or depth
  build() {
    const b = this.border * this.size;     // border width
    const iy = this.size - b;              // opening half-height
    const ix = iy * this.aspect;           // opening half-width
    const hx = ix + b, hy = this.size, hz = this.size;   // box half-extents
    this.half = [hx, hy, hz];

    // Screen: a little larger than the opening so its edges tuck behind the inner walls.
    // createTvScreen takes arc lengths; asin turns the half-widths we want into them.
    const margin = 0.05 * this.size;
    const sx = ix + margin, sy = iy + margin;     // screen half-extents (straight across)
    const R = 1.25 * Math.max(hx, hy);            // curvature radius grows with the TV; smaller curves more
    this.screenHalf = [sx, sy];
    this.screenMesh = createMesh(createTvScreen(30, 30, 2 * R * Math.asin(sx / R), 2 * R * Math.asin(sy / R), R));

    // How far the screen curves back from its center at a point (x, y) on it
    const sagAt = (x, y) => R * (1 - Math.cos(Math.asin(x / R)) * Math.cos(Math.asin(y / R)));

    // Bring the screen forward, so its center bulges out past the front of the frame. It can only
    // come forward until the edge of the opening (where it curves back least: the middle of the
    // top and bottom edges) is still just behind the frame, or it would poke through the border.
    const edgeSag = Math.min(sagAt(0, iy), sagAt(ix, 0));
    const forward = Math.min(0.15 * this.size, edgeSag - 0.02 * this.size);
    this.screenLocal = translate(0, 0, hz + forward);   // the screen's center (local z = 0) is its front-most point

    // How far behind the front the screen's corners end up; the inner walls have to reach at
    // least that far (plus a little), or you'd see past their ends at the corners
    const cornerDepth = sagAt(sx, sy) - forward;
    const d = Math.max(this.depth * this.size, cornerDepth + 0.05 * this.size);   // inner wall depth

    // Body: the -1..1 open cube scaled to the box
    this.parts = [{ mesh: openCubeMesh, local: scaleMatrix(hx, hy, hz) }];

    for (let k = 0; k < 4; k++) {
      const side = rotateZ(k * Math.PI / 2);        // k = 0 top, 1 left, 2 bottom, 3 right
      const horizontal = k % 2 === 0;               // top and bottom run along x
      const along = horizontal ? hx : hy;           // half-length of the box edge this side runs along
      const reach = horizontal ? hy : hx;           // distance from the center to that edge
      const innerAlong = horizontal ? ix : iy;      // same two, for the opening
      const innerReach = horizontal ? iy : ix;

      // Border strip: scale to (2 * along - b) x b, move to the edge at the front (z = hz).
      // Shifted by -b/2 along the strip so the four strips tile the corners without overlapping.
      this.parts.push({ mesh: planeMesh, local: multiply(side,
        multiply(translate(-b / 2, reach - b / 2, hz),
          scaleMatrix(2 * along - b, b, 1))) });

      // Inner wall: scale to (2 * innerAlong) x d, rotate 90 degrees about x so it stands
      // perpendicular to the border facing the center, then move it to the opening's edge,
      // running from z = hz back to z = hz - d.
      this.parts.push({ mesh: planeMesh, local: multiply(side,
        multiply(translate(0, innerReach, hz - d / 2),
          multiply(rotateX(Math.PI / 2), scaleMatrix(2 * innerAlong, d, 1)))) });
    }
    this.wallDepth = d;
    this.screenForward = forward;
  }

  draw(view, screenTexture) {
    const space = multiply(translate(...this.position), rotateY(this.turn));
    gl.uniformMatrix4fv(uParentView, false, multiply(view, space));

    bindMaterial(this.material);
    for (const { mesh, local } of this.parts) drawMesh(mesh, TRIPLANAR, local);

    gl.bindTexture(gl.TEXTURE_2D, screenTexture);
    gl.uniform4fv(uVideoCrop, this.videoCrop);
    gl.uniform3i(uScreenOrient, ...this.orient);
    drawMesh(this.screenMesh, VIDEO, this.screenLocal);
  }
}

// The part of a picture (as u0, v0, u1, v1) that covers a target of the given aspect without
// stretching: the picture is scaled to fill it, and whatever sticks out is cropped evenly.
function coverCrop(targetAspect, pictureAspect) {
  if (!pictureAspect) return [0, 0, 1, 1];
  const uSpan = Math.min(1, targetAspect / pictureAspect);
  const vSpan = Math.min(1, pictureAspect / targetAspect);
  const u0 = (1 - uSpan) / 2, v0 = (1 - vSpan) / 2;
  return [u0, v0, u0 + uSpan, v0 + vSpan];
}

// The box around all the screens, from the TVs' positions (x0, y0, x1, y1)
function screenWall(tvs) {
  const rects = tvs.map(({ position: [x, y], screenHalf: [sx, sy] }) => [x - sx, y - sy, x + sx, y + sy]);
  return [Math.min(...rects.map(r => r[0])), Math.min(...rects.map(r => r[1])),
          Math.max(...rects.map(r => r[2])), Math.max(...rects.map(r => r[3]))];
}

// Which part of a picture one screen shows.
//   spread: one picture covers the whole wall, like a video wall, and this screen shows the
//           part that falls on it
//   repeat: this screen shows the whole picture, cropped to the screen's shape
//   pieceOf: the screen whose place in the wall this one shows (itself, unless shuffled). The
//            piece keeps this screen's own size and shape, so nothing stretches, and it's
//            moved in if needed so it stays inside the wall.
//   turned:  the picture is turned a quarter (or three quarters) on this screen, so the piece
//            is taken the other way round (tall instead of wide), so nothing stretches
function screenCrop(tv, wall, pictureAspect, spread, pieceOf = tv, turned = false) {
  const [sx, sy] = turned ? [tv.screenHalf[1], tv.screenHalf[0]] : tv.screenHalf;
  if (!spread) return coverCrop(sx / sy, pictureAspect);

  const [x0, y0, x1, y1] = wall;
  const [u0, v0, u1, v1] = coverCrop((x1 - x0) / (y1 - y0), pictureAspect);
  const clamp = (value, lo, hi) => Math.min(Math.max(value, lo), hi);
  const x = clamp(pieceOf.position[0], x0 + sx, x1 - sx);
  const y = clamp(pieceOf.position[1], y0 + sy, y1 - sy);
  const u = px => u0 + (px - x0) / (x1 - x0) * (u1 - u0);
  const v = py => v0 + (py - y0) / (y1 - y0) * (v1 - v0);
  return [u(x - sx), v(y - sy), u(x + sx), v(y + sy)];
}

// Randomness in the stack, picked again on each page load
const STACK_RANDOM = {
  rowShift: 0.4,   // each row moves left or right by up to this much, so rows can overhang
  extraGap: 0.2,   // up to this much extra space between neighbors in a row
  turn: 0.09,      // each TV turns about its vertical axis by up to this much (radians, about 5°)
  depth: 0.3      // each TV moves forward or back by up to this much
};
const randomBetween = limit => (Math.random() * 2 - 1) * limit;   // from -limit to limit

// Stack TVs in rows from the bottom up. Each row lists its TVs' aspects; their left-to-right order
// is shuffled. Rows sit directly on top of each other, and the whole stack is centered on the
// origin. sideGap is the smallest space between neighbors in a row.
function stackTVs(rows, sideGap = 0.05) {
  const built = rows.map(aspects => {
    const row = aspects.map(aspect => new TV({ aspect }));
    for (let i = row.length - 1; i > 0; i--) {   // random order: wide one on the left or the right
      const j = Math.floor(Math.random() * (i + 1));
      [row[i], row[j]] = [row[j], row[i]];
    }
    for (const tv of row) tv.turn = randomBetween(STACK_RANDOM.turn);
    return row;
  });
  const rowHeights = built.map(row => Math.max(...row.map(tv => 2 * tv.half[1])));
  const stackHeight = rowHeights.reduce((a, b) => a + b, 0);

  let y = -stackHeight / 2;                 // bottom of the current row
  built.forEach((row, r) => {
    // Width each TV takes up once turned (its box seen from the front), so neighbors never touch
    const widths = row.map(tv => 2 * (tv.half[0] * Math.cos(tv.turn) + tv.half[2] * Math.abs(Math.sin(tv.turn))));
    const gaps = row.slice(1).map(() => sideGap + Math.random() * STACK_RANDOM.extraGap);
    const rowWidth = widths.reduce((a, b) => a + b, 0) + gaps.reduce((a, b) => a + b, 0);
    let x = -rowWidth / 2 + randomBetween(STACK_RANDOM.rowShift);   // left edge of the next TV
    row.forEach((tv, i) => {
      tv.position = [x + widths[i] / 2, y + tv.half[1], randomBetween(STACK_RANDOM.depth)];
      x += widths[i] + (gaps[i] || 0);
    });
    y += rowHeights[r];
  });
  return built.flat();
}

const tvs = stackTVs([
  [16 / 9, 4 / 3],   // bottom
  [16 / 9, 4 / 3],
  [16 / 9, 4 / 3],
  [16 / 9]           // top
]);

// The imaginary ground plane: the height the bottom row of TVs stands on
const GROUND_Y = Math.min(...tvs.map(tv => tv.position[1] - tv.half[1]));

// Where the other objects sit, in the fixed camera's world (the camera looks at the origin
// from CAMERA_DISTANCE in front)
const SPHERE_CENTER = [-7, 4, -14], SPHERE_RADIUS = 9;   // upper left, further back
const CYLINDER_CENTER = [3.2, -1, 6];                     // right, close to the camera
const TORUS_CENTER = [-4.5, GROUND_Y + TORUS_MINOR, 2.2]; // lying on the ground in front of the screens, well to the left

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0, 0, 0, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// Interactive controls. The camera stays put; the controls move the objects:
//   mouse drag and arrow keys / w, s: the TV stack only
//   slider, axis dropdown and Start/Stop: the ball, the cylinder and the torus only
const CAMERA_DISTANCE = 13;             // the fixed camera sits this far in front of the origin
let dragRotX = 0, dragRotY = -0.5;      // mouse drag: turns the TV stack about its center
let camX = 0, camY = 0, camZ = -13;     // arrow keys, w, s: move the TV stack (starts in its usual place)
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

  renderNoise();   // first, into the noise texture; then everything below draws to the canvas
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  clouds.draw(t);           // the sky, behind everything
  gl.useProgram(program);   // back to the scene program

  // Fixed camera; the light stays fixed in the world
  const view = translate(0, 0, -CAMERA_DISTANCE);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  // TVs: the mouse drag turns the stack about its center, the keys move it
  const tvView = multiply(translate(camX, camY, camZ), multiply(rotateY(dragRotY), rotateX(dragRotX)));
  // Ball, cylinder and torus: the slider, axis and Start/Stop turn each one about its own center
  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);

  gl.activeTexture(gl.TEXTURE0);
  updateVideoTexture(tvVideo);

  // Each screen's picture and crop; redone each frame, so it follows the video size and TV positions.
  // On ch 4 every screen has its own source and spread/repeat; otherwise they all share them.
  const wall = screenWall(tvs);
  for (const tv of tvs) {
    const mixed = channel === 3;
    const pic = mixed ? picture(tv.mix.source, tv.mix.pattern) : picture(SOURCES[channel], testPattern);
    const mode = SCREEN_MODES[screenMode];
    const spread = mixed ? tv.mix.spread
      : mode === "spread" || mode === "spread shuffle" || (mode === "mix" && tv.screenSpread);
    // Orientation shuffles that are switched on: [upside down, mirrored, quarter turns]
    tv.orient = [shuffleOn.x && tv.flipX ? 1 : 0, shuffleOn.y && tv.flipY ? 1 : 0, shuffleOn.z ? tv.quarterTurns : 0];
    tv.videoCrop = screenCrop(tv, wall, pic.aspect, spread, !mixed && mode === "spread shuffle" ? tv.pieceOf : tv,
                              tv.orient[2] % 2 === 1);
    tv.draw(tvView, pic.texture);
  }

  // Big orange sphere. It uses the same triplanar mapping as the TVs, with the radius in its
  // local matrix, so the peel's dimples come out the same size as on a TV.
  gl.uniformMatrix4fv(uParentView, false, multiply(view, multiply(translate(...SPHERE_CENTER), axisRotation)));
  bindMaterial(orangeMaterial);
  drawMesh(sphereMesh, TRIPLANAR, scaleMatrix(SPHERE_RADIUS, SPHERE_RADIUS, SPHERE_RADIUS));

  // Cylinder
  gl.uniformMatrix4fv(uParentView, false, multiply(view, multiply(translate(...CYLINDER_CENTER), axisRotation)));
  gl.bindTexture(gl.TEXTURE_2D, normalTexture);
  drawMesh(cylinderMesh, NORMAL_MAPPING, IDENTITY);

  // Torus in the fur, built at its real size, so the triplanar texture is the same scale as on the TVs.
  // Like the ball and cylinder, the slider, axis and Start/Stop turn it about its own center.
  gl.uniformMatrix4fv(uParentView, false, multiply(view, multiply(translate(...TORUS_CENTER), axisRotation)));
  bindMaterial(furMaterial);
  if (torusMesh) drawMesh(torusMesh, TRIPLANAR, IDENTITY);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
