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
uniform int uMapping;  // 0 = bump mapping, 1 = normal mapping, 2 = video (unlit), 3 = triplanar bump mapping
uniform vec4 uVideoCrop;  // part of the video frame shown on a screen: (u0, v0, u1, v1)

out vec4 fragColor;

const float bumpScale = 3.0;
const float triScale = 0.5;   // texture copies per unit; 0.5 = one copy per 2-unit cube face

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

void main() {
  // Video: the screen gives off its own light, so show the frame as is with no shading
  if (uMapping == 2) {
    vec2 uv = mix(uVideoCrop.xy, uVideoCrop.zw, vTexCoord);
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

    // Blend the colors, and the height slopes as 3D vectors along each projector's T and B.
    // Axis-aligned faces have one weight of 1 and two of 0, so only one projector is sampled.
    vec3 color = vec3(0.0);
    vec3 g = vec3(0.0);
    if (w.x > 0.001) {
      vec2 s = heightSlope(uvX);
      color += w.x * texture(uTexture, uvX).rgb;
      g += w.x * (s.x * vec3(0.0, 0.0, 1.0) + s.y * vec3(0.0, 1.0, 0.0));
    }
    if (w.y > 0.001) {
      vec2 s = heightSlope(uvY);
      color += w.y * texture(uTexture, uvY).rgb;
      g += w.y * (s.x * vec3(1.0, 0.0, 0.0) + s.y * vec3(0.0, 0.0, 1.0));
    }
    if (w.z > 0.001) {
      vec2 s = heightSlope(uvZ);
      color += w.z * texture(uTexture, uvZ).rgb;
      g += w.z * (s.x * vec3(1.0, 0.0, 0.0) + s.y * vec3(0.0, 1.0, 0.0));
    }

    baseColor = color;
    // Tilt the normal against the slope, then bring it from the parent's space to eye space
    N = normalize(mat3(uParentView) * (No - bumpScale * g));
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

let program, uParentView, uLocal, uProjection, uLightPos, uMapping, uVideoCrop;

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
const cylinderMesh = createMesh(createCylinder());

// Upload an image into an existing texture. Reusing the same texture object means
// everything that draws with it picks up the new image without changing any references.
function setTextureImage(texture, url) {
  const image = new Image();
  image.onload = () => {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);  // images start at the top row, texture coordinates at the bottom
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    if (url.startsWith("blob:")) URL.revokeObjectURL(url);
  };
  image.src = url;
}

function loadTexture(url) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 255, 255]));
  setTextureImage(texture, url);
  return texture;
}

const brickTexture = loadTexture("brick.png");
const normalTexture = loadTexture("texture2.jpg");

// Brick texture picker: <input type="file" id="brickInput" accept="image/*"> in index.html
document.getElementById("brickInput")?.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) setTextureImage(brickTexture, URL.createObjectURL(file));
});

function loadVideoTexture(url) {
  const video = document.createElement("video");
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
function updateVideoTexture({ texture, video }) {
  if (video.readyState < video.HAVE_CURRENT_DATA) return;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
}

const tvVideo = loadVideoTexture("video.mp4");   // shared by every TV

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
// are relative to size. The TV's own space is translate(position), and every
// part is built in real units inside it, so triplanar bricks stay the same size on all TVs.
class TV {
  constructor({ position = [0, 0, 0], size = 1, aspect = 1, border = 0.2, depth = 0.5 } = {}) {
    this.position = position;
    this.size = size;
    this.aspect = aspect;
    this.border = border;            // width of the border strips, relative to size
    this.depth = depth;              // how far the inner walls run back, relative to size
    this.texture = brickTexture;     // body texture; reassign at any time
    this.videoCrop = [0, 0, 1, 1];   // part of the video frame shown: u0, v0, u1, v1 (set by layoutVideoWall)
    this.build();
  }

  // Work out every part; call again after changing size, aspect, border or depth
  build() {
    const b = this.border * this.size;     // border width
    const d = this.depth * this.size;      // inner wall depth
    const iy = this.size - b;              // opening half-height
    const ix = iy * this.aspect;           // opening half-width
    const hx = ix + b, hy = this.size, hz = this.size;   // box half-extents
    this.half = [hx, hy, hz];

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

    // Screen: a little larger than the opening so its edges tuck behind the inner walls.
    // createTvScreen takes arc lengths; asin turns the half-widths we want into them.
    const margin = 0.05 * this.size;
    const sx = ix + margin, sy = iy + margin;     // screen half-extents (straight across)
    const R = 3 * Math.max(hx, hy);               // curvature radius grows with the TV
    this.screenHalf = [sx, sy];
    this.screenMesh = createMesh(createTvScreen(30, 30, 2 * R * Math.asin(sx / R), 2 * R * Math.asin(sy / R), R));
    // Screen's front-most point is its center (local z = 0); put it just behind the opening
    this.screenLocal = translate(0, 0, hz - margin);
  }

  draw(view) {
    const space = translate(...this.position);
    gl.uniformMatrix4fv(uParentView, false, multiply(view, space));

    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    for (const { mesh, local } of this.parts) drawMesh(mesh, TRIPLANAR, local);

    gl.bindTexture(gl.TEXTURE_2D, tvVideo.texture);
    gl.uniform4fv(uVideoCrop, this.videoCrop);
    drawMesh(this.screenMesh, VIDEO, this.screenLocal);
  }
}

// Spread the shared video across all the screens like a video wall: each TV shows the part
// of the frame that falls on its screen, based on the TVs' positions. The frame covers the
// screens' bounding box without stretching; whatever sticks out past it is cropped evenly.
function layoutVideoWall(tvs, video) {
  const rects = tvs.map(({ position: [x, y], screenHalf: [sx, sy] }) => [x - sx, y - sy, x + sx, y + sy]);
  const x0 = Math.min(...rects.map(r => r[0])), y0 = Math.min(...rects.map(r => r[1]));
  const x1 = Math.max(...rects.map(r => r[2])), y1 = Math.max(...rects.map(r => r[3]));

  // Part of the frame the whole wall uses, centered
  const wallAspect = (x1 - x0) / (y1 - y0);
  const videoAspect = video.videoWidth ? video.videoWidth / video.videoHeight : wallAspect;
  const uSpan = Math.min(1, wallAspect / videoAspect);
  const vSpan = Math.min(1, videoAspect / wallAspect);
  const u0 = (1 - uSpan) / 2, v0 = (1 - vSpan) / 2;

  tvs.forEach((tv, i) => {
    const [a, b, c, d] = rects[i];
    tv.videoCrop = [
      u0 + (a - x0) / (x1 - x0) * uSpan, v0 + (b - y0) / (y1 - y0) * vSpan,
      u0 + (c - x0) / (x1 - x0) * uSpan, v0 + (d - y0) / (y1 - y0) * vSpan
    ];
  });
}

// Stack TVs in rows from the bottom up. Each row lists its TVs' aspects from left to right;
// rows are centered on x = 0 and sit on top of each other, and the whole stack is centered
// on the origin so the camera orbits around its middle.
function stackTVs(rows, gap = 0.05) {
  const built = rows.map(aspects => aspects.map(aspect => new TV({ aspect })));
  const rowHeights = built.map(row => Math.max(...row.map(tv => 2 * tv.half[1])));
  const stackHeight = rowHeights.reduce((a, b) => a + b, 0) + gap * (rows.length - 1);

  let y = -stackHeight / 2;                 // bottom of the current row
  built.forEach((row, r) => {
    const widths = row.map(tv => 2 * tv.half[0]);
    const rowWidth = widths.reduce((a, b) => a + b, 0) + gap * (row.length - 1);
    let x = -rowWidth / 2;                  // left edge of the next TV
    row.forEach((tv, i) => {
      tv.position = [x + widths[i] / 2, y + tv.half[1], 0];
      x += widths[i] + gap;
    });
    y += rowHeights[r] + gap;
  });
  return built.flat();
}

const tvs = stackTVs([
  [16 / 9, 1],   // bottom
  [1, 16 / 9],
  [16 / 9, 1],
  [16 / 9]       // top
]);

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0, 0, 0, 1);
const projection = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);
const lightWorld = [3, 4, 5];

// Interactive controls: the drag, slider and spin all orbit the camera around the world origin
let dragRotX = 0, dragRotY = -0.5;      // mouse drag
let camX = 0, camY = 0, camZ = -13;     // arrow keys, w, s; far enough back to see the whole stack
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

  // Camera: rotate the whole world about its origin, then move it in front of the camera.
  // Every object and the light go through the same view, so they all turn together.
  const axisRotation = ROTATE[axisSelect.value](sliderAngle * Math.PI / 180 + spinAngle);
  const rotation = multiply(multiply(rotateY(dragRotY), rotateX(dragRotX)), axisRotation);
  const view = multiply(translate(camX, camY, camZ), rotation);
  gl.uniformMatrix4fv(uProjection, false, projection);
  gl.uniform3fv(uLightPos, transformPoint(view, lightWorld));

  gl.activeTexture(gl.TEXTURE0);
  updateVideoTexture(tvVideo);

  layoutVideoWall(tvs, tvVideo.video);   // cheap; redone each frame so it follows the video size and TV positions
  for (const tv of tvs) tv.draw(view);

  // Cylinder
  gl.uniformMatrix4fv(uParentView, false, multiply(view, translate(4.5, 0, 0)));   // beside the stack
  gl.bindTexture(gl.TEXTURE_2D, normalTexture);
  drawMesh(cylinderMesh, NORMAL_MAPPING, IDENTITY);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
