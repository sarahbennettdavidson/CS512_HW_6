function createCube() {
  const positions = new Float32Array([
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,   // front
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);
  const faceNormals = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  const normals = new Float32Array(faceNormals.flatMap(n => [...n, ...n, ...n, ...n]));

  // Tangent; bitangent = cross(N, T)
  const faceTangents = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]];
  const tangents = new Float32Array(faceTangents.flatMap(t => [...t, ...t, ...t, ...t]));

  const texCoords = new Float32Array(faceNormals.flatMap(() => [0, 0, 1, 0, 1, 1, 0, 1]));

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,   // top
    12, 13, 14,  12, 14, 15,   // bottom
    16, 17, 18,  16, 18, 19,   // right
    20, 21, 22,  20, 22, 23    // left
  ]);
  return { positions, normals, tangents, texCoords, indices };
}


function createCylinder(segments = 64, uRepeat = 3) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // Side: theta grows with u, so T points along the circle and B = cross(N, T) points up
  for (let i = 0; i <= segments; i++) {
    const u = i / segments, theta = 2 * Math.PI * u;
    const s = Math.sin(theta), c = Math.cos(theta);
    for (const y of [-1, 1]) {
      positions.push(s, y, c);
      normals.push(s, 0, c);
      tangents.push(c, 0, -s);
      texCoords.push(u * uRepeat, (y + 1) / 2);
    }
  }
  for (let i = 0; i < segments; i++) {
    const k = 2 * i;
    indices.push(k, k + 2, k + 3, k, k + 3, k + 1);
  }

  // Caps: planar mapping of x, z into the texture, T = +x on both
  for (const y of [1, -1]) {
    const center = positions.length / 3;
    positions.push(0, y, 0);
    normals.push(0, y, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5, 0.5);
    for (let i = 0; i <= segments; i++) {
      const theta = 2 * Math.PI * i / segments;
      const s = Math.sin(theta), c = Math.cos(theta);
      positions.push(s, y, c);
      normals.push(0, y, 0);
      tangents.push(1, 0, 0);
      texCoords.push(0.5 + 0.5 * s, 0.5 - 0.5 * y * c);  // keeps cross(N, T) along +v
    }
    for (let i = 0; i < segments; i++) {
      const a = center + 1 + i, b = a + 1;
      if (y > 0) indices.push(center, a, b);
      else indices.push(center, b, a);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

function createTvScreen(widthRes, heightRes, screenWidth, screenHeight, curvatureRadius) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];
  
  // Angular spread of the screen based on its physical size and curvature radius
  const maxAngleX = screenWidth / curvatureRadius;
  const maxAngleY = screenHeight / curvatureRadius;

  for (let yIndex = 0; yIndex <= heightRes; yIndex++) {
    for (let xIndex = 0; xIndex <= widthRes; xIndex++) {
      const u = xIndex / widthRes;
      const v = yIndex / heightRes;

      // Map UV to angles centered at 0
      const theta = (u - 0.5) * maxAngleX; // horizontal angle
      const phi = (v - 0.5) * maxAngleY;   // vertical angle

      // Spherical patch equations (curves edge-to-edge with no flat margins)
      const x = curvatureRadius * Math.sin(theta);
      const y = curvatureRadius * Math.sin(phi);
      const z = curvatureRadius * (Math.cos(theta) * Math.cos(phi) - 1);

      positions.push(x, y, z);
      texCoords.push(u, v);

      // Normals pointing outward from the curvature center
      const nx = Math.sin(theta);
      const ny = Math.sin(phi);
      const nz = Math.cos(theta) * Math.cos(phi);
      normals.push(nx, ny, nz);

      // Tangent vector along the U direction
      const tx = Math.cos(theta);
      const ty = 0;
      const tz = -Math.sin(theta);
      const tLen = Math.sqrt(tx * tx + tz * tz);
      tangents.push(tx / tLen, ty / tLen, tz / tLen);
    }
  }

  for (let yIndex = 0; yIndex < heightRes; yIndex++) {
    for (let xIndex = 0; xIndex < widthRes; xIndex++) {
      const a = yIndex * (widthRes + 1) + xIndex;
      const b = a + widthRes + 1;
      indices.push(a, a + 1, b,    b, a + 1, b + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

function createOpenCube() {
  const positions = new Float32Array([
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);
  const faceNormals = [[0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]];
  const normals = new Float32Array(faceNormals.flatMap(n => [...n, ...n, ...n, ...n]));

  // Tangent; bitangent = cross(N, T)
  const faceTangents = [[-1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]];
  const tangents = new Float32Array(faceTangents.flatMap(t => [...t, ...t, ...t, ...t]));

  const texCoords = new Float32Array(faceNormals.flatMap(() => [0, 0, 1, 0, 1, 1, 0, 1]));

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // back
     4,  5,  6,   4,  6,  7,   // top
     8,  9, 10,   8, 10, 11,   // bottom
    12, 13, 14,  12, 14, 15,   // right
    16, 17, 18,  16, 18, 19    // left
  ]);
  return { positions, normals, tangents, texCoords, indices };
}

function createPlane() {
  const positions = new Float32Array([
    -0.5, -0.5, 0,   0.5, -0.5, 0,   0.5, 0.5, 0,   -0.5, 0.5, 0
  ]);
  const normals = new Float32Array([0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1]);
  const tangents = new Float32Array([1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0]);
  const texCoords = new Float32Array([0, 0,  1, 0,  1, 1,  0, 1]);
  const indices = new Uint16Array([0, 1, 2,  0, 2, 3]);
  return { positions, normals, tangents, texCoords, indices };
}

function makeSphere(latRes, lonRes) {
  const pos = [], col = [], uv = [], idx = [];
  for (let lat = 0; lat <= latRes; lat++) {
    const latAngle = lat * Math.PI / latRes;          // 0 at top, PI at bottom
    const sinT = Math.sin(latAngle), cosT = Math.cos(latAngle);
    for (let lon = 0; lon <= lonRes; lon++) {
      const lonAngle = lon * 2 * Math.PI / lonRes;    // around the ring
      const x = sinT * Math.cos(lonAngle);
      const y = cosT;
      const z = sinT * Math.sin(lonAngle);
      pos.push(x, y, z);
      col.push((x + 1) / 2, (y + 1) / 2, (z + 1) / 2);
      uv.push(lon / lonRes, 1 - lat / latRes);
    }
  }

  for (let lat = 0; lat < latRes; lat++) {
    for (let lon = 0; lon < lonRes; lon++) {
      const a = lat * (lonRes + 1) + lon;  // current ring
      const b = a + lonRes + 1;            // next ring down
      idx.push(a, a + 1, b,   b, a + 1, b + 1);
    }
  }

  return {
    sphPositions: new Float32Array(pos),
    sphColors: new Float32Array(col),
    sphUVs: new Float32Array(uv),
    sphIndices: new Uint16Array(idx)
  };
}

const sphere = makeSphere(32, 32);
const spherePos = sphere.sphPositions;
const sphereColors = sphere.sphColors;
const sphereIndices = sphere.sphIndices;
const sphereUVs = sphere.sphUVs;
}
