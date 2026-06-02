import {
  DAG_WEBGL_EDGE_COLOR,
  DAG_WEBGL_NODE_RADIUS,
  DAG_WEBGL_TRIANGLE_DISPLAY_OFFSET,
  dagWebGlFill,
  dagWebGlStroke,
  type DagWebGlFillKey,
  type DagWebGlStrokeKey,
  type Rgba,
} from "./dagWebGlPalette";
import type { CanvasSize, DagNodeType, DagPayload, ViewTransform } from "./types";

const MAX_NODE_TYPES = 16;
const SHAPE_CODES: Record<string, number> = {
  circle: 0,
  upright_triangle: 1,
  upsidedown_triangle: 2,
  circle_with_cross: 3,
  pentagon: 4,
  circle_inside_circle: 5,
  upright_triangle_inscribed: 6,
  upsidedown_triangle_inscribed: 7,
  circle_with_cross_inscribed: 8,
  house: 9,
};

type NodeTypeStyle = {
  shape: string;
  fill: DagWebGlFillKey;
  stroke: DagWebGlStrokeKey;
};

const NODE_TYPE_STYLES: Record<string, NodeTypeStyle> = {
  basic_event: { shape: "circle", fill: "basic_event", stroke: "basic_event" },
  unknown_event: { shape: "circle", fill: "none", stroke: "none" },
  and_gate: { shape: "upright_triangle", fill: "and_gate", stroke: "and_gate" },
  nand_gate: { shape: "upright_triangle_inscribed", fill: "nand_gate", stroke: "nand_gate" },
  or_gate: { shape: "upsidedown_triangle", fill: "or_gate", stroke: "or_gate" },
  nor_gate: { shape: "upsidedown_triangle_inscribed", fill: "nor_gate", stroke: "nor_gate" },
  xor_gate: { shape: "circle_with_cross", fill: "xor_gate", stroke: "xor_gate" },
  xnor_gate: { shape: "circle_with_cross_inscribed", fill: "xnor_gate", stroke: "xnor_gate" },
  atleast_gate: { shape: "pentagon", fill: "atleast_gate", stroke: "atleast_gate" },
  not_gate: { shape: "circle_inside_circle", fill: "not_gate", stroke: "not_gate" },
  constant_event: { shape: "house", fill: "constant_event", stroke: "constant_event" },
};

type DagGeometry = {
  nodeCount: number;
  edgeVertexCount: number;
  nodeCenters: Float32Array;
  typeIndices: Float32Array;
  edgePositions: Float32Array;
};

type EdgeProgram = {
  program: WebGLProgram;
  position: number;
  resolution: WebGLUniformLocation;
  scale: WebGLUniformLocation;
  offset: WebGLUniformLocation;
  color: WebGLUniformLocation;
};

type NodeProgram = {
  program: WebGLProgram;
  center: number;
  typeIndex: number;
  resolution: WebGLUniformLocation;
  scale: WebGLUniformLocation;
  offset: WebGLUniformLocation;
  nodeRadius: WebGLUniformLocation;
  maxPointSize: WebGLUniformLocation;
  fillColors: WebGLUniformLocation;
  strokeColors: WebGLUniformLocation;
  fillEnabled: WebGLUniformLocation;
  shapeCodes: WebGLUniformLocation;
};

export class DagWebGlRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly geometry: DagGeometry;
  private readonly edgeProgram: EdgeProgram;
  private readonly nodeProgram: NodeProgram;
  private readonly edgeVao: WebGLVertexArrayObject;
  private readonly nodeVao: WebGLVertexArrayObject;
  private readonly edgeBuffer: WebGLBuffer;
  private readonly centerBuffer: WebGLBuffer;
  private readonly typeBuffer: WebGLBuffer;
  private readonly fillColors: Float32Array;
  private readonly strokeColors: Float32Array;
  private readonly fillEnabled: Float32Array;
  private readonly shapeCodes: Float32Array;
  private readonly edgeColor: Rgba;
  private readonly nodeRadius: number;
  private readonly maxPointSize: number;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    dag: DagPayload,
  ) {
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!gl) {
      throw new Error("WebGL2 is not available in this browser.");
    }

    this.gl = gl;
    this.geometry = prepareDagGeometry(dag);
    this.edgeProgram = createEdgeProgram(gl);
    this.nodeProgram = createNodeProgram(gl);
    this.edgeColor = DAG_WEBGL_EDGE_COLOR;
    const styleUniforms = buildStyleUniforms(dag.nodeTypes);
    this.fillColors = styleUniforms.fillColors;
    this.strokeColors = styleUniforms.strokeColors;
    this.fillEnabled = styleUniforms.fillEnabled;
    this.shapeCodes = styleUniforms.shapeCodes;
    this.nodeRadius = DAG_WEBGL_NODE_RADIUS;
    const pointSizeRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array | number[];
    this.maxPointSize = Math.max(1, Number(pointSizeRange[1]) || 64);

    const edge = createEdgeVao(gl, this.edgeProgram, this.geometry.edgePositions);
    this.edgeVao = edge.vao;
    this.edgeBuffer = edge.buffer;

    const node = createNodeVao(gl, this.nodeProgram, this.geometry.nodeCenters, this.geometry.typeIndices);
    this.nodeVao = node.vao;
    this.centerBuffer = node.centerBuffer;
    this.typeBuffer = node.typeBuffer;

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  resize(size: CanvasSize): void {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(size.width * dpr));
    this.canvas.height = Math.max(1, Math.round(size.height * dpr));
    this.canvas.style.width = `${size.width}px`;
    this.canvas.style.height = `${size.height}px`;
  }

  draw(size: CanvasSize, view: ViewTransform): void {
    const gl = this.gl;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (this.geometry.edgeVertexCount) {
      gl.useProgram(this.edgeProgram.program);
      gl.bindVertexArray(this.edgeVao);
      gl.uniform2f(this.edgeProgram.resolution, size.width, size.height);
      gl.uniform1f(this.edgeProgram.scale, view.scale);
      gl.uniform2f(this.edgeProgram.offset, view.offsetX, view.offsetY);
      gl.uniform4fv(this.edgeProgram.color, this.edgeColor);
      gl.drawArrays(gl.LINES, 0, this.geometry.edgeVertexCount);
    }

    if (this.geometry.nodeCount) {
      gl.useProgram(this.nodeProgram.program);
      gl.bindVertexArray(this.nodeVao);
      gl.uniform2f(this.nodeProgram.resolution, size.width, size.height);
      gl.uniform1f(this.nodeProgram.scale, view.scale);
      gl.uniform2f(this.nodeProgram.offset, view.offsetX, view.offsetY);
      gl.uniform1f(this.nodeProgram.nodeRadius, this.nodeRadius);
      gl.uniform1f(this.nodeProgram.maxPointSize, this.maxPointSize);
      gl.uniform4fv(this.nodeProgram.fillColors, this.fillColors);
      gl.uniform4fv(this.nodeProgram.strokeColors, this.strokeColors);
      gl.uniform1fv(this.nodeProgram.fillEnabled, this.fillEnabled);
      gl.uniform1fv(this.nodeProgram.shapeCodes, this.shapeCodes);
      gl.drawArrays(gl.POINTS, 0, this.geometry.nodeCount);
    }

    gl.bindVertexArray(null);
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteBuffer(this.edgeBuffer);
    gl.deleteBuffer(this.centerBuffer);
    gl.deleteBuffer(this.typeBuffer);
    gl.deleteVertexArray(this.edgeVao);
    gl.deleteVertexArray(this.nodeVao);
    gl.deleteProgram(this.edgeProgram.program);
    gl.deleteProgram(this.nodeProgram.program);
  }
}

function prepareDagGeometry(dag: DagPayload): DagGeometry {
  const nodeCount = Math.min(dag.nodes.count, dag.nodes.x.length, dag.nodes.y.length, dag.nodes.type.length);
  const nodeCenters = new Float32Array(nodeCount * 2);
  const typeIndices = new Float32Array(nodeCount);
  for (let index = 0; index < nodeCount; index += 1) {
    const graphX = dag.nodes.x[index];
    const graphY = dag.nodes.y[index];
    typeIndices[index] = Math.max(0, Math.min(MAX_NODE_TYPES - 1, dag.nodes.type[index] ?? 0));
    const displayOffset = triangleDisplayOffset(typeIndices[index], dag.nodeTypes);
    nodeCenters[index * 2] = graphX + displayOffset.x;
    nodeCenters[index * 2 + 1] = graphY + displayOffset.y;
  }

  const maxEdges = Math.min(dag.edges.count, dag.edges.source.length, dag.edges.target.length);
  const edgePositions = new Float32Array(maxEdges * 4);
  let edgeCursor = 0;
  for (let index = 0; index < maxEdges; index += 1) {
    const source = dag.edges.source[index];
    const target = dag.edges.target[index];
    if (!Number.isInteger(source) || !Number.isInteger(target)) continue;
    if (source < 0 || source >= nodeCount || target < 0 || target >= nodeCount) continue;
    edgePositions[edgeCursor * 4] = dag.nodes.x[source];
    edgePositions[edgeCursor * 4 + 1] = dag.nodes.y[source];
    edgePositions[edgeCursor * 4 + 2] = dag.nodes.x[target];
    edgePositions[edgeCursor * 4 + 3] = dag.nodes.y[target];
    edgeCursor += 1;
  }

  return {
    nodeCount,
    edgeVertexCount: edgeCursor * 2,
    nodeCenters,
    typeIndices,
    edgePositions: edgeCursor === maxEdges ? edgePositions : edgePositions.slice(0, edgeCursor * 4),
  };
}

function triangleDisplayOffset(typeIndex: number, nodeTypes: DagNodeType[]): { x: number; y: number } {
  const shape = shapeCodeForTypeIndex(typeIndex, nodeTypes);
  const magnitude = DAG_WEBGL_NODE_RADIUS * DAG_WEBGL_TRIANGLE_DISPLAY_OFFSET;
  // Shift the glyph toward the flat base so graph (x, y) sits on the vertex side of the outline.
  if (shape === 1 || shape === 6) return { x: 0, y: -magnitude };
  if (shape === 2 || shape === 7) return { x: 0, y: magnitude };
  // House: flat base at bottom — anchor graph (x, y) on the base like OR gates.
  if (shape === 9) return { x: 0, y: magnitude };
  return { x: 0, y: 0 };
}

function shapeCodeForTypeIndex(typeIndex: number, nodeTypes: DagNodeType[]): number {
  const style = nodeTypeStyle(nodeTypes[typeIndex]);
  return SHAPE_CODES[style.shape] ?? SHAPE_CODES.circle;
}

function createEdgeVao(gl: WebGL2RenderingContext, program: EdgeProgram, positions: Float32Array) {
  const vao = required(gl.createVertexArray(), "Unable to create edge VAO.");
  const buffer = required(gl.createBuffer(), "Unable to create edge buffer.");
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(program.position);
  gl.vertexAttribPointer(program.position, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, buffer };
}

function createNodeVao(
  gl: WebGL2RenderingContext,
  program: NodeProgram,
  centers: Float32Array,
  typeIndices: Float32Array,
) {
  const vao = required(gl.createVertexArray(), "Unable to create node VAO.");
  const centerBuffer = required(gl.createBuffer(), "Unable to create center buffer.");
  const typeBuffer = required(gl.createBuffer(), "Unable to create node type buffer.");

  gl.bindVertexArray(vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, centers, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(program.center);
  gl.vertexAttribPointer(program.center, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, typeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, typeIndices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(program.typeIndex);
  gl.vertexAttribPointer(program.typeIndex, 1, gl.FLOAT, false, 0, 0);

  gl.bindVertexArray(null);
  return { vao, centerBuffer, typeBuffer };
}

function createEdgeProgram(gl: WebGL2RenderingContext): EdgeProgram {
  const program = createProgram(gl, EDGE_VERTEX_SHADER, EDGE_FRAGMENT_SHADER);
  return {
    program,
    position: gl.getAttribLocation(program, "a_position"),
    resolution: required(gl.getUniformLocation(program, "u_resolution"), "Missing u_resolution."),
    scale: required(gl.getUniformLocation(program, "u_scale"), "Missing u_scale."),
    offset: required(gl.getUniformLocation(program, "u_offset"), "Missing u_offset."),
    color: required(gl.getUniformLocation(program, "u_color"), "Missing u_color."),
  };
}

function createNodeProgram(gl: WebGL2RenderingContext): NodeProgram {
  const program = createProgram(gl, NODE_VERTEX_SHADER, NODE_FRAGMENT_SHADER);
  return {
    program,
    center: gl.getAttribLocation(program, "a_center"),
    typeIndex: gl.getAttribLocation(program, "a_type_index"),
    resolution: required(gl.getUniformLocation(program, "u_resolution"), "Missing u_resolution."),
    scale: required(gl.getUniformLocation(program, "u_scale"), "Missing u_scale."),
    offset: required(gl.getUniformLocation(program, "u_offset"), "Missing u_offset."),
    nodeRadius: required(gl.getUniformLocation(program, "u_node_radius"), "Missing u_node_radius."),
    maxPointSize: required(gl.getUniformLocation(program, "u_max_point_size"), "Missing u_max_point_size."),
    fillColors: required(gl.getUniformLocation(program, "u_fill_colors[0]"), "Missing u_fill_colors."),
    strokeColors: required(gl.getUniformLocation(program, "u_stroke_colors[0]"), "Missing u_stroke_colors."),
    fillEnabled: required(gl.getUniformLocation(program, "u_fill_enabled[0]"), "Missing u_fill_enabled."),
    shapeCodes: required(gl.getUniformLocation(program, "u_shape_codes[0]"), "Missing u_shape_codes."),
  };
}

function createProgram(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = required(gl.createProgram(), "Unable to create shader program.");
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? "unknown link error";
    gl.deleteProgram(program);
    throw new Error(`WebGL program link failed: ${log}`);
  }
  return program;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = required(gl.createShader(type), "Unable to create shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "unknown compile error";
    gl.deleteShader(shader);
    throw new Error(`WebGL shader compile failed: ${log}`);
  }
  return shader;
}

function buildStyleUniforms(nodeTypes: DagNodeType[]) {
  const fillColors = new Float32Array(MAX_NODE_TYPES * 4);
  const strokeColors = new Float32Array(MAX_NODE_TYPES * 4);
  const fillEnabled = new Float32Array(MAX_NODE_TYPES);
  const shapeCodes = new Float32Array(MAX_NODE_TYPES);
  for (let index = 0; index < MAX_NODE_TYPES; index += 1) {
    const style = nodeTypeStyle(nodeTypes[index]);
    shapeCodes[index] = SHAPE_CODES[style.shape] ?? SHAPE_CODES.circle;
    fillEnabled[index] = style.fill === "none" ? 0 : 1;
    fillColors.set(dagWebGlFill(style.fill), index * 4);
    strokeColors.set(dagWebGlStroke(style.stroke), index * 4);
  }
  return { fillColors, strokeColors, fillEnabled, shapeCodes };
}

function nodeTypeStyle(nodeType: DagNodeType | undefined): NodeTypeStyle {
  if (!nodeType) return NODE_TYPE_STYLES.unknown_event;
  return NODE_TYPE_STYLES[nodeType] ?? NODE_TYPE_STYLES.unknown_event;
}

function required<T>(value: T | null, message: string): T {
  if (value === null) throw new Error(message);
  return value;
}

const EDGE_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
uniform vec2 u_resolution;
uniform float u_scale;
uniform vec2 u_offset;

void main() {
  vec2 screen = a_position * u_scale + u_offset;
  vec2 clip = (screen / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}
`;

const EDGE_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 out_color;

void main() {
  out_color = u_color;
}
`;

const NODE_VERTEX_SHADER = `#version 300 es
in vec2 a_center;
in float a_type_index;

uniform vec2 u_resolution;
uniform float u_scale;
uniform vec2 u_offset;
uniform float u_node_radius;
uniform float u_max_point_size;

out float v_type_index;
out float v_point_size;

void main() {
  vec2 screen = a_center * u_scale + u_offset;
  vec2 clip = (screen / u_resolution) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_type_index = a_type_index;
  v_point_size = clamp(u_node_radius * u_scale * 2.6, 4.0, u_max_point_size);
  gl_PointSize = v_point_size;
}
`;

const NODE_FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in float v_type_index;
in float v_point_size;

uniform vec4 u_fill_colors[${MAX_NODE_TYPES}];
uniform vec4 u_stroke_colors[${MAX_NODE_TYPES}];
uniform float u_fill_enabled[${MAX_NODE_TYPES}];
uniform float u_shape_codes[${MAX_NODE_TYPES}];

out vec4 out_color;

const float PI = 3.141592653589793;
const float TAU = 6.283185307179586;
// Match circle outline; polygons use inradius so circumradius stays inside the point sprite.
const float SHAPE_RADIUS = 0.78;
// Inscribed inner marker for negated gates (matches NOT / XNOR inner ring scale).
const float INSCRIBED_RADIUS = 0.42;
// Triangles need a smaller circumradius than circles for the same visual inset.
const float INSCRIBED_TRIANGLE_RADIUS = 0.34;

float sdCircle(vec2 p, float r) {
  return length(p) - r;
}

float sdRegularPolygon(vec2 p, float sides, float inradius, float rotation) {
  float angle = atan(p.y, p.x) - rotation;
  float sector = TAU / sides;
  float local = mod(angle + sector * 0.5, sector) - sector * 0.5;
  return length(p) * cos(local) - inradius;
}

float polygonInradius(float sides, float circumradius) {
  return circumradius * cos(PI / sides);
}

// Convex pentagon SDF (CCW vertices) for the constant "house" glyph.
float sdPolygon5(vec2 p, vec2 a, vec2 b, vec2 c, vec2 d, vec2 e) {
  vec2 verts[5] = vec2[5](a, b, c, d, e);
  float d2 = 1e10;
  float s = 1.0;
  for (int i = 0; i < 5; i++) {
    int j = i == 0 ? 4 : i - 1;
    vec2 vi = verts[i];
    vec2 vj = verts[j];
    vec2 ev = vi - vj;
    vec2 wv = p - vj;
    vec2 bv = wv - ev * clamp(dot(wv, ev) / dot(ev, ev), 0.0, 1.0);
    d2 = min(d2, dot(bv, bv));
    if (wv.x * ev.y - wv.y * ev.x > 0.0) s = -1.0;
  }
  return sqrt(d2) * s;
}

float sdHouse(vec2 p) {
  float w = SHAPE_RADIUS * 0.59;
  float base = -SHAPE_RADIUS * 0.88;
  float eave = SHAPE_RADIUS * 0.06;
  float peak = SHAPE_RADIUS * 0.92;
  return sdPolygon5(
    p,
    vec2(-w, base),
    vec2(w, base),
    vec2(w, eave),
    vec2(0.0, peak),
    vec2(-w, eave)
  );
}

float strokeAlpha(float sd, float width, float aa) {
  return 1.0 - smoothstep(width, width + aa, abs(sd));
}

float fillAlpha(float sd, float aa) {
  return 1.0 - smoothstep(0.0, aa, sd);
}

vec4 over(vec4 top, vec4 bottom) {
  float alpha = top.a + bottom.a * (1.0 - top.a);
  if (alpha <= 0.0) return vec4(0.0);
  vec3 rgb = (top.rgb * top.a + bottom.rgb * bottom.a * (1.0 - top.a)) / alpha;
  return vec4(rgb, alpha);
}

void main() {
  int typeIndex = clamp(int(v_type_index + 0.5), 0, ${MAX_NODE_TYPES - 1});
  int shape = clamp(int(u_shape_codes[typeIndex] + 0.5), 0, 9);
  vec2 p = vec2(gl_PointCoord.x * 2.0 - 1.0, 1.0 - gl_PointCoord.y * 2.0);

  float sd = sdCircle(p, SHAPE_RADIUS);
  if (shape == 1 || shape == 6) {
    sd = sdRegularPolygon(p, 3.0, polygonInradius(3.0, SHAPE_RADIUS), PI * 0.5);
  } else if (shape == 2 || shape == 7) {
    sd = sdRegularPolygon(p, 3.0, polygonInradius(3.0, SHAPE_RADIUS), -PI * 0.5);
  } else if (shape == 4) {
    sd = sdRegularPolygon(p, 5.0, polygonInradius(5.0, SHAPE_RADIUS), -PI * 0.5);
  } else if (shape == 9) {
    sd = sdHouse(p);
  }

  float aa = max(fwidth(sd), 2.0 / max(v_point_size, 1.0));
  float strokeWidth = max(0.045, 1.35 / max(v_point_size, 1.0));
  float inside = fillAlpha(sd, aa);
  float outline = strokeAlpha(sd, strokeWidth, aa);

  vec4 fill = u_fill_colors[typeIndex];
  vec4 stroke = u_stroke_colors[typeIndex];
  vec4 color = vec4(fill.rgb, fill.a * u_fill_enabled[typeIndex] * inside);
  color = over(vec4(stroke.rgb, stroke.a * outline), color);

  if (shape == 3 || shape == 8) {
    float diagA = abs(p.x - p.y) * 0.7071067811865476;
    float diagB = abs(p.x + p.y) * 0.7071067811865476;
    float crossSd = min(diagA, diagB);
    float cross = (1.0 - smoothstep(strokeWidth, strokeWidth + aa, crossSd)) * inside;
    color = over(vec4(stroke.rgb, stroke.a * cross), color);
  }

  if (shape == 5 || shape == 8) {
    float innerSd = sdCircle(p, INSCRIBED_RADIUS);
    float inner = strokeAlpha(innerSd, strokeWidth, aa) * inside;
    color = over(vec4(stroke.rgb, stroke.a * inner), color);
  } else if (shape == 6) {
    float innerSd = sdRegularPolygon(p, 3.0, polygonInradius(3.0, INSCRIBED_TRIANGLE_RADIUS), PI * 0.5);
    float inner = strokeAlpha(innerSd, strokeWidth, aa) * inside;
    color = over(vec4(stroke.rgb, stroke.a * inner), color);
  } else if (shape == 7) {
    float innerSd = sdRegularPolygon(p, 3.0, polygonInradius(3.0, INSCRIBED_TRIANGLE_RADIUS), -PI * 0.5);
    float inner = strokeAlpha(innerSd, strokeWidth, aa) * inside;
    color = over(vec4(stroke.rgb, stroke.a * inner), color);
  }

  if (color.a <= 0.001) discard;
  out_color = color;
}
`;
