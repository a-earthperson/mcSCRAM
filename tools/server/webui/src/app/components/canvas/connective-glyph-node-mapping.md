# DAG WebGL Connective, Glyph, Node, Edge, Layer, and Partition Mapping

Last audited: 2026-06-01.

This document describes the WebGL DAG views rendered by `DagWebGlDiagram` and
`drawDagWebGl`. It covers the three DAG panels in `Playground`: source MEF DAG,
compiled PDAG, and kernel execution DAG. It does not describe the older 2D
`FaultTreeDiagram` glyph decisions except where `canvas.css` affects the shared
frame or the final perceived WebGL canvas colors.

## Ownership Boundaries

| Artifact decision | Owner | Current implementation |
|---|---|---|
| HTTP route shape and compile query options | Backend API server | `tools/server/src/fuzzybits_server/routes.py` in fuzzybit-org/fuzzybits exposes `/model/source-dag`, `/model/dag`, `/model/kernel-dag`, and `/model/dag-bundle`; compile flags are converted into native compile options there. |
| Stable DAG node type vocabulary and integer indices | Backend API server | `DAG_NODE_TYPES` in fuzzybit-org/fuzzybits `tools/server/src/fuzzybits_server/fault_tree_payload.py`. The client trusts `payload.nodeTypes[typeIndex]`. |
| Source MEF tag to DAG type index | Backend API server | `_parse_fault_tree_graph` and `NODE_TYPE_BY_CONNECTIVE` in `fault_tree_payload.py`. |
| Compiled/native graph node kind/connective data | Backend Python binding for `fuzzybits`, backed by native fuzzybits | `fuzzybits.native().run_compile(...).to_graph()` and `.to_kernel_graph()` provide nodes, edges, root complement, graph kind, and optional kernel metadata. The binding source is in `targets/python/bindings.cpp`; the Python server consumes its dictionaries. |
| Compiled/native kind/connective to DAG type index | Backend API server | `_compiled_node_type` maps native `kind` and `connective` strings to `DAG_NODE_TYPES`. |
| Kernel layer and partition grouping | Native mcSCRAM layer manager plus backend layout | `layer_manager::describe_kernel_dag` groups nodes by execution layer and homogeneous kernel partition. `_position_layer_nodes` uses layer/partition only to choose world coordinates. |
| WebGL node shape and fill/stroke role by type name | Client-side JavaScript | `NODE_TYPE_STYLES` and `SHAPE_CODES` in `drawDagWebGl.ts`. |
| WebGL concrete colors | Client-side JavaScript palette | `dagWebGlPalette.ts`; values are sRGB-encoded floats written directly to the framebuffer. |
| WebGL glyph geometry, stroke, antialiasing, and point sprite sizing | Client-side WebGL shader | `NODE_VERTEX_SHADER` and `NODE_FRAGMENT_SHADER` in `drawDagWebGl.ts`. |
| Edge geometry and edge color | Backend API server for topology/coordinates; client WebGL for draw color | Backend emits `edges.source`/`edges.target` and `edges.complemented`; client draws regular edges as `gl.LINES` and complemented edges as screen-space quads. |
| Canvas frame, toolbar, cursor, final blend with page | Hardcoded CSS | `canvas.css` owns `.fault-tree-frame`, `.fault-tree-toolbar`, `.fault-tree-canvas`, `.dag-webgl-frame`, and `.dag-webgl-canvas { mix-blend-mode: exclusion; }`. |
| Semantic disambiguation of complemented/signed edges in the visual DAG | Backend API server plus client WebGL | `_compiled_graph_to_render_graph` stores negative native edge targets as `edges.complemented[j] = 1`; the client renders those edges thicker and amber. Root complement is still metadata only. |
| Dedicated visual styling for `null_gate` and `constant_event` | Client-side JavaScript plus WebGL shader | `NODE_TYPE_STYLES` maps `null_gate` to a square and `constant_event` to a house glyph. |

## Payloads and Views

| View | Frontend fetch | Server builder | Native dependency | Layout source | Notes |
|---|---|---|---|---|---|
| Source MEF DAG | `fetchDagBundleForModel(...).source` | `_build_source_dag_payload` | None | Backend depth layout | Parsed directly from XML. Connective tag names drive type indices. |
| Compiled PDAG | `fetchDagBundleForModel(...).compiled` | `_build_native_dag_payload(..., graph_kind="compiled")` or bundle path | `fuzzybits.native().run_compile(...).to_graph()` | Backend depth layout unless every native node includes `layer` and `partition` | Compile options can preserve, expand, or recognize threshold/XOR gates before this graph is exposed. |
| Kernel execution DAG | `fetchDagBundleForModel(...).kernel` | `_build_native_dag_payload(..., graph_kind="kernel")` or bundle path | `fuzzybits.native().run_compile(...).to_kernel_graph()` | Native layer/partition metadata, then backend coordinate rotation | Backend rotates kernel payload coordinates visually counter-clockwise with `(x, y) -> (y, -x)`. |

All three views are rendered by the same `DagWebGlRenderer`. The renderer does
not inspect `graphKind`, `nodes.layer`, `nodes.partition`, or `kernel.partitions`
while drawing. Those fields matter only before drawing, when the backend chooses
the node coordinates.

## Stable Node Type Contract

The server emits:

```text
nodeTypes = [
  "basic_event",
  "unknown_event",
  "and_gate",
  "nand_gate",
  "or_gate",
  "nor_gate",
  "xor_gate",
  "xnor_gate",
  "atleast_gate",
  "not_gate",
  "null_gate",
  "constant_event",
]
```

Each rendered vertex carries an integer `nodes.type[i]`. The client clamps each
type index to `[0, 15]`, looks up `nodeTypes[index]`, maps the type name to a
style, and falls back to `unknown_event` when the type name is missing or has no
client-side style.

Important fallback behavior:

| Case | Current behavior | Owner |
|---|---|---|
| Missing `nodeTypes[index]` for indices 12-15 | Unknown-event outline | Client-side JavaScript fallback |
| Type index greater than 15 | Clamped to 15, then unknown-event outline | Client-side JavaScript fallback |
| Negative type index | Clamped to 0, therefore drawn as `basic_event` | Client-side JavaScript clamp |
| Backend emits `null_gate` at index 10 | Styled as an amber square using the `not_gate` palette role | Client-side JavaScript style table |
| Backend emits `constant_event` at index 11 | Styled as a fuchsia house glyph using the `constant_event` palette role | Client-side JavaScript style table |
| Unknown source XML connective | Type index 1, unknown-event outline | Backend source parser plus client style |
| Unknown compiled native kind/connective | Type index 1, unknown-event outline | Backend native graph adapter plus client style |

## Node Glyph and Color Table

Colors below are sRGB floats. The hex values are the palette comments' baked
sRGB approximations. Fill alpha is intentionally low; stroke is the dominant
visual channel.

| Server index | Type name | Source XML connective or event | Native compiled/kernel source | Shape | Fill | Stroke | Visible overlap or ambiguity | Decision owner |
|---:|---|---|---|---|---|---|---|---|
| 0 | `basic_event` | `<define-basic-event>` and `<basic-event name="...">` references | Native `kind == "basic_event"`; kernel partitions use `kind="basic_event", connective="sample"` | Circle | `[0.808844, 0.795639, 0.249941, 0.22]` / `#cecb40` | `[0.659408, 0.644099, 0.019619, 1.0]` / `#a8a405` | Unique color/shape among mapped supported types. | Backend type index; client JS style; WebGL shader geometry; palette color |
| 1 | `unknown_event` | Unknown connective, unresolved gate placeholder, or non-gate/non-basic native kind | Native non-gate, non-basic, non-constant kind; unknown connective candidate | Circle, outline-only | Fill role is `none`; red fallback fill `[1, 0, 0, 1]` exists but `fillEnabled=0`, so it is not visible | `DAG_WEBGL_STROKE_NONE = [0.599156, 0.595092, 0.810668, 0.9]` / approx `#9998cf` | Same visible style as missing `null_gate`, missing `constant_event`, and indices 12-15. | Backend fallback plus client fallback |
| 2 | `and_gate` | `<and>` | Native `connective == "and"` | Upright triangle | `[0.096146, 0.691099, 0.977711, 0.2]` / `#19b0f9` | `[0.01793, 0.640085, 0.913893, 1.0]` / `#05a3e9` | Same shape and color as `nand_gate`; no negation marker. | Backend mapping; client JS style; shader polygon |
| 3 | `nand_gate` | `<nand>` | Native `connective == "nand"` | Upright triangle | Same as `and_gate` | Same as `and_gate` | Visually indistinguishable from `and_gate`. | Backend mapping; client JS style |
| 4 | `or_gate` | `<or>` | Native `connective == "or"` | Upside-down triangle | `[0.979533, 0.402418, 0.450999, 0.2]` / `#fa6773` | `[0.909049, 0.338076, 0.393436, 1.0]` / `#e85664` | Same shape and color as `nor_gate`; no negation marker. | Backend mapping; client JS style; shader polygon |
| 5 | `nor_gate` | `<nor>` | Native `connective == "nor"` | Upside-down triangle | Same as `or_gate` | Same as `or_gate` | Visually indistinguishable from `or_gate`. | Backend mapping; client JS style |
| 6 | `xor_gate` | `<xor>` | Native `connective == "xor"` | Circle with diagonal cross | `[0.340127, 0.806187, 0.438209, 0.2]` / `#57ce70` | `[0.222174, 0.704073, 0.341718, 1.0]` / `#39b457` | Same shape and color as `xnor_gate`; no negation marker. | Backend mapping; client JS style; shader circle/cross |
| 7 | `xnor_gate` | `<xnor>` | Backend can map native `connective == "xnor"` if binding emits it; native kernel partition order inspected here does not include xnor | Circle with diagonal cross | Same as `xor_gate` | Same as `xor_gate` | Visually indistinguishable from `xor_gate`. Kernel DAG may not emit it directly. | Backend mapping; client JS style; native availability uncertain |
| 8 | `atleast_gate` | `<atleast min="...">` | Native `connective == "atleast"` | Pentagon, currently point-up after rotation fix | `[0.724892, 0.488624, 0.967278, 0.2]` / `#b97df7` | `[0.663468, 0.427058, 0.90006, 1.0]` / `#a96de6` | Unique mapped gate style. Source `min` is not rendered in WebGL. | Backend mapping; client JS style; shader polygon |
| 9 | `not_gate` | `<not>` | Native `connective == "not"` | Circle inside circle | `[0.977485, 0.631504, 0.160365, 0.2]` / `#f9a129` | `[0.856043, 0.539825, 0.067185, 1.0]` / `#da8a11` | Unique mapped gate style. | Backend mapping; client JS style; shader circle/inner ring |
| 10 | `null_gate` | Not mapped by source parser except via native compiled `connective == "null"` | Native `connective == "null"` from kernel connective order | Square | Same as `not_gate`: `[0.977485, 0.631504, 0.160365, 0.2]` / `#f9a129` | Same as `not_gate`: `[0.856043, 0.539825, 0.067185, 1.0]` / `#da8a11` | Color overlaps with `not_gate`; shape is distinct. | Backend/native can emit; client JS style and shader box SDF |
| 11 | `constant_event` | Not emitted by source parser | Native `kind == "constant"` | House | `[0.895947, 0.604909, 0.853011, 0.22]` / `#e49ada` | `[0.752398, 0.448841, 0.711252, 1.0]` / `#c072b5` | Unique mapped style. | Backend/native can emit; client JS style and shader house SDF |

## Shape Implementation Details

The shader receives a per-type `shapeCode` uniform array. It defaults to circle
and switches only for triangles, pentagon, cross, and inner circle overlays.

| Shape name | Shape code | Shader SDF path | Orientation and details |
|---|---:|---|---|
| `circle` | 0 | `sdCircle(p, SHAPE_RADIUS)` | Base shape for events and unknowns. |
| `upright_triangle` | 1 | `sdRegularPolygon(..., 3, ..., PI * 0.5)` | Used by AND/NAND. |
| `upsidedown_triangle` | 2 | `sdRegularPolygon(..., 3, ..., -PI * 0.5)` | Used by OR/NOR. |
| `circle_with_cross` | 3 | Circle SDF plus two diagonal stroke SDFs | Used by XOR/XNOR. |
| `pentagon` | 4 | `sdRegularPolygon(..., 5, ..., -PI * 0.5)` | Used by ATLEAST. The sign is intentionally negative so the pentagon is not flipped by 180 degrees. |
| `circle_inside_circle` | 5 | Circle SDF plus inner circle stroke at radius `0.42` | Used by NOT. |
| `upright_triangle_inscribed` | 6 | Upright triangle plus smaller inner triangle stroke | Used by NAND. |
| `upsidedown_triangle_inscribed` | 7 | Upside-down triangle plus smaller inner triangle stroke | Used by NOR. |
| `circle_with_cross_inscribed` | 8 | Circle with diagonal cross plus inner circle stroke | Used by XNOR. |
| `house` | 9 | Convex five-vertex house SDF | Used by constants. |
| `square` | 10 | Axis-aligned box SDF | Used by null gates. |

Global shader constants:

| Constant or expression | Meaning |
|---|---|
| `DAG_WEBGL_NODE_RADIUS = 0.666` | World-space node radius owned by `dagWebGlPalette.ts`; used by vertex shader point size. |
| `gl_PointSize = clamp(u_node_radius * u_scale * 2.6, 4.0, u_max_point_size)` | Nodes are WebGL points, not meshes. Minimum rendered point is 4 px; maximum comes from `ALIASED_POINT_SIZE_RANGE`. |
| `SHAPE_RADIUS = 0.78` | Shape radius inside normalized point-sprite coordinates. |
| `polygonInradius(sides, SHAPE_RADIUS)` | Polygons are passed an inradius so their circumradius remains inside the point sprite. |
| `strokeWidth = max(0.045, 1.35 / max(v_point_size, 1.0))` | Stroke remains visible as nodes shrink. |
| `aa = max(fwidth(sd), 2.0 / max(v_point_size, 1.0))` | Shader-side antialias width. |

## Edge Mapping

| Aspect | Current behavior | Owner |
|---|---|---|
| Edge payload shape | `edges.source[j]` and `edges.target[j]` are integer node indices. | Backend API server |
| Source parser direction | Parent gate/source -> child event or gate/target. | Backend source parser |
| Compiled graph direction | Native edge source -> target, with target sign removed by `abs(int(edge["target"]))`. | Native binding emits; backend adapter normalizes |
| Edge validation | Client drops non-integer endpoints and endpoints outside `[0, nodeCount)`. | Client-side JavaScript |
| Draw primitive | `gl.LINES`, two vertices per edge, no arrowhead. | Client WebGL |
| Edge geometry | Straight line from source center to target center. | Client WebGL |
| Edge color | `DAG_WEBGL_EDGE_COLOR = [0.205578, 0.577387, 0.680913, 0.5]` / `#3493ae` at 50% alpha. | Client palette |
| Complement/signed edge styling | Native negative target signs are exposed as `edges.complemented[j] = 1`; client renders those edges as amber screen-space quads with half-width `DAG_WEBGL_COMPLEMENTED_EDGE_WIDTH_PX = 2.75`. `rootComplement` is metadata only. | Backend adapter plus client WebGL and palette |
| Edge layer/partition styling | None. Layers and partitions do not change line color, width, dashing, or alpha. | No current visual owner |

Because edges have no arrowheads, dashing, or labels, the visual canvas
communicates connectivity and complemented native arguments, but not root
complement state or edge direction except by spatial convention.

## Layers, Levels, and Partitions

The code uses several overloaded terms:

| Term | Meaning | Visual effect | Owner |
|---|---|---|---|
| Source/compiled DAG depth layer | Topological depth computed by `_dag_depths`; roots begin at layer 0 and children are placed at deeper layers. | Sets backend `y = layer`; source and ordinary compiled DAGs appear vertically layered. | Backend API server |
| Native execution layer | Layer assigned by native mcSCRAM `LayeredTopologicalOrder` and exposed by kernel DAG metadata. | Used by backend kernel layout. After kernel rotation, layers become the dominant horizontal axis. | Native mcSCRAM plus backend API server |
| Kernel partition | A homogeneous execution group within a native layer: one basic-event sampling group if present, plus gate groups by connective. | Used only for backend x-position grouping before kernel rotation. No color, boundary, label, or legend is drawn. | Native mcSCRAM plus backend API server |
| Compiler level `C0` through `C8` | Compile option `compilation_level` from the Playground controls and server route query. | Changes the graph the backend receives from native compilation; no direct glyph/color encoding. | Client control plus backend API plus native compiler |

### Source and Ordinary Compiled Layout

When not every node has both `layer` and `partition`, `_columnar_dag_payload`
uses `_dag_depths`:

1. Initial roots with zero indegree are layer 0.
2. Each child depth is `max(parentDepth + 1)`.
3. Nodes in the same layer are placed at consecutive x offsets centered around 0.
4. The backend emits `nodes.x`, `nodes.y`, `nodes.layer`, and `nodes.partition`.
5. The WebGL renderer uses only `x`, `y`, and `type`.

### Kernel Layout

When every node has `layer` and `partition`, `_columnar_dag_payload` calls
`_position_layer_nodes`:

1. Nodes are grouped by integer partition inside each layer.
2. Partitions are ordered by partition id.
3. Nodes inside a partition are sorted by integer node id.
4. Partition width is `max(1.0, 0.35 * (nodeCount - 1) + 1.0)`.
5. Adjacent partitions have `0.8` world-unit gaps.
6. Nodes inside a partition are spaced by `0.35` world units.
7. `y = layer`.
8. For `graphKind == "kernel_dag"`, the backend rotates all coordinates with
   `x' = y`, `y' = -x`, then recomputes bounds.

Native `layer_manager::describe_kernel_dag` creates partition metadata in this
order for each layer:

1. Basic events, if any: `kind = "basic_event"`, `connective = "sample"`.
2. Gates by connective order: `and`, `or`, `atleast`, `xor`, `not`, `nand`,
   `nor`, `null`.

That partition connective order is not itself a color/shape legend. The actual
node glyph still comes from the server-emitted type index and client style table.

## Compile Options That Affect Which Glyphs Exist

The Playground sends these options to `/model/dag-bundle`:

| UI control | Query option | Effect on visual glyph set |
|---|---|---|
| Compiler `C0`-`C8` | `compilation_level` | Changes native normalization passes and therefore the node set, edges, and possibly which connectives survive. |
| Expand `K/N` | `expand_atleast_gates` | When true, native compilation may replace `atleast` gates with AND/OR structure, removing pentagons from compiled/kernel views. |
| Expand `XOR` | `expand_xor_gates` | When true, native compilation may replace XOR gates with AND/OR structure, removing circle-with-cross glyphs from compiled/kernel views. |
| Merge `K/N` | `recognize_atleast_groups` | When true, native compilation may recognize AND/OR structure as `atleast`, introducing pentagons. |
| Merge `XOR` | `recognize_xor_groups` | When true, native compilation may recognize structure as XOR, introducing circle-with-cross glyphs. |
| CCF | `ccf` | Always sent as true by current Playground bundle fetch. It can affect native graph shape but has no direct glyph. |
| Keep null gates | `keep_null_gates` | When true, native graph may preserve null gates. These render as amber squares. |

## CSS and Perceived Color

`canvas.css` does not provide WebGL palette colors. It only provides the shared
frame, toolbar, cursor, sizing, and final blend mode:

| CSS selector | Effect on WebGL DAG |
|---|---|
| `.fault-tree-frame` | Shared container size (`height: 680px`), border, overflow, transparent background. |
| `.dag-webgl-frame` | Adds `margin-top: 18px`. |
| `.fault-tree-canvas` | Canvas display, size, transparent background, grab cursor, touch-action suppression. |
| `.dag-webgl-canvas` | Applies `mix-blend-mode: exclusion`. This can change perceived colors against the page/background even though shader output is fixed. |
| `.playground-dag-view--kernel .fault-tree-frame` | Kernel panel height becomes `calc(100vh - 48px)`. |

The many `--ft-*` CSS variables in `canvas.css` are consumed by the 2D
`FaultTreeDiagram` code path, not by `DagWebGlRenderer`.

## Known Overlaps, Ambiguities, and Missing Mappings

| Issue | Consequence | Likely owner to fix |
|---|---|---|
| `and_gate` and `nand_gate` share identical triangle shape and color. | NAND is visually indistinguishable from AND. | Client-side DAG style decision, possibly shader if adding a negation marker. |
| `or_gate` and `nor_gate` share identical triangle shape and color. | NOR is visually indistinguishable from OR. | Client-side DAG style decision, possibly shader if adding a negation marker. |
| `xor_gate` and `xnor_gate` share identical circle-cross shape and color. | XNOR is visually indistinguishable from XOR. | Client-side DAG style decision, possibly shader if adding a negation marker. |
| `null_gate` shares the amber palette role with `not_gate`. | Null is distinguishable by square shape but not by hue. | Client-side JavaScript style table and palette. |
| Unknown/missing types and indices 12-15 share the unknown outline. | Multiple invalid or unsupported semantic cases collapse into one visible glyph. | Client-side style table or backend type contract. |
| Source `atleast min` is not present in WebGL DAG payload labels. | A 2-of-3 and 3-of-5 threshold both draw as the same pentagon. | Backend payload contract plus client label/glyph decision. |
| Edges encode complemented native arguments but not root complement state. | Signed native child arguments are visible; a complemented top/root remains invisible except in payload metadata. | Backend adapter and client edge shader/style decision. |
| `rootComplement` is emitted for compiled/native payloads but not drawn. | A complemented top event has no visible indicator. | Client-side visualization decision. |
| `nodes.layer`, `nodes.partition`, and `kernel.partitions` are not rendered as labels or colors. | Kernel partitioning is inferable only from spatial grouping. | Client-side visualization decision. |
| Type clamp maps negative type indices to `basic_event`. | Corrupt negative type data would appear as a valid basic event. | Client-side validation/fallback decision. |
| `mix-blend-mode: exclusion` changes final perceived color by background. | Palette values are not guaranteed to visually match raw sRGB floats on every background. | CSS design decision. |
| Backend Python binding source for `fuzzybits` is in `targets/python/bindings.cpp`. | Exact native dictionary ownership is defined at the Python API boundary and native layer-manager code. | Binding is in-tree; the server consumes its dictionaries. |

## Practical Legend for the Current WebGL View

Use this as the current viewer legend:

| Visual glyph | Meaning in current mapped set | Caveats |
|---|---|---|
| Yellow circle | Basic event | Unique among mapped standard types. |
| Lavender outline-only circle | Unknown event or unstyled emitted type | Also represents missing type styles and unused indices. |
| Cyan upright triangle | AND-family gate | AND and NAND overlap. |
| Red upside-down triangle | OR-family gate | OR and NOR overlap. |
| Green circle with diagonal cross | XOR-family gate | XOR and XNOR overlap. |
| Purple pentagon | K-of-N / ATLEAST gate | Threshold `min` value is not shown. |
| Amber circle with inner ring | NOT gate | Unique among mapped standard types. |
| Amber square | NULL gate | Shares hue with NOT; shape distinguishes it. |
| Fuchsia house | Constant event | Native compiled/kernel DAG only. |
| Blue-green straight line | Unsigned dependency edge from source index to target index | No arrowhead, layer, or partition styling. |
| Amber thick straight line | Complemented dependency edge from source index to target index | Represents a negative native edge target; root complement is not shown. |
