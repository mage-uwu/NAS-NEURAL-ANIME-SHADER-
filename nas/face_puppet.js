// Face puppet for char_run.glb: animates the model's own painted face by deforming its mesh
// (Live2D-style), plus head/neck look-at on the rig. No overlays, no extra geometry.
//
// The asset has no blendshapes and no face bones, but the face mesh is dense (~170-270 vertices per
// eye / around the mouth). Every deformation runs in the vertex stage, in head-local space:
//   x = right, y = up, z = forward, in head radii, origin at the head center (Head .. HeadTop_End bones).
//
//   blink   each eye's band (lower lid .. top of the lashes) is squashed toward the lower lid; the skin
//           above follows down with a falloff to the brow, so the painted lashes fold into a closed line
//   mouth   vertices are split at the painted mouth line: the lower lip drops, the upper lip lifts a
//           little, and the lip region scales horizontally (vowel width). The painted mouth is a thin
//           closed line, so the triangles the split stretches open are shaded as the mouth interior
//           (dark, tongue toward the lower lip): the opening's shape is the mesh's own gap.
//   look    yaw/pitch toward a target added on top of the animation, 40% neck / 60% head
//
// Only vertices on the face surface move: a cubic fit of the face (z as a function of x, y, from a
// calibration render) gates out back hair and the fringe hanging in front of the eyes.
//
// const puppet = createFacePuppet(THREE, mesh)
//   puppet.patch(material, colorSpace)  inject into any material; colorSpace 'srgb' | 'linear' for the
//                                       mouth interior color (the space the material's color math runs in)
//   puppet.beginFrame()         call before mixer.update (restores the un-turned head/neck pose)
//   puppet.update(dt)           call after mixer.update: applies look-at, refreshes head frame uniforms
//   puppet.blink[0|1], puppet.mouthOpen, puppet.mouthWidth, puppet.lookTarget (Vector3, world)
function createFacePuppet(THREE, mesh) {
  const bones = mesh.skeleton.bones;
  const bone = (suffix) => bones.find((b) => b.name.endsWith(suffix));
  const head = bone('Head'), headTop = bone('HeadTop_End'), headFront = bone('headfront'), neck = bone('Neck');

  const U = {
    puppetModelInv: { value: new THREE.Matrix4() },
    puppetCenter: { value: new THREE.Vector3() }, puppetRadius: { value: 0.1 },
    puppetRight: { value: new THREE.Vector3(1, 0, 0) }, puppetUp: { value: new THREE.Vector3(0, 1, 0) },
    puppetFwd: { value: new THREE.Vector3(0, 0, 1) },
    // Layout in head-local units, measured from a straight-on calibration render.
    puppetEyeA: { value: new THREE.Vector4(-0.387, -0.206, 0.17, 0.135) },  // her right eye: center, half-size
    puppetEyeB: { value: new THREE.Vector4(0.245, -0.18, 0.17, 0.135) },    // her left eye
    puppetMouth: { value: new THREE.Vector3(-0.116, -0.625, 0.085) },       // center, half-width
    puppetFaceFit: { value: [0.5239, -0.1684, 0.6862, -0.7158, 3.2962, -0.1908, 0.0465, 3.5262, 0.6349, -0.3110] },
    puppetBlink: { value: new THREE.Vector2(1, 1) },  // openness per eye
    puppetMouthOpen: { value: 0 }, puppetMouthWidth: { value: 1 },
  };

  const GLSL = `
    uniform mat4 puppetModelInv;
    uniform vec3 puppetCenter, puppetRight, puppetUp, puppetFwd; uniform float puppetRadius;
    uniform vec4 puppetEyeA, puppetEyeB; uniform vec3 puppetMouth;
    uniform float puppetFaceFit[10];
    uniform vec2 puppetBlink; uniform float puppetMouthOpen, puppetMouthWidth;
    varying float vPuppetSplit;   // 0 below the mouth line, 1 above; in between only on stretched triangles
    varying vec2 vPuppetMouth;    // undeformed position relative to the mouth center (head radii)

    float puppetFaceZ(float x, float y) {
      float c[10]; for (int i = 0; i < 10; i++) c[i] = puppetFaceFit[i];
      return c[0] + c[1]*x + c[2]*y + c[3]*x*x + c[4]*y*y + c[5]*x*y + c[6]*x*x*x + c[7]*y*y*y + c[8]*x*x*y + c[9]*x*y*y;
    }
    // Squash one eye toward its lower lid. Returns the new y.
    float puppetEye(vec2 l, vec4 E, float open) {
      float hx = 1.0 - smoothstep(1.0, 1.4, abs(l.x - E.x) / E.z);
      float yb = E.y - 0.85 * E.w, yt = E.y + 1.05 * E.w, yf = E.y + 2.1 * E.w;
      float s = mix(0.07, 1.0, open);
      float y = l.y;
      float ny = y;
      if (y >= yb && y <= yt) ny = yb + (y - yb) * s;
      else if (y > yt && y < yf) ny = y + (yt - yb) * (s - 1.0) * (1.0 - (y - yt) / (yf - yt));
      return mix(y, ny, hx);
    }
    vec3 puppetDeform(vec3 wp) {
      vec3 d = wp - puppetCenter;
      vec3 l = vec3(dot(d, puppetRight), dot(d, puppetUp), dot(d, puppetFwd)) / puppetRadius;
      // Only the face surface: near the fitted face z (not back hair, not the fringe floating in front).
      vPuppetMouth = l.xy - puppetMouth.xy;
      vPuppetSplit = step(0.0, vPuppetMouth.y);
      if (l.z < 0.15 || l.y < -0.95 || l.y > 0.25 || abs(l.x) > 0.95) { vPuppetMouth = vec2(9.0); return wp; }
      float dz = l.z - puppetFaceZ(l.x, l.y);
      float gate = (1.0 - smoothstep(0.11, 0.15, dz)) * smoothstep(-0.25, -0.15, dz);
      if (gate <= 0.0) { vPuppetMouth = vec2(9.0); return wp; }
      vec3 n = l;
      n.y = puppetEye(n.xy, puppetEyeA, puppetBlink.x);
      n.y = puppetEye(n.xy, puppetEyeB, puppetBlink.y);
      // Mouth: split at the mouth line, lower part drops, upper lifts; width scales about the center.
      vec2 m = l.xy - puppetMouth.xy;
      float reach = 1.0 - smoothstep(0.45, 1.0, length(m / vec2(puppetMouth.z * 2.2, puppetMouth.z * 1.3)));
      float split = smoothstep(-0.01, 0.01, m.y);
      vPuppetSplit = split;
      float amp = 0.085 * puppetMouthOpen;
      n.y += mix(-amp, 0.35 * amp, split) * reach;
      n.x += m.x * (puppetMouthWidth - 1.0) * reach;
      vec3 out_l = mix(l, n, gate);
      return puppetCenter + (out_l.x * puppetRight + out_l.y * puppetUp + out_l.z * puppetFwd) * puppetRadius;
    }
    vec3 puppetApply(vec3 objPos) {
      vec4 wp = modelMatrix * vec4(objPos, 1.0);
      wp.xyz = puppetDeform(wp.xyz);
      return (puppetModelInv * wp).xyz;
    }`;

  // Mouth interior: where the split stretched triangles open (split strictly between the lips),
  // within the mouth's width. Returns the interior color (sRGB) in rgb and its coverage in a.
  const FRAG = `
    uniform vec3 puppetMouth; uniform float puppetMouthOpen, puppetMouthWidth;
    varying float vPuppetSplit;
    varying vec2 vPuppetMouth;
    vec4 puppetMouthInterior() {
      float between = smoothstep(0.08, 0.2, vPuppetSplit) * (1.0 - smoothstep(0.8, 0.92, vPuppetSplit));
      float across = 1.0 - smoothstep(0.7, 1.0, abs(vPuppetMouth.x) / (puppetMouth.z * puppetMouthWidth * 1.15));
      float a = between * across * smoothstep(0.04, 0.2, puppetMouthOpen);
      vec3 c = mix(vec3(0.86, 0.44, 0.48), vec3(0.36, 0.08, 0.11), smoothstep(0.2, 0.55, vPuppetSplit));
      return vec4(c, a);
    }`;
  function patchSource(vs) {
    if (vs.includes('puppetApply')) return vs;
    vs = vs.replace('void main() {', GLSL + '\nvoid main() {');
    return vs.replace('#include <skinning_vertex>', '#include <skinning_vertex>\n  transformed = puppetApply(transformed);');
  }
  function patch(material, colorSpace = 'srgb') {
    if (material.isShaderMaterial) {
      Object.assign(material.uniforms, U);
      material.vertexShader = patchSource(material.vertexShader);
      // ShaderMaterials opt in by marking where their base color is final with: // PUPPET_MOUTH(base)
      const m = material.fragmentShader.match(/\/\/ PUPPET_MOUTH\((\w+)\)/);
      if (m && !material.fragmentShader.includes('puppetMouthInterior')) {
        material.fragmentShader = material.fragmentShader.replace('void main() {', FRAG + '\nvoid main() {')
          .replace(m[0], `{ vec4 pm = puppetMouthInterior(); ${m[1]} = mix(${m[1]}, pm.rgb, pm.a); }`);
      }
      material.needsUpdate = true;
      return material;
    }
    const prev = material.onBeforeCompile;
    material.onBeforeCompile = (shader, r) => {
      if (prev) prev(shader, r);
      Object.assign(shader.uniforms, U);
      shader.vertexShader = patchSource(shader.vertexShader);
      // Built-ins with a diffuse map: tint diffuseColor after the map is applied.
      if (shader.fragmentShader.includes('#include <map_fragment>')) {
        const col = colorSpace === 'linear' ? 'pow(pm.rgb, vec3(2.2))' : 'pm.rgb';
        shader.fragmentShader = shader.fragmentShader.replace('void main() {', FRAG + '\nvoid main() {')
          .replace('#include <map_fragment>', `#include <map_fragment>\n  { vec4 pm = puppetMouthInterior(); diffuseColor.rgb = mix(diffuseColor.rgb, ${col}, pm.a); }`);
      }
    };
    material.customProgramCacheKey = () => 'face-puppet';
    material.needsUpdate = true;
    return material;
  }

  // ---- look-at on neck + head, layered on the animation ----
  const base = new Map();
  const lookTarget = new THREE.Vector3();
  const look = { yaw: 0, pitch: 0 };
  let lookEnabled = true;
  function beginFrame() {
    for (const [b, q] of base) b.quaternion.copy(q);
  }
  const _q = new THREE.Quaternion(), _pq = new THREE.Quaternion(), _axis = new THREE.Vector3();
  function rotateWorld(b, axisWorld, angle) {
    // Rotate bone b about a world-space axis: local' = inv(parentWorld) * R * parentWorld * local.
    b.parent.getWorldQuaternion(_pq);
    _axis.copy(axisWorld).applyQuaternion(_pq.clone().invert());
    _q.setFromAxisAngle(_axis, angle);
    b.quaternion.premultiply(_q);
  }
  const c = new THREE.Vector3(), top = new THREE.Vector3(), front = new THREE.Vector3();
  function headFrame() {
    head.getWorldPosition(c); headTop.getWorldPosition(top); headFront.getWorldPosition(front);
    const center = c.clone().lerp(top, 0.5);
    const up = top.clone().sub(c).normalize();
    const fwd = front.clone().sub(center); fwd.sub(up.clone().multiplyScalar(fwd.dot(up))).normalize();
    return { center, up, fwd, right: new THREE.Vector3().crossVectors(up, fwd).normalize(), radius: c.distanceTo(top) * 0.62 };
  }
  function update(dt) {
    for (const b of [neck, head]) base.set(b, b.quaternion.clone());
    if (lookEnabled) {
      const f = headFrame();
      const d = lookTarget.clone().sub(f.center).normalize();
      const yaw = THREE.MathUtils.clamp(Math.atan2(d.dot(f.right), d.dot(f.fwd)), -0.7, 0.7);
      const pitch = THREE.MathUtils.clamp(Math.asin(THREE.MathUtils.clamp(d.dot(f.up), -1, 1)), -0.35, 0.35);
      const k = 1 - Math.exp(-dt * 6);
      look.yaw += (yaw - look.yaw) * k; look.pitch += (pitch - look.pitch) * k;
      for (const [b, w] of [[neck, 0.4], [head, 0.6]]) {
        b.updateWorldMatrix(true, false);
        const fr = headFrame();
        rotateWorld(b, fr.up, look.yaw * w);
        rotateWorld(b, fr.right, -look.pitch * w);
        b.updateWorldMatrix(false, true);
      }
    }
    mesh.updateMatrixWorld(true);
    const f = headFrame();
    U.puppetCenter.value.copy(f.center); U.puppetRadius.value = f.radius;
    U.puppetRight.value.copy(f.right); U.puppetUp.value.copy(f.up); U.puppetFwd.value.copy(f.fwd);
    U.puppetModelInv.value.copy(mesh.matrixWorld).invert();
  }

  return {
    patch, beginFrame, update, lookTarget,
    get lookEnabled() { return lookEnabled; }, set lookEnabled(v) { lookEnabled = v; },
    blink: { get 0() { return U.puppetBlink.value.x; }, set 0(v) { U.puppetBlink.value.x = v; },
             get 1() { return U.puppetBlink.value.y; }, set 1(v) { U.puppetBlink.value.y = v; } },
    get mouthOpen() { return U.puppetMouthOpen.value; }, set mouthOpen(v) { U.puppetMouthOpen.value = v; },
    get mouthWidth() { return U.puppetMouthWidth.value; }, set mouthWidth(v) { U.puppetMouthWidth.value = v; },
  };
}
if (typeof module !== 'undefined') module.exports = { createFacePuppet };
