// Hand-engineered 2.5D "Genshin-style" toon shader for char_run.glb (three.js r147).
//
// Features (all hand-built; colors come from the NAS palette P):
//   palette   albedo -> NAS learned palette remap (flat anime colors)
//   ramp      half-Lambert -> 2-band ramp with a soft edge; colored shadows (warm on skin, cool on cloth)
//   face SDF  face shadow driven by light direction in the head frame, not by normals: a clean
//             terminator that sweeps across the face as the light orbits (Genshin's face-shadow trick)
//   flatten   hair/head normals pulled toward a sphere around the head -> big flat shading shapes
//   angel     anisotropic (Kajiya-Kay) hair highlight with the tangent along the head's up axis, stepped:
//             a thin band that wraps around the head ("angel ring"), tinted green
//   rim       stepped fresnel rim on the lit side only
//   outline   inverted hull: back faces pushed out along smoothed normals at constant screen width,
//             colored as a darkened version of the surface underneath
//
// createGenshin(THREE, mesh, P) -> { material, outline, update(camera, renderer), setLightAzimuth(deg) }
function createGenshin(THREE, mesh, P) {
  const vec3s = (flat) => Array.from({ length: flat.length / 3 }, (_, i) => new THREE.Vector3(flat[3*i], flat[3*i+1], flat[3*i+2]));
  const map = mesh.material.map;

  // ---- head frame from the rig ----
  const bones = mesh.skeleton.bones;
  const bone = (suffix) => bones.find((b) => b.name.endsWith(suffix));
  const head = bone('Head'), headTop = bone('HeadTop_End'), headFront = bone('headfront');

  // ---- smoothed normals for the outline hull (glTF splits normals at UV seams, which cracks the hull) ----
  const geo = mesh.geometry;
  const pos = geo.attributes.position, nrm = geo.attributes.normal;
  const key = (i) => `${pos.getX(i).toFixed(5)},${pos.getY(i).toFixed(5)},${pos.getZ(i).toFixed(5)}`;
  const acc = new Map();
  for (let i = 0; i < pos.count; i++) {
    const k = key(i), a = acc.get(k) || [0, 0, 0];
    a[0] += nrm.getX(i); a[1] += nrm.getY(i); a[2] += nrm.getZ(i);
    acc.set(k, a);
  }
  const smooth = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const a = acc.get(key(i)), l = Math.hypot(a[0], a[1], a[2]) || 1;
    smooth.set([a[0] / l, a[1] / l, a[2] / l], i * 3);
  }
  geo.setAttribute('smoothNormal', new THREE.BufferAttribute(smooth, 3));

  const uniforms = {
    map: { value: map },
    palSrc: { value: vec3s(P.pal_src) }, palDst: { value: vec3s(P.pal_dst) },
    palTemp: { value: P.pal_temp }, palDetail: { value: P.pal_detail },
    lightDir: { value: new THREE.Vector3(-0.5, 0.6, 0.6).normalize() },  // world space
    headCenter: { value: new THREE.Vector3() }, headRadius: { value: 0.1 },
    headFwd: { value: new THREE.Vector3(0, 0, 1) }, headRight: { value: new THREE.Vector3(1, 0, 0) },
    headUp: { value: new THREE.Vector3(0, 1, 0) },
    // art direction
    rampThr: { value: 0.5 }, rampSoft: { value: 0.035 },
    skinShadow: { value: new THREE.Color(1.0, 0.80, 0.80) },
    clothShadow: { value: new THREE.Color(0.74, 0.84, 0.80) },
    deepShadow: { value: new THREE.Color(0.62, 0.70, 0.70) },
    angelColor: { value: new THREE.Color(0.55, 1.0, 0.45) },
    rimColor: { value: new THREE.Color(1.0, 1.0, 0.95) },
  };

  const vertexShader = `
    #include <common>
    #include <skinning_pars_vertex>
    varying vec2 vUv;
    varying vec3 vPosW, vNormalW;
    void main() {
      vUv = uv;
      #include <beginnormal_vertex>
      #include <skinbase_vertex>
      #include <skinnormal_vertex>
      #include <begin_vertex>
      #include <skinning_vertex>
      vec4 worldPos = modelMatrix * vec4(transformed, 1.0);
      vPosW = worldPos.xyz;
      vNormalW = normalize(mat3(modelMatrix) * objectNormal);
      gl_Position = projectionMatrix * viewMatrix * worldPos;
    }`;

  const fragmentShader = `
    #define K ${P.pal_src.length / 3}
    uniform sampler2D map;
    uniform vec3 palSrc[K], palDst[K]; uniform float palTemp, palDetail;
    uniform vec3 lightDir, headCenter, headFwd, headRight, headUp; uniform float headRadius;
    uniform float rampThr, rampSoft;
    uniform vec3 skinShadow, clothShadow, deepShadow, angelColor, rimColor;
    varying vec2 vUv;
    varying vec3 vPosW, vNormalW;

    vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
    vec3 palette(vec3 a) {
      float dmin = 1e9;
      for (int k = 0; k < K; k++) { vec3 d = a - palSrc[k]; dmin = min(dmin, dot(d, d)); }
      vec3 src = vec3(0.0), dst = vec3(0.0); float ws = 0.0;
      for (int k = 0; k < K; k++) {
        vec3 d = a - palSrc[k]; float w = exp(-(dot(d, d) - dmin) / palTemp);
        src += w * palSrc[k]; dst += w * palDst[k]; ws += w;
      }
      return clamp(dst / ws + palDetail * (a - src / ws), 0.0, 1.0);
    }

    void main() {
      vec3 raw = texture2D(map, vUv).rgb;           // texture bytes = sRGB
      vec3 base = palette(raw);                     // NAS palette, sRGB
      vec3 n = normalize(vNormalW);
      vec3 V = normalize(cameraPosition - vPosW);
      vec3 L = normalize(lightDir);

      // Region masks from position + color (no material IDs in this asset).
      vec3 toHead = vPosW - headCenter;
      float headDist = length(toHead) / headRadius;
      float inHead = 1.0 - smoothstep(1.05, 1.35, headDist);
      float skin = smoothstep(0.06, 0.12, raw.r - raw.b) * smoothstep(0.55, 0.7, raw.r) * step(raw.b, raw.g + 0.02);
      float hair = inHead * (1.0 - smoothstep(0.25, 0.4, dot(raw, vec3(0.299, 0.587, 0.114))));
      float face = inHead * skin * smoothstep(0.0, 0.35, dot(normalize(toHead), headFwd));

      // Flatten: head normals pulled toward the head sphere (big, clean shading shapes on the hair).
      vec3 sphereN = normalize(toHead);
      n = normalize(mix(n, sphereN, 0.6 * hair));

      // Body ramp: half-Lambert -> two bands.
      float hl = dot(n, L) * 0.5 + 0.5;
      float lit = smoothstep(rampThr - rampSoft, rampThr + rampSoft, hl);
      float deep = 1.0 - smoothstep(0.18, 0.24, hl);

      // Face SDF: in the head frame, a straight terminator that sweeps with the light's angle.
      vec3 Lh = normalize(L - headUp * dot(L, headUp) + 1e-4);
      float lf = dot(Lh, headFwd), lr = dot(Lh, headRight);
      float x = dot(toHead, headRight) / headRadius * (lr >= 0.0 ? 1.0 : -1.0);
      float faceLit = smoothstep(-lf - 0.06, -lf + 0.06, x * 1.6);
      lit = mix(lit, faceLit, face);
      deep *= 1.0 - face;

      vec3 shadowTint = mix(clothShadow, skinShadow, skin);
      vec3 col = base * mix(shadowTint, vec3(1.0), lit);
      col *= mix(vec3(1.0), deepShadow, deep * (1.0 - lit));

      // Angel ring: Kajiya-Kay with hair strands running along head-up -> a band that wraps the head.
      vec3 H = normalize(L + V);
      vec3 T = normalize(headUp - sphereN * dot(sphereN, headUp) + 1e-4);
      float sinTH = sqrt(max(1.0 - pow(dot(T, H), 2.0), 0.0));
      float upper = smoothstep(0.05, 0.3, dot(toHead, headUp) / headRadius);  // crown, not the fringe tips
      float frontLit = smoothstep(-0.2, 0.3, dot(normalize(L - headUp * dot(L, headUp) + 1e-4), headFwd));
      float ring = smoothstep(0.986, 0.993, sinTH) * hair * upper * lit * frontLit;
      col = mix(col, angelColor, ring * 0.6);

      // Rim: stepped fresnel on the lit side.
      float rim = smoothstep(0.66, 0.70, 1.0 - max(dot(normalize(vNormalW), V), 0.0)) * smoothstep(0.0, 0.2, dot(normalize(vNormalW), L));
      col = mix(col, rimColor * base * 1.3, rim * 0.5);

      gl_FragColor = vec4(srgbToLinear(clamp(col, 0.0, 1.0)), 1.0);  // renderer encodes back to sRGB
      #include <encodings_fragment>
    }`;

  const material = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader });

  // ---- inverted-hull outline ----
  const outlineUniforms = {
    map: { value: map },
    width: { value: 1.6 }, resolution: { value: new THREE.Vector2(1, 1) },
    tint: { value: new THREE.Color(0.30, 0.42, 0.32) },
  };
  const outlineMat = new THREE.ShaderMaterial({
    uniforms: outlineUniforms, side: THREE.BackSide,
    vertexShader: `
      #include <common>
      #include <skinning_pars_vertex>
      attribute vec3 smoothNormal;
      uniform float width; uniform vec2 resolution;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vec3 objectNormal = smoothNormal;
        #include <skinbase_vertex>
        #include <skinnormal_vertex>
        #include <begin_vertex>
        #include <skinning_vertex>
        vec4 clip = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
        vec3 nView = normalize(normalMatrix * objectNormal);
        vec2 dir = normalize(nView.xy + 1e-5);
        clip.xy += dir * width * 2.0 / resolution * clip.w;  // constant width in pixels
        gl_Position = clip;
      }`,
    fragmentShader: `
      uniform sampler2D map; uniform vec3 tint;
      varying vec2 vUv;
      void main() {
        vec3 c = texture2D(map, vUv).rgb * tint;   // darkened surface color, sRGB
        gl_FragColor = vec4(pow(c, vec3(2.2)), 1.0);
        #include <encodings_fragment>
      }`,
  });
  const outline = new THREE.SkinnedMesh(geo, outlineMat);
  outline.bind(mesh.skeleton, mesh.bindMatrix);
  outline.frustumCulled = false;
  mesh.parent.add(outline);
  outline.visible = false;

  const c = new THREE.Vector3(), top = new THREE.Vector3(), front = new THREE.Vector3();
  function update(camera, renderer) {
    head.getWorldPosition(c);
    headTop.getWorldPosition(top);
    headFront.getWorldPosition(front);
    const center = c.clone().lerp(top, 0.5);
    uniforms.headCenter.value.copy(center);
    uniforms.headRadius.value = c.distanceTo(top) * 0.62;
    const up = top.clone().sub(c).normalize();
    const fwd = front.clone().sub(center);
    fwd.sub(up.clone().multiplyScalar(fwd.dot(up))).normalize();
    uniforms.headUp.value.copy(up);
    uniforms.headFwd.value.copy(fwd);
    uniforms.headRight.value.copy(new THREE.Vector3().crossVectors(up, fwd).normalize());
    const s = renderer.getDrawingBufferSize(new THREE.Vector2());
    outlineUniforms.resolution.value.copy(s);
    outlineUniforms.width.value = 1.6 * renderer.getPixelRatio();
  }
  function setLightAzimuth(deg) {
    const a = THREE.MathUtils.degToRad(deg);
    uniforms.lightDir.value.set(Math.sin(a) * 0.8, 0.6, Math.cos(a) * 0.8).normalize();
  }
  setLightAzimuth(-35);
  return { material, outline, update, setLightAzimuth };
}
if (typeof module !== 'undefined') module.exports = { createGenshin };
