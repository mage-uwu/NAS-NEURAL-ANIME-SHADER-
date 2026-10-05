// NAS cel shader compositor: GLSL mirror of nas/shader.py (CelShader.forward).
// createNASMaterial(THREE, P) -> ShaderMaterial for a full-screen quad.
// P is nas/out/params.json. Set uniforms tAlbedo/tNormal/tDepth (RGBA8 G-buffers) and texel (1/size).
// Output is premultiplied sRGB; draw with the material's blending over the background.
function createNASMaterial(THREE, P) {
  const vec3s = (flat) => Array.from({ length: flat.length / 3 }, (_, i) => new THREE.Vector3(flat[3*i], flat[3*i+1], flat[3*i+2]));
  return new THREE.ShaderMaterial({
    transparent: true, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
    uniforms: {
      tAlbedo: { value: null }, tNormal: { value: null }, tDepth: { value: null },
      texel: { value: new THREE.Vector2() },
      palSrc: { value: vec3s(P.pal_src) }, palDst: { value: vec3s(P.pal_dst) },
      palTemp: { value: P.pal_temp }, palDetail: { value: P.pal_detail },
      lightDir: { value: new THREE.Vector3(...P.light) },
      celThr: { value: P.cel_thr }, celSoft: { value: P.cel_soft }, shadowTint: { value: new THREE.Vector3(...P.shadow_tint) },
      rimThr: { value: P.rim_thr }, rimColor: { value: new THREE.Vector3(...P.rim_color) },
      lineW: { value: new THREE.Vector4(...P.line_w) }, lineThr: { value: P.line_thr }, lineSoft: { value: P.line_soft },
      lineColor: { value: new THREE.Vector3(...P.line_color) }, lineOpacity: { value: P.line_opacity },
    },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: `
      precision highp float;
      varying vec2 vUv;
      uniform sampler2D tAlbedo, tNormal, tDepth;
      uniform vec2 texel;
      #define K ${P.pal_src.length / 3}
      uniform vec3 palSrc[K], palDst[K]; uniform float palTemp, palDetail;
      uniform vec3 lightDir; uniform float celThr, celSoft; uniform vec3 shadowTint;
      uniform float rimThr; uniform vec3 rimColor;
      uniform vec4 lineW; uniform float lineThr, lineSoft, lineOpacity; uniform vec3 lineColor;

      float sig(float x) { return 1.0 / (1.0 + exp(-x)); }
      // Soft-assign albedo to K source anchors, output the learned target colors (+ detail residual).
      vec3 palette(vec3 a) {
        float dmin = 1e9;
        for (int k = 0; k < K; k++) { vec3 d = a - palSrc[k]; dmin = min(dmin, dot(d, d)); }
        vec3 src = vec3(0.0), dst = vec3(0.0); float wsum = 0.0;
        for (int k = 0; k < K; k++) {
          vec3 d = a - palSrc[k];
          float w = exp(-(dot(d, d) - dmin) / palTemp);  // shifted by dmin for numerical safety
          src += w * palSrc[k]; dst += w * palDst[k]; wsum += w;
        }
        return clamp(dst / wsum + palDetail * (a - src / wsum), 0.0, 1.0);
      }
      float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
      // Sobel magnitude per channel group, from the same 3x3 neighborhood.
      vec4 edges() {
        float dx[4], dy[4];
        for (int k = 0; k < 4; k++) { dx[k] = 0.0; dy[k] = 0.0; }
        vec3 nx = vec3(0.0), ny = vec3(0.0);
        for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
          vec2 uv = clamp(vUv + vec2(float(x), float(y)) * texel, 0.5 * texel, 1.0 - 0.5 * texel);
          float wx = float(x) * (y == 0 ? 2.0 : 1.0), wy = float(y) * (x == 0 ? 2.0 : 1.0);
          vec4 a = texture2D(tAlbedo, uv);
          vec3 n = texture2D(tNormal, uv).rgb;
          float d = texture2D(tDepth, uv).r;
          float l = luma(a.rgb);
          dx[0] += wx * d; dy[0] += wy * d;
          nx += wx * n; ny += wy * n;
          dx[2] += wx * l; dy[2] += wy * l;
          dx[3] += wx * a.a; dy[3] += wy * a.a;
        }
        float eN = sqrt(nx.r*nx.r + ny.r*ny.r + 1e-8) + sqrt(nx.g*nx.g + ny.g*ny.g + 1e-8) + sqrt(nx.b*nx.b + ny.b*ny.b + 1e-8);
        return vec4(sqrt(dx[0]*dx[0] + dy[0]*dy[0] + 1e-8), eN,
                    sqrt(dx[2]*dx[2] + dy[2]*dy[2] + 1e-8), sqrt(dx[3]*dx[3] + dy[3]*dy[3] + 1e-8));
      }
      void main() {
        vec4 alb = texture2D(tAlbedo, vUv);
        vec4 e = edges();
        float line = sig((dot(e, lineW) - lineThr) / lineSoft) * lineOpacity;
        float cover = alb.a;  // as in training: shader output composited by the albedo coverage
        if (cover < 0.001) { gl_FragColor = vec4(0.0); return; }

        vec3 base = palette(alb.rgb);
        vec3 n = normalize(texture2D(tNormal, vUv).rgb * 2.0 - 1.0);
        float lit = sig((dot(n, lightDir) - celThr) / celSoft);
        vec3 col = base * (shadowTint + (1.0 - shadowTint) * lit);
        col += sig((1.0 - n.z - rimThr) / 0.05) * lit * rimColor;
        col = mix(col, lineColor, line);
        // Shader math is in sRGB, like training. A raw ShaderMaterial gets no output encoding,
        // so these values land in the (sRGB) framebuffer as-is.
        col = clamp(col, 0.0, 1.0);
        gl_FragColor = vec4(col * cover, cover);
      }`,
  });
}
if (typeof module !== 'undefined') module.exports = { createNASMaterial };
