
import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment } from '@react-three/drei';
import * as THREE from 'three';
import { BreathPhase, ColorTheme, QualityTier } from '../types';

type Props = {
  phase: BreathPhase;
  theme: ColorTheme;
  quality: QualityTier;
  reduceMotion: boolean;
  isActive: boolean;
  progressRef: React.MutableRefObject<number>;
  entropyRef?: React.MutableRefObject<number>;
};

const THEMES: Record<ColorTheme, { deep: THREE.Color; mid: THREE.Color; glow: THREE.Color; accent: THREE.Color }> = {
  warm: {
    deep: new THREE.Color('#2b0505'),
    mid: new THREE.Color('#a3341e'),
    glow: new THREE.Color('#ffd39a'),
    accent: new THREE.Color('#ff8f6a'),
  },
  cool: {
    deep: new THREE.Color('#00121a'),
    mid: new THREE.Color('#0b4f6e'),
    glow: new THREE.Color('#7afff3'),
    accent: new THREE.Color('#1ad3ff'),
  },
  neutral: {
    deep: new THREE.Color('#0d0d12'),
    mid: new THREE.Color('#5e5e6e'),
    glow: new THREE.Color('#ffffff'),
    accent: new THREE.Color('#c8d6e5'),
  },
};

function resolveTier(q: QualityTier) {
  const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
  const cores = typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency || 4) : 4;
  const auto = q === 'auto';
  const isLow = q === 'low' || (auto && cores <= 4);
  const isHigh = q === 'high' || (auto && cores >= 8);

  if (isLow) return { dpr: Math.min(dpr, 1.25), seg: 28, halo: false, ring: false };
  if (isHigh) return { dpr: Math.min(dpr, 2), seg: 64, halo: true, ring: true };
  return { dpr: Math.min(dpr, 1.5), seg: 44, halo: true, ring: false };
}

const CORE_VERT = `
varying vec3 vPos;
varying vec3 vNormal;
void main() {
  vPos = position;
  vNormal = normalMatrix * normal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CORE_FRAG = `
precision highp float;
varying vec3 vPos;
varying vec3 vNormal;
uniform float uTime;
uniform float uBreath;
uniform float uEntropy;
uniform vec3 uDeep;
uniform vec3 uMid;
uniform vec3 uGlow;
uniform vec3 uAccent;

float hash(vec3 p){
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 p){
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f*f*(3.0-2.0*f);
  float n000 = hash(i + vec3(0,0,0));
  float n100 = hash(i + vec3(1,0,0));
  float n010 = hash(i + vec3(0,1,0));
  float n110 = hash(i + vec3(1,1,0));
  float n001 = hash(i + vec3(0,0,1));
  float n101 = hash(i + vec3(1,0,1));
  float n011 = hash(i + vec3(0,1,1));
  float n111 = hash(i + vec3(1,1,1));
  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);
  float nxy0 = mix(nx00, nx10, f.y);
  float nxy1 = mix(nx01, nx11, f.y);
  return mix(nxy0, nxy1, f.z);
}

float fbm(vec3 p){
  float v = 0.0;
  float a = 0.55;
  for(int i=0;i<4;i++){
    v += a * noise(p);
    p *= 2.02;
    a *= 0.5;
  }
  return v;
}

void main(){
  vec3 n = normalize(vNormal);
  vec3 p = vPos * 1.2;
  float t = uTime * 0.22;
  float f = fbm(p + vec3(0.0, t, t*0.7));
  float fil = smoothstep(0.45, 0.85, f);
  vec3 base = mix(uDeep, uMid, fil);
  vec3 glow = mix(base, uGlow, clamp(uBreath, 0.0, 1.0) * 0.75);
  float shimmer = fbm(p * 2.2 + vec3(t*2.0)) * 0.5 + 0.5;
  glow = mix(glow, uAccent, shimmer * clamp(uEntropy, 0.0, 1.0) * 0.18);
  float fres = pow(1.0 - abs(dot(n, vec3(0.0,0.0,1.0))), 2.5);
  float energy = (0.35 + 0.95 * uBreath) * (0.65 + 0.35 * fil);
  vec3 col = glow * energy + uGlow * fres * (0.35 + 0.5 * uBreath);
  float alpha = 0.25 + 0.55 * uBreath + 0.15 * fil;
  gl_FragColor = vec4(col, alpha);
}
`;

function ZenOrb(props: Props) {
  const { phase, theme, quality, reduceMotion, isActive, progressRef, entropyRef } = props;
  const tier = useMemo(() => resolveTier(quality), [quality]);
  const colors = useMemo(() => THEMES[theme] ?? THEMES.neutral, [theme]);

  const group = useRef<THREE.Group>(null);
  const shellMat = useRef<THREE.MeshPhysicalMaterial>(null);
  const coreMat = useRef<THREE.ShaderMaterial>(null);

  const breathRef = useRef(0);
  const entropySmoothRef = useRef(0);

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const motion = reduceMotion ? 0.45 : 1.0;
    const p = THREE.MathUtils.clamp(progressRef.current, 0, 1);

    let targetBreath = 0;
    if (isActive) {
      if (phase === 'inhale') targetBreath = p;
      else if (phase === 'holdIn') targetBreath = 1;
      else if (phase === 'exhale') targetBreath = 1 - p;
      else targetBreath = 0;
    } else {
      targetBreath = (Math.sin(time * 0.7) * 0.5 + 0.5) * 0.35;
    }

    const targetEntropy = entropyRef?.current ?? 0;
    breathRef.current = THREE.MathUtils.lerp(breathRef.current, targetBreath, 1 - Math.pow(0.001, delta * 3.2));
    entropySmoothRef.current = THREE.MathUtils.lerp(entropySmoothRef.current, targetEntropy, 1 - Math.pow(0.001, delta * 2.0));

    const breath = breathRef.current;
    const entropy = THREE.MathUtils.clamp(entropySmoothRef.current, 0, 1);

    const baseScale = 1.35;
    const pulse = 0.14 * breath + 0.02 * Math.sin(time * 2.2);
    group.current?.scale.setScalar(baseScale + pulse);

    if (group.current) {
      group.current.rotation.y += delta * 0.18 * motion;
      group.current.rotation.x = Math.sin(time * 0.15) * 0.08 * motion;
    }

    if (shellMat.current) {
      shellMat.current.roughness = THREE.MathUtils.lerp(0.55, 0.18, breath);
      shellMat.current.clearcoat = THREE.MathUtils.lerp(0.35, 1.0, breath);
      shellMat.current.clearcoatRoughness = THREE.MathUtils.lerp(0.5, 0.1, breath);
      shellMat.current.transmission = THREE.MathUtils.lerp(0.45, 0.92, breath);
      shellMat.current.ior = 1.3;
      shellMat.current.thickness = THREE.MathUtils.lerp(0.3, 0.7, breath);
      shellMat.current.attenuationColor.copy(colors.deep);
      shellMat.current.attenuationDistance = THREE.MathUtils.lerp(1.8, 0.9, breath);
    }

    if (coreMat.current) {
      coreMat.current.uniforms.uTime.value = time;
      coreMat.current.uniforms.uBreath.value = breath;
      coreMat.current.uniforms.uEntropy.value = entropy;
      coreMat.current.uniforms.uDeep.value.copy(colors.deep);
      coreMat.current.uniforms.uMid.value.copy(colors.mid);
      coreMat.current.uniforms.uGlow.value.copy(colors.glow);
      coreMat.current.uniforms.uAccent.value.copy(colors.accent);
    }
  });

  return (
    <group ref={group}>
      <mesh>
        <sphereGeometry args={[1.0, tier.seg, tier.seg]} />
        <meshPhysicalMaterial
          ref={shellMat}
          color={colors.mid}
          metalness={0.0}
          roughness={0.35}
          transmission={0.8}
          thickness={0.5}
          ior={1.3}
          clearcoat={0.8}
          clearcoatRoughness={0.15}
          envMapIntensity={1.1}
          transparent
        />
      </mesh>

      <mesh>
        <sphereGeometry args={[0.72, Math.max(18, Math.floor(tier.seg * 0.6)), Math.max(18, Math.floor(tier.seg * 0.6))]} />
        <shaderMaterial
          ref={coreMat}
          vertexShader={CORE_VERT}
          fragmentShader={CORE_FRAG}
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          uniforms={{
            uTime: { value: 0 },
            uBreath: { value: 0 },
            uEntropy: { value: 0 },
            uDeep: { value: colors.deep.clone() },
            uMid: { value: colors.mid.clone() },
            uGlow: { value: colors.glow.clone() },
            uAccent: { value: colors.accent.clone() },
          }}
        />
      </mesh>

      {tier.halo && (
        <mesh>
          <sphereGeometry args={[1.08, Math.max(18, Math.floor(tier.seg * 0.5)), Math.max(18, Math.floor(tier.seg * 0.5))]} />
          <meshBasicMaterial
            color={colors.glow}
            transparent
            opacity={0.08}
            blending={THREE.AdditiveBlending}
            side={THREE.BackSide}
            depthWrite={false}
          />
        </mesh>
      )}

      {tier.ring && (
        <mesh rotation={[Math.PI / 2.2, 0, 0]}>
          <torusGeometry args={[1.15, 0.015, 10, 120]} />
          <meshBasicMaterial
            color={colors.accent}
            transparent
            opacity={0.18}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
}

export default function OrbBreathVizZenSciFi(props: Props) {
  const tier = useMemo(() => resolveTier(props.quality), [props.quality]);
  
  return (
    <Canvas
      dpr={tier.dpr}
      camera={{ position: [0, 0, 4.8], fov: 45 }}
      gl={{ antialias: tier.dpr > 1.1, alpha: true, powerPreference: 'high-performance' }}
    >
      {tier.seg >= 44 && <Environment preset="city" />}
      <ambientLight intensity={0.55} />
      <pointLight position={[3, 3, 4]} intensity={1.2} />
      <ZenOrb {...props} />
    </Canvas>
  );
}
