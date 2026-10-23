import { useEffect, useRef } from "react";
import * as THREE from "three";

// An illustration, not data: one long RNA strand - a continuous backbone with its bases -
// crossing the hero at a steep slant and running off the screen. It spins slowly about
// its own axis. Modified bases are in the accent colour and glow softly under the cursor;
// dragging (mouse or pen) turns the strand by hand, and it eases back on release.
// Kept light: one tube and two instanced meshes, paused off screen or in a hidden tab,
// still for readers who ask for reduced motion.

const LENGTH = 64;                 // world units: longer than any screen diagonal
const PER_UNIT = 3.2;              // nucleotides per unit length
const N = Math.round(LENGTH * PER_UNIT);
const RADIUS = 1.15;
const TURNS = LENGTH / 3.6;
const TUBE = 0.14;
const DISTANCE = 30;               // camera distance: sets the strand's on-screen thickness

// Modified bases: about 6%, scattered irregularly but the same on every load (an integer
// hash of the index, not Math.random), so the picture never changes between visits.
const MODIFIED = new Set(Array.from({ length: N }, (_, i) => i)
  .filter((i) => (Math.imul(i + 7, 2654435761) >>> 0) % 100 < 6));

function tokens() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => new THREE.Color(css.getPropertyValue(name).trim() || "#888");
  return { backbone: get("--strand-backbone"), base: get("--strand-base"), tag: get("--accent") };
}

const helix = (t) => {
  const theta = t * TURNS * Math.PI * 2;
  return new THREE.Vector3((t - 0.5) * LENGTH, RADIUS * Math.cos(theta), RADIUS * Math.sin(theta));
};

// Where the strand sits: right of the text on wide screens, above it on tall ones.
function placement(aspect) {
  return aspect >= 1
    ? { slant: -0.78, ndc: [0.42, -0.02] }
    : { slant: -0.3, ndc: [0.05, 0.68] };
}

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export default function RnaStrand({ className }) {
  const host = useRef(null);
  const calloutRef = useRef(null), ringRef = useRef(null), lineRef = useRef(null), labelRef = useRef(null);

  useEffect(() => {
    const el = host.current;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const canvas = renderer.domElement;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);
    camera.position.set(0, 0, DISTANCE);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(3, 6, 9);
    scene.add(sun);

    // anchor (screen placement) > tilt (slant, plus any drag) > spin (about the strand's axis)
    const anchor = new THREE.Group();
    const tilt = new THREE.Group();
    const spin = new THREE.Group();
    anchor.add(tilt);
    tilt.add(spin);
    scene.add(anchor);

    const c = tokens();
    const backboneMat = new THREE.MeshStandardMaterial({ color: c.backbone, roughness: 0.4, metalness: 0.08 });
    const baseMat = new THREE.MeshStandardMaterial({ color: c.base, roughness: 0.5, metalness: 0.02 });
    const tagMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45, emissive: c.tag, emissiveIntensity: 0.15 });

    const curve = new THREE.CatmullRomCurve3(Array.from({ length: 1201 }, (_, i) => helix(i / 1200)));
    const tubeGeo = new THREE.TubeGeometry(curve, 2400, TUBE, 10, false);
    spin.add(new THREE.Mesh(tubeGeo, backboneMat));

    const baseLen = RADIUS * 0.82;
    const baseGeo = new THREE.CapsuleGeometry(0.1, baseLen, 4, 8);
    const plain = new THREE.InstancedMesh(baseGeo, baseMat, N - MODIFIED.size);
    const tags = new THREE.InstancedMesh(baseGeo, tagMat, MODIFIED.size);
    const tagCentres = [];
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), one = new THREE.Vector3(1, 1, 1);
    // Tagged bases are thicker: easier to see and to hover (the raycast uses the same mesh).
    const fat = new THREE.Vector3(1.45, 1.08, 1.45);
    let a = 0, b = 0;
    for (let i = 0; i < N; i++) {
      const p = helix((i + 0.5) / N);
      const inward = new THREE.Vector3(0, -p.y, -p.z).normalize();
      const mid = p.clone().addScaledVector(inward, baseLen / 2 + TUBE * 0.6);
      q.setFromUnitVectors(up, inward);
      m.compose(mid, q, MODIFIED.has(i) ? fat : one);
      if (MODIFIED.has(i)) {
        tags.setMatrixAt(b++, m);
        tagCentres.push(mid);
      } else plain.setMatrixAt(a++, m);
    }
    const paintTags = (k, lit = -1) => {
      for (let i = 0; i < tagCentres.length; i++)
        tags.setColorAt(i, i === lit ? k.tag.clone().lerp(new THREE.Color(0xffffff), 0.3) : k.tag);
      tags.instanceColor.needsUpdate = true;
    };
    paintTags(c);
    spin.add(plain, tags);

    // A soft halo for the base under the cursor.
    const haloTex = glowTexture();
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex, color: c.tag, transparent: true, opacity: 0, depthWrite: false }));
    halo.scale.setScalar(2.9);
    spin.add(halo);

    let running = false;
    const restyle = () => {
      const k = tokens();
      backboneMat.color.copy(k.backbone);
      baseMat.color.copy(k.base);
      tagMat.emissive.copy(k.tag);
      halo.material.color.copy(k.tag);
      paintTags(k, hovered);
      if (!running) draw();
    };
    const themeObserver = new MutationObserver(restyle);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    scheme.addEventListener("change", restyle);

    // Rest orientation; a drag adds a rotation that eases back to it.
    const rest = new THREE.Quaternion();
    const resize = () => {
      const { width, height } = el.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      const { slant, ndc } = placement(camera.aspect);
      rest.setFromEuler(new THREE.Euler(0, 0.25, slant));
      tilt.quaternion.copy(rest);
      const halfH = DISTANCE * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      anchor.position.set(ndc[0] * halfH * camera.aspect, ndc[1] * halfH, 0);
      if (!running) draw();
    };

    // Hover: which modified base is under the cursor.
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let hovered = -1, glow = 0;
    const pick = (e) => {
      const r = canvas.getBoundingClientRect();
      pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(tags, false)[0];
      const next = hit ? hit.instanceId : -1;
      if (next !== hovered) {
        hovered = next;
        paintTags(tokens(), hovered);
        canvas.style.cursor = hovered >= 0 ? "pointer" : dragging ? "grabbing" : "grab";
        if (hovered >= 0) halo.position.copy(tagCentres[hovered]);
        if (!running) draw();
      }
    };

    // Drag: mouse or pen turns the strand; touch is left to scrolling.
    let dragging = false, lastX = 0, lastY = 0;
    const onDown = (e) => {
      if (e.pointerType === "touch") return;
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
    };
    const onMove = (e) => {
      if (dragging) {
        const dx = (e.clientX - lastX) * 0.006, dy = (e.clientY - lastY) * 0.006;
        lastX = e.clientX;
        lastY = e.clientY;
        const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(dy, dx, 0));
        tilt.quaternion.premultiply(turn);
        if (!running) draw();
      } else if (e.pointerType !== "touch") pick(e);
    };
    const onUp = () => {
      dragging = false;
      canvas.style.cursor = hovered >= 0 ? "pointer" : "grab";
    };
    const onLeave = () => {
      if (!dragging && hovered >= 0) {
        hovered = -1;
        paintTags(tokens());
        if (!running) draw();
      }
    };
    canvas.style.cursor = "grab";
    canvas.style.touchAction = "pan-y";
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);

    // Callout: a ring that follows the hovered base on screen, joined by a line to a fixed
    // label saying what the orange bases are. Updated after every render, so it tracks the
    // base as the strand turns.
    const v = new THREE.Vector3();
    const callout = () => {
      const box = calloutRef.current, ring = ringRef.current, line = lineRef.current, label = labelRef.current;
      if (!box || !ring || !line || !label) return;
      if (hovered < 0) { box.classList.remove("on"); return; }
      const w = canvas.clientWidth, h = canvas.clientHeight;
      v.copy(tagCentres[hovered]).applyMatrix4(spin.matrixWorld).project(camera);
      const px = ((v.x + 1) / 2) * w, py = ((1 - v.y) / 2) * h;
      const host = el.getBoundingClientRect(), lb = label.getBoundingClientRect();
      // The line meets the label at its nearer side, at mid-height.
      const lx = (px < lb.left - host.left ? lb.left : lb.right) - host.left, ly = lb.top - host.top + lb.height / 2;
      ring.setAttribute("cx", px.toFixed(1));
      ring.setAttribute("cy", py.toFixed(1));
      const dx = lx - px, dy = ly - py, d = Math.hypot(dx, dy) || 1, r = 15;
      line.setAttribute("x1", (px + (dx / d) * r).toFixed(1));
      line.setAttribute("y1", (py + (dy / d) * r).toFixed(1));
      line.setAttribute("x2", lx.toFixed(1));
      line.setAttribute("y2", ly.toFixed(1));
      box.classList.add("on");
    };

    let angle = 0.6, last = performance.now(), frame = 0, visible = true;
    const draw = () => {
      spin.rotation.x = angle;
      halo.material.opacity = 0.55 * glow;
      renderer.render(scene, camera);
      callout();
    };
    const step = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      angle += dt * 0.3;
      glow += ((hovered >= 0 ? 1 : 0) - glow) * 0.15;
      if (!dragging) tilt.quaternion.slerp(rest, 0.04);
      draw();
      frame = requestAnimationFrame(step);
    };
    const update = () => {
      const should = visible && !document.hidden && !reduced;
      if (should && !running) {
        running = true;
        last = performance.now();
        frame = requestAnimationFrame(step);
      } else if (!should && running) {
        running = false;
        cancelAnimationFrame(frame);
      }
    };
    const io = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; update(); });
    io.observe(el);
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    document.addEventListener("visibilitychange", update);
    resize();
    update();

    return () => {
      running = false;
      cancelAnimationFrame(frame);
      io.disconnect();
      ro.disconnect();
      themeObserver.disconnect();
      scheme.removeEventListener("change", restyle);
      document.removeEventListener("visibilitychange", update);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      [tubeGeo, baseGeo, backboneMat, baseMat, tagMat, halo.material, haloTex].forEach((x) => x.dispose());
      plain.dispose();
      tags.dispose();
      renderer.dispose();
      canvas.remove();
    };
  }, []);

  return (
    <div ref={host} className={className} aria-hidden="true">
      <div className="callout" ref={calloutRef}>
        <svg className="callout-lines">
          <line ref={lineRef} />
          <circle ref={ringRef} r="18" />
        </svg>
        <div className="callout-label" ref={labelRef}>
          <strong>m6A site</strong>
          <span>N6-methyladenosine: an adenosine (A) carrying a methyl group. The most common internal modification of mRNA.</span>
        </div>
      </div>
    </div>
  );
}
