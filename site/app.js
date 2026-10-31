// render-engine oracle report. Everything on the page comes from the oracle's own output:
//   oracle/report.json  the CI run on lavapipe that built this site (FLIP gate, images)
//   gpu/report.json     the latest committed local run on the RTX 3060 Ti (GPU-time gate)
//   build.json          commit and workflow run that produced the site
const REPO = 'https://github.com/moritzbrantner/render-engine';

const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const fmt = v => (v == null ? '–' : v === 0 ? '0' : v < 0.001 ? v.toExponential(1) : v.toFixed(4));
const ms = v => (v == null ? '–' : `${v.toFixed(3)} ms`);
const g = v => +v.toPrecision(3);

async function json(path) {
  try {
    const r = await fetch(path, { cache: 'no-cache' });
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function load() {
  const [ci, gpu, build] = await Promise.all([json('oracle/report.json'), json('gpu/report.json'), json('build.json')]);
  return { ci, gpu, build };
}

// Theme: follows the system until the visitor picks one; the pick is remembered per browser.
export function theme() {
  const root = document.documentElement;
  const button = document.querySelector('.theme');
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch {}
  if (saved) root.dataset.theme = saved;
  const current = () => root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const label = () => { button.textContent = current() === 'dark' ? 'Light theme' : 'Dark theme'; };
  label();
  button.addEventListener('click', () => {
    root.dataset.theme = current() === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch {}
    label();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', label);
}

const prLinks = f => f.prs.map(n => `<a href="${REPO}/pull/${n}">PR #${n}</a>`).join('');
const decisionLink = f => `<a href="${REPO}/blob/main/docs/technical-decisions.md#${esc(f.decision)}">Decision record</a>`;
const featureOf = (report, s) => report.features.find(f => f.id === s.feature) || { id: s.feature, label: s.feature, prs: [], decision: '' };

// A FLIP value on a log scale from a thousandth of the gate to twice the gate; the tick is the
// gate. Passing scenarios sit far below it, so a linear scale would show them all as empty.
function scale(label, value, max) {
  const lo = Math.log10(max / 1000);
  const hi = Math.log10(max * 2);
  const at = v => Math.max(0, Math.min(1, (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo))) * 100;
  const w = at(value);
  return `<div class="scale"><span class="k">${label}</span>
    <span class="track" title="${label} ${value}; gate ${max} (log scale)"><span class="fill" style="width:${w}%;--full:${w > 0 ? (100 / w) * 100 : 100}%"></span><span class="gate-tick" style="left:${at(max)}%"></span></span>
    <span class="v">${fmt(value)}</span></div>`;
}

function ratioBar(e, b, margin) {
  if (e == null || b == null) return '–';
  const r = e / b;
  const max = Math.max(1.5, margin * 1.25);
  return `<div class="ratiobar ${r > margin ? 'slow' : ''}"><span class="track"><span class="fill" style="width:${Math.min(1, r / max) * 100}%"></span><span class="gate-tick" style="left:${(margin / max) * 100}%"></span></span><span>${r.toFixed(2)}×</span></div>`;
}

const status = s => `<span class="status ${s.passed ? 'pass' : 'fail'}">${s.passed ? 'Pass' : 'Fail'}</span>`;

function gpuTable(gpu, names) {
  if (!gpu) return '<p class="empty">No local GPU run is committed yet. Run <code>bash scripts/record-local-gpu.sh</code> on the RTX 3060 Ti and commit <code>docs/oracle-local-gpu.json</code>.</p>';
  const rows = gpu.scenarios.filter(s => !names || names.includes(s.scenario));
  return `<div class="table-wrap"><table>
    <thead><tr><th>Scenario</th><th>vs Bevy</th><th>Result</th><th>render-engine</th><th>Bevy</th></tr></thead>
    <tbody>${rows.map(s => `<tr><td><a href="scenario.html?s=${encodeURIComponent(s.scenario)}">${esc(s.scenario)}</a></td>
      <td class="ratio">${ratioBar(s.engine_gpu_ms, s.bevy_gpu_ms, gpu.gpu_margin)}</td><td>${status(s)}</td><td>${ms(s.engine_gpu_ms)}</td><td>${ms(s.bevy_gpu_ms)}</td></tr>`).join('')}</tbody>
  </table></div>`;
}

function runInfo(build, gpu) {
  const parts = [];
  if (build?.sha) parts.push(`Built from <a href="${REPO}/commit/${esc(build.sha)}">${esc(build.sha.slice(0, 7))}</a>${build.date ? ` on ${esc(build.date)}` : ''}${build.run_url ? ` by <a href="${esc(build.run_url)}">this CI run</a>` : ''}.`);
  if (gpu) parts.push(`Local GPU numbers: ${esc(gpu.adapter)}${gpu.commit ? `, commit <a href="${REPO}/commit/${esc(gpu.commit)}">${esc(gpu.commit.slice(0, 7))}</a>` : ''}${gpu.date ? `, ${esc(gpu.date)}` : ''}.`);
  return parts.join(' ');
}

export function renderIndex({ ci, gpu, build }) {
  const main = document.querySelector('main');
  if (!ci) {
    main.innerHTML = '<div class="wrap"><p class="empty">No oracle report yet. Run the oracle (<code>cargo run --release --manifest-path oracle/Cargo.toml</code>), then <code>bash scripts/build-pages.sh</code>.</p></div>';
    return;
  }
  const total = ci.scenarios.length;
  const passed = ci.scenarios.filter(s => s.passed).length;
  const worstMean = Math.max(...ci.scenarios.map(s => s.flip_mean));
  const worstP99 = Math.max(...ci.scenarios.map(s => s.flip_p99));
  const gpuPassed = gpu ? gpu.scenarios.filter(s => s.passed).length : null;
  const worstRatio = gpu ? Math.max(...gpu.scenarios.filter(s => s.engine_gpu_ms && s.bevy_gpu_ms).map(s => s.engine_gpu_ms / s.bevy_gpu_ms)) : null;
  const features = ci.features.filter(f => ci.scenarios.some(s => s.feature === f.id));

  main.innerHTML = `
  <section class="hero"><div class="wrap">
    <h1 class="verdict ${passed < total ? 'failing' : ''}">${passed} <span class="of">of ${total} scenarios match Bevy</span></h1>
    <p class="lede">Every scenario is rendered headless by Bevy, the reference renderer, and by render-engine, then compared with NVIDIA FLIP, a perceptual image difference. Hover a scenario to see its FLIP heat map, or open it to compare the images side by side, by swiping or by toggling.</p>
    <div class="gates">
      <div class="gate"><h2>Image gate, CI on lavapipe</h2><p>Mean FLIP ≤ ${g(ci.flip_mean_max)}, 99th percentile ≤ ${g(ci.flip_p99_max)}</p><p>Worst: <span class="num">${fmt(worstMean)}</span> mean, <span class="num">${fmt(worstP99)}</span> p99</p></div>
      <div class="gate"><h2>GPU-time gate, local RTX 3060 Ti</h2><p>render-engine ≤ ${(gpu?.gpu_margin ?? ci.gpu_margin).toFixed(2)}× Bevy's median frame time</p>${gpu ? `<p>Slowest: <span class="num">${worstRatio.toFixed(2)}×</span>, ${gpuPassed} of ${gpu.scenarios.length} pass</p>` : '<p>No local run committed.</p>'}</div>
    </div>
  </div></section>
  <div class="wrap">
    <div class="filters" role="group" aria-label="Filter by feature">
      <button class="chip" data-f="" aria-pressed="true">All<span>${total}</span></button>
      ${features.map(f => `<button class="chip" data-f="${esc(f.id)}" aria-pressed="false">${esc(f.label)}<span>${ci.scenarios.filter(s => s.feature === f.id).length}</span></button>`).join('')}
    </div>
    ${features.map(f => `<section class="group" data-f="${esc(f.id)}">
      <header><h2>${esc(f.label)}</h2><span class="links">${prLinks(f)}${decisionLink(f)}</span></header>
      <div class="grid">${ci.scenarios.filter(s => s.feature === f.id).map(s => `
        <a class="card" href="scenario.html?s=${encodeURIComponent(s.scenario)}">
          <div class="thumb"><img loading="lazy" src="oracle/${esc(s.scenario)}.engine.png" alt="${esc(s.scenario)}, render-engine"><img class="heat" loading="lazy" src="oracle/${esc(s.scenario)}.flip.png" alt=""></div>
          <div class="body">
            <div class="row"><h3>${esc(s.scenario)}</h3>${status(s)}</div>
            <p class="cap">${esc(s.capability)}</p>
            ${scale('Mean', s.flip_mean, g(ci.flip_mean_max))}
            ${scale('p99', s.flip_p99, g(ci.flip_p99_max))}
          </div>
        </a>`).join('')}</div>
    </section>`).join('')}
    <section class="section" id="gpu-time">
      <h2>GPU time</h2>
      <p>Median GPU frame time over ${gpu?.frames ?? 200} frames per side; the gate is ${(gpu?.gpu_margin ?? ci.gpu_margin).toFixed(2)}× Bevy, marked on each bar. Measured locally on ${esc(gpu?.adapter ?? 'the RTX 3060 Ti')}. CI has no GPU, so these numbers come from the latest committed local run, not from the CI run that built this page.</p>
      ${gpuTable(gpu)}
    </section>
    <footer>${runInfo(build, gpu)} <a href="${REPO}">Source</a>.</footer>
  </div>`;

  const chips = [...main.querySelectorAll('.chip')];
  const apply = id => {
    chips.forEach(c => c.setAttribute('aria-pressed', String(c.dataset.f === id)));
    main.querySelectorAll('.group').forEach(g => { g.hidden = id !== '' && g.dataset.f !== id; });
  };
  chips.forEach(c => c.addEventListener('click', () => {
    apply(c.dataset.f);
    history.replaceState(null, '', c.dataset.f ? `#${c.dataset.f}` : location.pathname);
  }));
  const initial = location.hash.slice(1);
  if (features.some(f => f.id === initial)) apply(initial);
  else if (initial) document.getElementById(initial)?.scrollIntoView();
}

export function renderScenario({ ci, gpu, build }) {
  const main = document.querySelector('main');
  const name = new URLSearchParams(location.search).get('s');
  const i = ci ? ci.scenarios.findIndex(s => s.scenario === name) : -1;
  if (i < 0) {
    main.innerHTML = `<div class="wrap"><p class="empty">${ci ? `No scenario named “${esc(name ?? '')}” in this report.` : 'No oracle report yet.'} <a href="./">Show all scenarios</a></p></div>`;
    return;
  }
  const s = ci.scenarios[i];
  const f = featureOf(ci, s);
  const local = gpu?.scenarios.find(g => g.scenario === s.scenario);
  const prev = ci.scenarios[(i - 1 + ci.scenarios.length) % ci.scenarios.length];
  const next = ci.scenarios[(i + 1) % ci.scenarios.length];
  const img = k => `oracle/${encodeURIComponent(s.scenario)}.${k}.png`;
  document.title = `${s.scenario} · render-engine vs Bevy`;

  main.innerHTML = `<div class="wrap">
    <p class="crumbs"><a href="./">All scenarios</a> / <a href="./#${esc(f.id)}">${esc(f.label)}</a></p>
    <h1 class="title">${esc(s.scenario)}</h1>
    <p class="sub">${esc(s.capability)} · ${s.width}×${s.height} · ${prLinks(f).replace(/<\/a><a/g, '</a> · <a')} · ${decisionLink(f)}</p>
    <div class="facts">
      <div class="gate"><h2>FLIP on lavapipe (CI)</h2>${scale('Mean', s.flip_mean, g(ci.flip_mean_max))}${scale('p99', s.flip_p99, g(ci.flip_p99_max))}<p>${status(s)} gate: mean ≤ ${g(ci.flip_mean_max)}, p99 ≤ ${g(ci.flip_p99_max)}</p></div>
      <div class="gate"><h2>GPU time, local RTX 3060 Ti</h2>${local ? `<p>render-engine <span class="num">${ms(local.engine_gpu_ms)}</span></p><p>Bevy ${ms(local.bevy_gpu_ms)}</p>${ratioBar(local.engine_gpu_ms, local.bevy_gpu_ms, gpu.gpu_margin)}` : '<p>Not in the committed local run.</p>'}</div>
      <div class="gate"><h2>GPU time on lavapipe (advisory)</h2><p>render-engine ${ms(s.engine_gpu_ms)}</p><p>Bevy ${ms(s.bevy_gpu_ms)}</p><p>Software rendering; not gated.</p></div>
    </div>
    <div class="modes" role="group" aria-label="Comparison view">
      <button data-m="side" aria-pressed="true">Side by side</button>
      <button data-m="swipe" aria-pressed="false">Swipe</button>
      <button data-m="toggle" aria-pressed="false">Toggle</button>
    </div>
    <div id="view"></div>
    <nav class="pager"><a href="scenario.html?s=${encodeURIComponent(prev.scenario)}">Previous: ${esc(prev.scenario)}</a><a href="scenario.html?s=${encodeURIComponent(next.scenario)}">Next: ${esc(next.scenario)}</a></nav>
    <footer>${runInfo(build, gpu)}</footer>
  </div>`;

  const legend = `<div class="legend"><span>0, identical</span><span class="ramp"></span><span>1, maximally different</span></div>`;
  const view = main.querySelector('#view');
  const modes = {
    side: () => {
      view.innerHTML = `<div class="triple">
        <figure><img src="${img('engine')}" alt="render-engine"><figcaption>render-engine</figcaption></figure>
        <figure><img src="${img('bevy')}" alt="Bevy"><figcaption>Bevy (oracle)</figcaption></figure>
        <figure><img src="${img('flip')}" alt="FLIP error map"><figcaption>FLIP error map</figcaption>${legend}</figure>
      </div>`;
    },
    swipe: () => {
      view.innerHTML = `<div class="stage" style="--x:50%">
        <img src="${img('bevy')}" alt="Bevy">
        <img class="top" src="${img('engine')}" alt="render-engine">
        <span class="handle"></span><span class="tag l">render-engine</span><span class="tag r">Bevy</span>
        <input type="range" min="0" max="100" step="0.1" value="50" aria-label="Swipe between render-engine and Bevy">
      </div><p class="hint">Drag across the image, or focus it and use the arrow keys.</p>`;
      const stage = view.querySelector('.stage');
      const range = stage.querySelector('input');
      range.addEventListener('input', () => stage.style.setProperty('--x', `${range.value}%`));
    },
    toggle: () => {
      const states = [['engine', 'render-engine'], ['bevy', 'Bevy (oracle)'], ['flip', 'FLIP error map']];
      let at = 0;
      view.innerHTML = `<div class="stage toggle" role="button" tabindex="0" aria-label="Show the next image">
        <img src="${img('engine')}" alt=""><span class="tag l"></span>
      </div><p class="hint">Click, tap or press Space to cycle render-engine, Bevy and the FLIP map.</p>${legend}`;
      const stage = view.querySelector('.stage');
      const image = stage.querySelector('img');
      const tag = stage.querySelector('.tag');
      states.forEach(([k]) => { new Image().src = img(k); });
      const show = n => { at = n; image.src = img(states[at][0]); image.alt = states[at][1]; tag.textContent = states[at][1]; };
      show(0);
      stage.addEventListener('click', () => show((at + 1) % states.length));
      stage.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); show((at + 1) % states.length); } });
    },
  };
  const buttons = [...main.querySelectorAll('.modes button')];
  const pick = m => {
    buttons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.m === m)));
    modes[m]();
    try { localStorage.setItem('compare', m); } catch {}
  };
  buttons.forEach(b => b.addEventListener('click', () => pick(b.dataset.m)));
  let saved = 'side';
  try { saved = localStorage.getItem('compare') || 'side'; } catch {}
  pick(modes[saved] ? saved : 'side');
}
