"""Step 4: the listening page, <run>/index.html (static, relative audio paths; modeled on english-practice's
native-server/tests/kokoro_port/fixtures/compare_packs.py page).

    .venv/bin/python tests/e2e/kokoro/page.py [--run NAME]
    cd e2e-runs/kokoro/<run> && ../../../.venv/bin/python -m http.server 8766 --bind 127.0.0.1

Columns: text info (text, length class, chunks, duration) / voice / Python / lokinfer Mac GPU / lokinfer iOS GPU,
plus the similarities and duration-match marks; sortable (click a header), filterable (search, class, voice,
"only mismatches / outliers"). The Python cell also plays the Python 2nd run (other seeds) as the natural
run-to-run variation.
"""
import html
import json
import sys

from common import arg, run_dir


def main():
    run = run_dir(arg(sys.argv[1:], "--run"))
    rows = json.loads((run / "rows.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    slim = []
    for r in rows:
        slim.append({k: r[k] for k in ("i", "id", "text", "cls", "voice", "source", "chunks", "tokens", "phonemes",
                                        "dur_mac_gpu", "dur_ios_gpu", "dur_python_b", "sim_mac_py", "sim_ios_py",
                                        "sim_mac_ios", "sim_base", "flags")}
                    | {b: (r[b] and {k: r[b][k] for k in ("samples", "ms", "error") if k in r[b]})
                       for b in ("python", "python_b", "mac_gpu", "ios_gpu")})
    outliers = {o["id"] for b in summary["backends"].values() for o in b["outliers"]}
    for r in slim:
        r["outlier"] = r["id"] in outliers
    page = PAGE.replace("__DATA__", json.dumps({"rows": slim, "summary": summary}, ensure_ascii=False)
                        .replace("</", "<\\/")).replace("__TITLE__", html.escape(run.name))
    (run / "index.html").write_text(page)
    print("wrote", run / "index.html")


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kokoro e2e __TITLE__</title>
<style>
  body { font: 14px/1.4 -apple-system, system-ui, sans-serif; margin: 0 auto; max-width: 1400px; padding: 16px; background: #fafafa; color: #222; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .summary { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; margin: 10px 0 14px; }
  .summary table { width: auto; } .summary td, .summary th { border: none; padding: 2px 10px 2px 0; position: static; background: none; cursor: default; }
  .pass { color: #276749; font-weight: 700; } .fail, .bad { color: #c53030; font-weight: 700; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; position: sticky; top: 0; background: #fafafa; padding: 6px 0; z-index: 2; }
  input[type=search] { flex: 1; min-width: 200px; padding: 6px 8px; }
  table { width: 100%; border-collapse: collapse; background: #fff; }
  td, th { border-bottom: 1px solid #eee; padding: 6px; text-align: left; vertical-align: top; }
  th { position: sticky; top: 44px; background: #f0f0f0; cursor: pointer; white-space: nowrap; z-index: 1; }
  button { padding: 4px 10px; border-radius: 6px; border: 1px solid #bbb; background: #fff; cursor: pointer; margin: 1px; }
  button.playing { background: #2b6cb0; color: #fff; border-color: #2b6cb0; }
  .num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .meta { color: #666; font-size: 12px; } .ph { color: #888; font-size: 11px; display: none; } .showph .ph { display: block; }
  .flag { background: #fff5f5; } .warn { color: #b7791f; font-weight: 600; }
</style></head><body>
<h1>Kokoro-82M end to end: Python (english-practice) vs lokinfer Mac GPU vs lokinfer iOS GPU &mdash; __TITLE__</h1>
<div class="summary" id="summary"></div>
<div class="controls">
  <input type="search" id="filter" placeholder="Search text / id...">
  <select id="cls"><option value="">all classes</option></select>
  <select id="voice"><option value="">all voices</option></select>
  <label><input type="checkbox" id="onlyBad"> only mismatches / outliers</label>
  <label><input type="checkbox" id="showPh"> phonemes</label>
  <span id="count" class="meta"></span>
</div>
<table id="tbl"><thead><tr>
  <th data-k="i">Text info</th><th data-k="voice">Voice</th>
  <th data-k="py_ms">Python</th><th data-k="mac_ms">lokinfer Mac GPU</th><th data-k="ios_ms">lokinfer iOS GPU</th>
  <th data-k="sim_mac_py" class="num">Mac&harr;Py</th><th data-k="sim_ios_py" class="num">iOS&harr;Py</th>
  <th data-k="sim_mac_ios" class="num">Mac&harr;iOS</th><th data-k="sim_base" class="num">Py&harr;Py2</th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
const DATA = __DATA__;
const S = DATA.summary;
const audio = new Audio();
let sortKey = 'i', sortDir = 1;
const fmt = s => (s / 24000).toFixed(2) + ' s';
const f4 = x => x == null ? '&ndash;' : x.toFixed(4);
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
function summary() {
  const b = S.baseline_python_vs_python || {};
  let h = `<b class="${S.pass ? 'pass' : 'fail'}">${S.pass ? 'PASS' : 'FAIL'}</b> &mdash; ${S.texts} texts, ${S.chunks} chunks.
    Similarity = english-practice's spectral similarity (cosine of log-magnitude spectrograms, n_fft 1024, hop 256);
    Kokoro's decoder is random on every call, so even two Python runs differ. Tolerances: median ${S.tolerances.median}, min ${S.tolerances.min}.<br>
    <table><tr><th></th><th>texts</th><th>durations = Python</th><th>sim. median</th><th>sim. min</th><th>outliers</th><th>verdict</th></tr>
    <tr><td>Baseline Python&harr;Python (other seeds)</td><td>${b.n ?? '-'}</td><td>${S.baseline_durations_identical}</td><td>${f4(b.median)}</td><td>${f4(b.min)}</td><td></td><td></td></tr>`;
  for (const [k, v] of Object.entries(S.backends)) {
    const s = v.similarity_vs_python || {};
    h += `<tr><td>${k === 'mac_gpu' ? 'lokinfer Mac GPU' : 'lokinfer iOS GPU'} &harr; Python</td><td>${v.texts_done}/${v.texts_expected}</td>
      <td class="${v.durations_identical === v.texts_expected ? '' : 'bad'}">${v.durations_identical}/${v.texts_expected}</td>
      <td>${f4(s.median)}</td><td>${f4(s.min)}</td><td>${v.outliers.length}</td><td class="${v.pass ? 'pass' : 'fail'}">${v.pass ? 'PASS' : 'FAIL'}</td></tr>`;
  }
  if (S.mac_vs_ios) h += `<tr><td>Mac GPU &harr; iOS GPU</td><td>${S.mac_vs_ios.n}</td><td></td><td>${f4(S.mac_vs_ios.median)}</td><td>${f4(S.mac_vs_ios.min)}</td><td></td><td></td></tr>`;
  h += '</table><br><b>Time</b> (model calls only, setup excluded):<table><tr><th></th><th>total</th><th>audio</th><th>real-time factor</th><th>median / text</th></tr>';
  const names = {python: 'Python (PyTorch CPU)', python_b: 'Python 2nd run', mac_gpu: 'lokinfer Mac GPU', ios_gpu: 'lokinfer iOS GPU (on device)'};
  for (const [k, t] of Object.entries(S.timing)) h += `<tr><td>${names[k]}</td><td>${t.total_s.toFixed(1)} s</td><td>${t.audio_s.toFixed(0)} s</td><td>${t.rtf.toFixed(4)}</td><td>${t.median_ms_per_text.toFixed(1)} ms</td></tr>`;
  document.getElementById('summary').innerHTML = h + '</table>';
}
function play(src, button) {
  document.querySelectorAll('button.playing').forEach(b => b.classList.remove('playing'));
  if (audio.dataset.src === src && !audio.paused) { audio.pause(); return; }
  audio.src = src; audio.dataset.src = src; audio.play(); button.classList.add('playing');
  audio.onended = () => button.classList.remove('playing');
}
function cell(r, b, label, durKey) {
  const x = r[b];
  if (!x) return '<span class="meta">not run</span>';
  if (x.error) return `<span class="bad">error: ${esc(x.error)}</span>`;
  const mark = durKey == null ? '' : r[durKey] ? ' <span title="same sample count as Python">&#10003;</span>' : ' <span class="bad" title="duration differs from Python">&#10007; ' + fmt(x.samples) + '</span>';
  return `<button data-src="audio/${b}/${r.id}.wav">${label}</button>${mark}<div class="meta">${x.ms.toFixed(0)} ms</div>`;
}
const key = (r, k) => k === 'py_ms' ? r.python?.ms : k === 'mac_ms' ? r.mac_gpu?.ms : k === 'ios_ms' ? r.ios_gpu?.ms : r[k];
function bad(r) { return r.flags.length > 0 || r.outlier; }
function render() {
  const q = document.getElementById('filter').value.toLowerCase();
  const cls = document.getElementById('cls').value, voice = document.getElementById('voice').value;
  const onlyBad = document.getElementById('onlyBad').checked;
  let rows = DATA.rows.filter(r => (r.text.toLowerCase().includes(q) || r.id.includes(q)) && (!cls || r.cls === cls)
    && (!voice || r.voice === voice) && (!onlyBad || bad(r)));
  rows.sort((a, b) => { const va = key(a, sortKey) ?? -1, vb = key(b, sortKey) ?? -1; return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir; });
  document.getElementById('count').textContent = `${rows.length} / ${DATA.rows.length} rows`;
  const body = document.getElementById('rows');
  body.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    if (bad(r)) tr.className = 'flag';
    const dur = r.python ? fmt(r.python.samples) : '';
    tr.innerHTML = `<td><div>${esc(r.text).replace(/\n/g, '<br>')}</div>
        <div class="meta">${r.id} &middot; ${r.cls} &middot; ${r.chunks} chunk${r.chunks > 1 ? 's' : ''} (T ${r.tokens.join(', ')}) &middot; ${dur} &middot; ${esc(r.source)}
        ${r.flags.length ? '<br><span class="bad">' + r.flags.join(', ') + '</span>' : ''}${r.outlier ? ' <span class="warn">similarity outlier</span>' : ''}</div>
        <div class="ph">${esc(r.phonemes)}</div></td>
      <td>${r.voice}</td>
      <td class="num">${cell(r, 'python', 'Python', null)}${r.python_b ? ` <button data-src="audio/python_b/${r.id}.wav" title="2nd Python run, other seeds">Py2</button>` : ''}</td>
      <td class="num">${cell(r, 'mac_gpu', 'Mac GPU', 'dur_mac_gpu')}</td>
      <td class="num">${cell(r, 'ios_gpu', 'iOS GPU', 'dur_ios_gpu')}</td>
      <td class="num">${f4(r.sim_mac_py)}</td><td class="num">${f4(r.sim_ios_py)}</td><td class="num">${f4(r.sim_mac_ios)}</td><td class="num">${f4(r.sim_base)}</td>`;
    tr.querySelectorAll('button').forEach(b => b.onclick = () => play(b.dataset.src, b));
    body.appendChild(tr);
  }
}
for (const [id, k] of [['cls', 'cls'], ['voice', 'voice']]) {
  const sel = document.getElementById(id);
  for (const v of [...new Set(DATA.rows.map(r => r[k]))].sort()) sel.insertAdjacentHTML('beforeend', `<option>${v}</option>`);
  sel.onchange = render;
}
document.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {
  sortDir = sortKey === th.dataset.k ? -sortDir : 1; sortKey = th.dataset.k; render();
});
document.getElementById('filter').oninput = render;
document.getElementById('onlyBad').onchange = render;
document.getElementById('showPh').onchange = e => document.getElementById('tbl').classList.toggle('showph', e.target.checked);
summary(); render();
</script></body></html>
'''


if __name__ == "__main__":
    main()
