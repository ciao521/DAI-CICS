// ─── Scenario descriptions ────────────────────────────
const SC_DESC = {
  A: '<strong>ホモ・エコノミカス分断・利他行動ゼロ</strong>',
  B: '<strong>共同体過依存・搾取</strong>',
  C: '<strong>AI支援・超包括ケア</strong>',
  'C-noN2': '<strong>社会的処方ナッジ（N2）なし。</strong>',
  'C-noN3': '<strong>負荷再配分ナッジ（N3）なし。</strong>',
  'C-onlyL1': '<strong>L1情報提示のみ。</strong>',
};
const sel = document.getElementById('selScenario');
const updDesc = () => document.getElementById('scDesc').innerHTML = SC_DESC[sel.value] || '';
sel.addEventListener('change', updDesc); updDesc();

const slDay = document.getElementById('slDay');
slDay.addEventListener('input', () => {
  document.getElementById('dayLbl').textContent = slDay.value;
  document.getElementById('dayNum').textContent = slDay.value;
});

// ─── Color helpers ─────────────────────────────────────
function cls(v, w, d, inv=false) {
  if (typeof v !== 'number') return '';
  if (inv) return v >= w ? '' : v >= d ? 'c-warn' : 'c-danger';
  return v < w ? '' : v < d ? 'c-warn' : 'c-danger';
}
function scoreColor(v) {
  return typeof v!=='number' ? 'var(--muted)' : v>=.8 ? 'var(--ok)' : v>=.6 ? 'var(--warn)' : 'var(--danger)';
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function fmt(v) {
  if (typeof v !== 'number') return v;
  return v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(1) : v.toFixed(3);
}

// ─── Robust text extraction from turn object ────────────
const DIALOGUE_KEYS = ['spoken_dialogue','dialogue','speech','message','text',
  'response','発言','content','statement','utterance','output'];
const REASONING_KEYS = ['internal_reasoning','reasoning','thinking','thought',
  'rationale','内部思考','internal_thought','chain_of_thought','system_analysis',
  'system_reasoning'];
const INTERVENTION_KEYS = ['proposed_intervention','intervention','plan','proposal',
  'action_plan'];
const SKIP_KEYS = new Set(['agent','turn','timestamp','fatigue','altruism',
  'fc_triggered','nudge_level','applied_nudge','proposed_action_type','milestone_achieved',
  'milestone_hit','cm_fatigue_after_nudge']);

function extractText(t, keys) {
  for (const k of keys) { if (t[k] && typeof t[k]==='string' && t[k].length>5) return t[k]; }
  return '';
}

// Sanitize JSON: escape raw control chars inside string literals (same logic as Python _extract_json)
function sanitizeJsonControls(s) {
  let inStr = false, esc = false;
  const out = [];
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (esc) { out.push(ch); esc = false; continue; }
    if (ch === '\\') { out.push(ch); if (inStr) esc = true; continue; }
    if (ch === '"') { inStr = !inStr; out.push(ch); continue; }
    if (inStr && (ch === '\n' || ch === '\r' || ch === '\t')) {
      out.push(ch === '\n' ? '\\n' : ch === '\r' ? '\\r' : '\\t');
      continue;
    }
    out.push(ch);
  }
  return out.join('');
}

// Extract balanced { ... } object by brace-counting (avoids lastIndexOf bug)
function extractBalancedObject(text) {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (esc) { esc = false; continue; }
    if (ch === '\\' && inStr) { esc = true; continue; }
    if (ch === '"') { inStr = !inStr; continue; }
    if (!inStr) {
      if (ch === '{') depth++;
      else if (ch === '}') { if (--depth === 0) return text.slice(start, i + 1); }
    }
  }
  return null;
}

// Regex-based key extraction when JSON parsing completely fails
function extractByRegex(text, keys) {
  const content = text.replace(/```(?:json)?/g, '').replace(/```/g, '').trim();
  for (const key of keys) {
    const kIdx = content.indexOf(`"${key}"`);
    if (kIdx === -1) continue;
    const after = content.slice(kIdx + key.length + 2);
    const colonM = after.match(/^\s*:\s*"/);
    if (!colonM) continue;
    const valStart = kIdx + key.length + 2 + colonM[0].length;
    let pos = valStart, valEnd = -1;
    while (pos < content.length) {
      const ch = content[pos];
      if (ch === '\\') { pos += 2; continue; }
      if (ch === '"') {
        const nxt = content.slice(pos + 1).replace(/^[ \t]*/,'');
        if (nxt === '' || nxt[0] === ',' || nxt[0] === '}' || nxt[0] === '\n') {
          valEnd = pos; break;
        }
      }
      pos++;
    }
    if (valEnd > valStart) {
      const val = content.slice(valStart, valEnd)
        .replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\"/g, '"').replace(/\\r/g, '');
      if (val.length > 5) return val;
    }
  }
  return '';
}

// Try to parse a JSON object from text (markdown code block or raw JSON)
function tryParseJsonBlock(text) {
  if (!text) return null;
  const candidates = [];
  // From markdown code block (non-greedy, with or without closing ```)
  const m = text.match(/```(?:json)?[^\n]*\n([\s\S]+?)(?:```|$)/);
  if (m) candidates.push(m[1].trim());
  // From balanced brace extraction (correct even if trailing content has braces)
  const bal = extractBalancedObject(text);
  if (bal) candidates.push(bal);
  // Raw text
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) candidates.push(trimmed);
  for (const c of [...new Set(candidates)]) {
    try { return JSON.parse(c); } catch {}
    try { return JSON.parse(sanitizeJsonControls(c)); } catch {}
  }
  return null;
}

// Attempt to extract dialogue/reasoning from a parsed object, with regex fallback
function resolveDialogue(text) {
  const parsed = tryParseJsonBlock(text);
  if (parsed) return extractText(parsed, DIALOGUE_KEYS) || extractText(parsed, INTERVENTION_KEYS) || '';
  // Regex fallback when JSON parsing fails completely
  return extractByRegex(text, [...DIALOGUE_KEYS, ...INTERVENTION_KEYS]);
}
function resolveReasoning(text) {
  const parsed = tryParseJsonBlock(text);
  if (parsed) return extractText(parsed, REASONING_KEYS) || '';
  return extractByRegex(text, REASONING_KEYS);
}

function looksLikeJsonBlock(s) {
  return s.includes('```') || s.trim().startsWith('{');
}
function extractDialogue(t) {
  // Direct field lookup
  const direct = extractText(t, DIALOGUE_KEYS) || extractText(t, INTERVENTION_KEYS);
  if (direct) {
    if (looksLikeJsonBlock(direct)) {
      const r = resolveDialogue(direct);
      if (r) return r;
    }
    return direct;
  }
  // Last resort: longest string field not in skip keys
  const longest = Object.entries(t)
    .filter(([k,v]) => typeof v==='string' && !SKIP_KEYS.has(k) && v.length>20)
    .sort((a,b) => b[1].length - a[1].length).map(e=>e[1])[0] || '';
  if (longest) {
    if (looksLikeJsonBlock(longest)) return resolveDialogue(longest) || longest;
  }
  return longest;
}
function extractReasoning(t) {
  const direct = extractText(t, REASONING_KEYS);
  if (direct) return direct;
  // Try to extract reasoning from JSON block in longest string field
  const longest = Object.entries(t)
    .filter(([k,v]) => typeof v==='string' && !SKIP_KEYS.has(k) && v.length>20)
    .sort((a,b) => b[1].length - a[1].length).map(e=>e[1])[0] || '';
  if (longest && looksLikeJsonBlock(longest)) return resolveReasoning(longest);
  return '';
}

// ─── Render context card ────────────────────────────────
function renderContext(ctx) {
  document.getElementById('ctxLbl').textContent =
    `${ctx.abm_scenario} · Day ${ctx.abm_day}`;
  document.getElementById('cardCtx').style.display = '';
  const ms = [
    {label:'CM疲労度',   val:ctx.cm_fatigue,           sub:'mean_fatigue',  c:cls(ctx.cm_fatigue,.70,.85)},
    {label:'孤立度',     val:ctx.mean_isolation,        sub:'isolation',     c:cls(ctx.mean_isolation,.50,.75)},
    {label:'SDHリスク',  val:ctx.mean_sdh_risk,         sub:'sdh_risk',      c:cls(ctx.mean_sdh_risk,.50,.70)},
    {label:'累積急性期', val:ctx.cum_acute_events,      sub:'入院相当',       c:cls(ctx.cum_acute_events,400,800)},
    {label:'Gini係数',   val:ctx.gini_fatigue,          sub:'負荷集中度',     c:cls(ctx.gini_fatigue,.25,.40)},
    {label:'連携レベル', val:ctx.coordination_level,    sub:'coordination',  c:cls(ctx.coordination_level,.70,.40,true)},
    {label:'バーンアウト',val:ctx.burnout_count,        sub:'累積人数',       c:cls(ctx.burnout_count,5,15)},
    {label:'ナッジ数',   val:ctx.total_nudge_interventions||0, sub:'累積', c:''},
  ];
  document.getElementById('metricsGrid').innerHTML = ms.map(m=>
    `<div class="metric-item">
       <div class="m-label">${m.label}</div>
       <div class="m-value ${m.c}">${fmt(m.val)}</div>
       <div class="m-sub">${m.sub}</div>
     </div>`).join('');
  const flags = [];
  if (ctx.fc_a1>0) flags.push(`<span class="flag flag-a">FC-A1: ${ctx.fc_a1}回</span>`);
  if (ctx.fc_a2>0) flags.push(`<span class="flag flag-a">FC-A2: ${ctx.fc_a2}回</span>`);
  if (ctx.fc_b1>0) flags.push(`<span class="flag flag-b">FC-B1: ${ctx.fc_b1}回</span>`);
  if (ctx.fc_b2>0) flags.push(`<span class="flag flag-b">FC-B2: ${ctx.fc_b2}回</span>`);
  if (ctx.fc_c1>0) flags.push(`<span class="flag flag-c">FC-C1: ${ctx.fc_c1}回</span>`);
  if (!flags.length) flags.push('<span class="flag">✓ 失敗条件なし</span>');
  document.getElementById('flagsRow').innerHTML = flags.join('');
}

// ─── Render turn as chat bubble ─────────────────────────
const AGENT_META = {
  Doctor:      { emoji:'👨‍⚕️', name:'Doctor（医師）',              pos:'left',   hd:'doc',  av:'av-doctor' },
  CareManager: { emoji:'🧑‍💼', name:'Care Manager（ケアマネジャー）',       pos:'right',  hd:'care', av:'av-care'   },
  PlannerAI:   { emoji:'🤖', name:'PlannerAI（AI支援）',   pos:'center', hd:'ai',   av:'av-ai'     },
};
function renderTurn(t) {
  const agent = t.agent || 'Unknown';
  const meta  = AGENT_META[agent] || { emoji:'💬', name:agent, pos:'left', hd:'doc', av:'av-doctor' };

  const dialogue  = extractDialogue(t);
  const reasoning = extractReasoning(t);

  const tags = [];
  if (t.milestone_achieved)    tags.push(`<span class="tag tag-ok">✅ ${esc(t.milestone_achieved)}</span>`);
  if (t.milestone_hit)         tags.push(`<span class="tag tag-ok">🏁 ${esc(t.milestone_hit)}</span>`);
  if (t.fc_triggered)          tags.push(`<span class="tag tag-warn">⚠️ ${esc(t.fc_triggered)}</span>`);
  if (t.nudge_level)           tags.push(`<span class="tag tag-nudge">🎯 L${t.nudge_level} ${esc(t.applied_nudge||'')}</span>`);
  if (t.proposed_action_type)  tags.push(`<span class="tag">${esc(t.proposed_action_type)}</span>`);

  const rid = `r-${t.turn || Date.now()}`;

  const row = document.createElement('div');
  row.className = `chat-row ${meta.pos}`;
  row.innerHTML =
    (meta.pos !== 'center' ? `<div class="avatar ${meta.av}">${meta.emoji}</div>` : '') +
    `<div class="chat-bubble">
       <div class="bub-hd ${meta.hd}">
         <span>${meta.emoji} ${meta.name}</span>
         <span class="bub-turn-no">T${t.turn||'?'}</span>
       </div>` +
    (reasoning ? `<div class="bub-reasoning"><div class="bub-reasoning-label">💭 内部思考</div><div class="bub-reasoning-body" id="${rid}">${esc(reasoning)}</div></div>` : '') +
    (dialogue ? `<div class="bub-text"><div class="bub-speech-label">💬 発話内容</div>${esc(dialogue)}</div>` :
      `<div class="bub-text" style="color:var(--muted);font-style:italic">（発言テキストなし — ターンデータ: ${esc(JSON.stringify(t).substring(0,120))}）</div>`) +
    (tags.length ? `<div class="bub-foot">${tags.join('')}</div>` : '') +
    `</div>`;

  document.getElementById('chatList').appendChild(row);
  document.getElementById('cardDlg').style.display = '';
  row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function toggleR(id) {
  const body = document.getElementById(id);
  const btn  = document.getElementById(id+'-btn');
  if (!body) return;
  const expanded = body.classList.toggle('expanded');
  body.classList.toggle('collapsed', !expanded);
  btn.textContent = expanded ? '▲ 閉じる' : '▼ 展開';
}

// ─── Render eval card ───────────────────────────────────
function renderEval(ev) {
  document.getElementById('cardEval').style.display = '';
  const sum = ev.summary || {};
  const ff = sum.mean_faithfulness    ?? '—';
  const cr = sum.mean_context_relevance ?? '—';
  document.getElementById('evalScores').innerHTML =
    `<div class="eval-score">
       <div class="es-label">Faithfulness</div>
       <div class="es-value" style="color:${scoreColor(ff)}">${typeof ff==='number'?ff.toFixed(2):ff}</div>
     </div>
     <div class="eval-score">
       <div class="es-label">Context Relevance</div>
       <div class="es-value" style="color:${scoreColor(cr)}">${typeof cr==='number'?cr.toFixed(2):cr}</div>
     </div>`;
  if (ev.error) {
    document.getElementById('evalRecords').innerHTML =
      `<div class="eval-rec" style="color:var(--danger)">${esc(ev.error)}</div>`;
    return;
  }
  document.getElementById('evalRecords').innerHTML = (ev.records||[]).map((r,i)=>
    `<div class="eval-rec"><div class="er-ctx">Step ${i+1} · ${esc(r.context||'')}</div><div class="er-text">${esc(r.reasoning||r.output||'—')}</div><div class="er-scores"><span>Faithfulness: ${r.faithfulness??'—'}</span><span>ContextRelevance: ${r.context_relevance??'—'}</span></div></div>`).join('');
}

// ─── Log helper ─────────────────────────────────────────
function logLine(msg, type='') {
  const box = document.getElementById('logBox');
  if (box.children.length===1 && box.children[0].tagName==='SPAN') box.innerHTML='';
  const el = document.createElement('div');
  if (type==='ok')  el.className='log-ok';
  if (type==='err') el.className='log-err';
  el.textContent = msg;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

// ─── Export (print to PDF / screenshot) ─────────────────
function exportRecord() {
  // Expand all reasoning sections before printing
  document.querySelectorAll('.bub-reasoning-body.collapsed').forEach(el => {
    el.classList.remove('collapsed');
    el.classList.add('expanded');
  });
  setTimeout(() => window.print(), 100);
}

async function runSim() {
  const btn = document.getElementById('btnRun');
  btn.disabled = true; btn.classList.add('busy');
  document.getElementById('btnLbl').textContent = '実行中…';
  document.getElementById('logBox').innerHTML = '<span style="color:#94a3b8">実行中…</span>';
  ['cardCtx','cardDlg','cardEval'].forEach(id=>document.getElementById(id).style.display='none');
  document.getElementById('chatList').innerHTML = '';
  document.getElementById('emptyState').style.display = 'none';

  const sc  = document.getElementById('selScenario').value;
  const day = parseInt(document.getElementById('slDay').value);
  const body = { scenario: sc, day };
  const fat  = document.getElementById('ovFat').value;
  const gini = document.getElementById('ovGini').value;
  const acu  = document.getElementById('ovAcute').value;
  const iso  = document.getElementById('ovIso').value;
  if (fat)  body.cm_fatigue       = parseFloat(fat);
  if (gini) body.gini_fatigue     = parseFloat(gini);
  if (acu)  body.cum_acute_events = parseInt(acu);
  if (iso)  body.mean_isolation   = parseFloat(iso);

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const e = await res.json().catch(()=>({}));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let evt;
        try { evt = JSON.parse(line); } catch { continue; }
        switch (evt.type) {
          case 'log':     logLine(evt.msg, evt.msg.startsWith('✅') ? 'ok' : ''); break;
          case 'context': renderContext(evt.data); break;
          case 'turn':    renderTurn(evt.data);    break;
          case 'eval':    renderEval(evt.data);    break;
          case 'done':    logLine('🏁 完了', 'ok'); break;
          case 'error':   logLine('❌ '+evt.msg, 'err');
            document.getElementById('emptyState').style.display='flex'; break;
        }
      }
    }
  } catch(e) {
    logLine('❌ '+e.message, 'err');
    document.getElementById('emptyState').style.display = 'flex';
  } finally {
    btn.disabled = false; btn.classList.remove('busy');
    document.getElementById('btnLbl').textContent = '▶ シミュレーション実行';
  }
}