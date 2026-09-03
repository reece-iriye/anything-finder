/* Shared rendering for the telemetry console (scripts/trace_ui.html) and the
   generated comparison report (scripts/compare_report.html).

   The one thing worth keeping in sync between the two pages is how a trace is
   drawn: a run is a sequence of steps, and each tool the model calls is a
   round-trip — ① the arguments the model generated, ② your app running the
   tool, ③ the result handed back. `renderConversationInto` is that renderer;
   everything above it are the primitives it needs.

   Both pages inline this file (no <script src>) because the report must open
   from disk with no server. Nothing here touches global page state or fetch. */

const $ = (s,r=document)=>r.querySelector(s);
const el = (t,cls,txt)=>{const e=document.createElement(t); if(cls)e.className=cls; if(txt!=null)e.textContent=txt; return e;};
const fmt = n => (n==null?"—":n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e4?(n/1e3).toFixed(1)+"K":Number(n).toLocaleString());
const esc = s => String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const mdInline = s => esc(s)
  .replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>")
  .replace(/\*([^*\n]+)\*/g,"<em>$1</em>")
  .replace(/`([^`]+)`/g,"<code>$1</code>")
  .replace(/^#{1,6}\s+(.+)$/gm,"<strong>$1</strong>");
function ago(iso){ if(!iso) return ""; const s=(Date.now()-new Date(iso).getTime())/1000;
  if(s<60) return "just now"; if(s<3600) return Math.floor(s/60)+"m ago";
  if(s<86400) return Math.floor(s/3600)+"h ago"; return Math.floor(s/86400)+"d ago"; }
function clock(iso){ try{ return new Date(iso).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"}); }catch{ return ""; } }

const IC = {
  brain:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9.5 2A2.5 2.5 0 007 4.5v.5a3 3 0 00-1 5.83V15a3 3 0 003 3h.5A2.5 2.5 0 0012 15.5v-11A2.5 2.5 0 009.5 2zM14.5 2A2.5 2.5 0 0117 4.5v.5a3 3 0 011 5.83V15a3 3 0 01-3 3h-.5A2.5 2.5 0 0112 15.5"/></svg>',
  wrench:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14.7 6.3a4 4 0 01-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 005.4-5.4l-2.6 2.6-2.4-2.4 2.6-2.6z"/></svg>',
  result:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 6L9 17l-5-5"/></svg>',
  chev:'<svg class="ic chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>',
  msg:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>',
  bolt:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M13 2L3 14h9l-1 8 10-12h-9z"/></svg>',
  eye:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>',
  scale:'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v18M5 7h14M7 7l-4 7h8zM17 7l-4 7h8z"/></svg>',
};

function modeModelChips(mode, models){
  return `<span class="chip mode">${esc(mode||"?")}</span>` +
    (models||[]).map(m=>`<span class="chip model">${esc(m)}</span>`).join("");
}

/* ---------- primitives ---------- */
function card(kind, titleHTML, opts={}){
  const c = el("div","card "+kind+(opts.open?" open":""));
  const h = el("div","card-head");
  h.innerHTML = (kind==="text"?"":IC.chev) + `<span class="ttl">${titleHTML}</span>` +
    (opts.id?`<span class="idpill">${esc(opts.id)}</span>`:"");
  const b = el("div","body"); if (opts.bodyHTML) b.innerHTML = opts.bodyHTML;
  if (kind!=="text") h.onclick = () => c.classList.toggle("open");
  c.append(h,b); return c;
}
function codeBlock(v){
  let s;
  if (typeof v === "string"){ const t=v.trim();
    if (t && (t[0]==="{"||t[0]==="[")){ try{ s=JSON.stringify(JSON.parse(t),null,2); }catch{ s=v; } } else s=v;
  } else s = JSON.stringify(v,null,2);
  if (s.length > 40000) s = s.slice(0,40000) + "\n… (" + (s.length-40000) + " more chars)";
  return `<pre class="code">${esc(s)}</pre>`;
}
const textOf = t => Array.isArray(t) ? t.filter(b=>b&&b.type==="text").map(b=>b.text).join("") : (t||"");

function telemBlock(s){
  const tok = s.tokens||{}, out = s.output||{}, p = s.params||{};
  const d = el("details","telem"); d.open = true;
  const sum = el("summary");
  sum.innerHTML = IC.bolt +
    `<span><b>${tok.input??0}</b> <span class="m">in</span></span>` +
    `<span><b>${tok.output??0}</b> <span class="m">out</span></span>` +
    (tok.cache_read?`<span><b>${tok.cache_read}</b> <span class="m">cache</span></span>`:"") +
    (s.est_cost_usd?`<span><b>$${s.est_cost_usd.toFixed(6)}</b></span>`:"") +
    (out.finish_reason?`<span class="m">finish</span> <b>${esc(out.finish_reason)}</b>`:"") +
    (s.duration_s!=null?`<span><b>${s.duration_s}</b><span class="m">s</span></span>`:"");
  const body = el("div","telem-body");
  const bits = [];
  if (p.temperature!=null) bits.push(`<span>temperature <b>${p.temperature}</b></span>`);
  if (p.max_tokens!=null) bits.push(`<span>max_tokens <b>${p.max_tokens}</b></span>`);
  if (p.tools && p.tools.length) bits.push(`<span>tools <b>${p.tools.length}</b>: ${p.tools.map(esc).join(", ")}</span>`);
  // The comparison report ships slimmed traces (input_messages stripped to keep
  // the file small) and leaves the count behind, so say how many there were.
  const msgs = s.input_messages || [];
  const nMsgs = s.input_message_count != null ? s.input_message_count : msgs.length;
  const msgBody = msgs.length ? codeBlock(msgs)
    : `<div class="kv"><span class="m">(not included in this report — rebuild with <b>make compare ARGS=--full</b>)</span></div>`;
  body.innerHTML =
    (bits.length?`<div class="kv">${bits.join("")}</div>`:"") +
    `<details class="sub"><summary>${nMsgs} input message(s)</summary>${msgBody}</details>` +
    (out.content_blocks?`<details class="sub"><summary>raw content blocks</summary>${codeBlock(out.content_blocks)}</details>`:"");
  d.append(sum, body);
  return d;
}

function primer(){
  const d = el("details","primer");
  try { d.open = localStorage.getItem("af_primer_open") !== "0"; } catch { d.open = true; }
  d.addEventListener("toggle", () => { try { localStorage.setItem("af_primer_open", d.open?"1":"0"); } catch {} });
  d.innerHTML = `<summary>How to read this run</summary>
    <div class="pbody">
      The model does <b>not</b> search, geocode, or read files itself. Each step it does one of two things:
      reply with <b>text</b>, or emit one or more <b>tool requests</b> — a tool name plus arguments it writes as JSON.
      Your application executes each tool and returns the result; the model reads it and continues to the next step.
      <ol>
        <li><b>①</b> the arguments the <em>model</em> generated for a tool</li>
        <li><b>②</b> your app runs that tool with those arguments</li>
        <li><b>③</b> the result handed back to the model — which it uses to write the next step's arguments</li>
      </ol>
      An <span class="fromprev">↑ from step N</span> tag means that argument value came straight out of an earlier tool's result.
    </div>`;
  return d;
}

function stepHeader(n, s, txt, tcs){
  const h = el("div","stephdr");
  let desc;
  if (s.error) desc = "the model call failed";
  else if (tcs.length) desc = `the model ${txt.trim()?"replied, then ":""}requested ${tcs.length} tool${tcs.length>1?"s":""}: ` +
    tcs.map(t=>`<b>${esc(t.name)}</b>`).join(", ");
  else desc = "the model wrote its final answer";
  h.innerHTML = `<span class="n">${n}</span><span class="d">Step ${n} — ${desc}</span>
    <span class="model" style="margin-left:auto">${esc(s.model||"?")}${s.duration_s!=null?` · ${s.duration_s}s`:""}</span>`;
  return h;
}

function callout(html){ const c = el("div","callout"); c.innerHTML = html; return c; }

// which top-level arg values trace back to an earlier tool's output
function fromPrev(args, priorOut){
  const hits = {};
  if (!args || typeof args !== "object") return hits;
  Object.entries(args).forEach(([k,v]) => {
    if (v == null || v === "" || v === true || v === false) return;
    const needle = JSON.stringify(v);
    if (needle.length < 3) return;
    const src = priorOut.find(o => o.json.includes(needle));
    if (src) hits[k] = src;
  });
  return hits;
}

function argsTable(args, priorOut){
  if (!args || typeof args !== "object" || Array.isArray(args)) return codeBlock(args ?? {});
  if (!Object.keys(args).length) return `<div class="kv" style="font-size:11.5px"><span class="m">(no arguments)</span></div>`;
  const hits = fromPrev(args, priorOut);
  const rows = Object.entries(args).map(([k,v]) => {
    const src = hits[k];
    const tag = src ? `<span class="fromprev">↑ from ${esc(src.name)}${src.step?` (step ${src.step})`:""}</span>` : "";
    return `<div><b style="color:var(--violet)">${esc(k)}</b> = ${esc(JSON.stringify(v))}${tag}</div>`;
  });
  return `<div class="kv" style="flex-direction:column;gap:5px;font-size:11.5px">${rows.join("")}</div>`;
}

function roundTrip(tc, tool, priorOut){
  const rt = el("div","rt");
  const dur = tool && tool.duration_s!=null ? `<span class="dur">${tool.duration_s}s</span>` : "";
  rt.innerHTML = `<div class="rt-name">${IC.wrench} ${esc(tc.name||"tool")}
    ${dur}${tc.id?`<span class="idpill">${esc(String(tc.id).slice(0,22))}</span>`:""}</div>`;

  const m = el("div","rt-step model");
  m.innerHTML = `<div class="lab"><span class="mk">①</span> arguments the model generated</div>` +
    argsTable(tc.args, priorOut);
  rt.appendChild(m);

  rt.appendChild(Object.assign(el("div","rt-arrow"), {textContent:"↓"}));

  const r = el("div","rt-step run");
  r.innerHTML = `<div class="lab"><span class="mk">②</span> your app runs <code>${esc(tc.name||"tool")}</code></div>`;
  rt.appendChild(r);

  rt.appendChild(Object.assign(el("div","rt-arrow"), {textContent:"↓"}));

  if (!tool){
    const p = el("div","rt-step res");
    p.innerHTML = `<div class="lab"><span class="mk">③</span> result</div><div class="pending">⏳ not executed yet…</div>`;
    rt.appendChild(p);
    return rt;
  }
  const res = el("div","rt-step res" + (tool.error?" bad":""));
  res.innerHTML = `<div class="lab"><span class="mk">③</span> ${tool.error?"tool errored":"result handed back to the model"}</div>` +
    codeBlock(tool.error != null ? tool.error : tool.output);
  rt.appendChild(res);
  return rt;
}

function orphanTool(s){
  const turn = el("div","turn");
  turn.innerHTML = `<div class="turn-label">${IC.result}<span class="who">Tool run</span>
    <span class="model" style="margin-left:auto">${esc(s.name||"tool")}${s.duration_s!=null?` · ${s.duration_s}s`:""}</span></div>`;
  const rt = el("div","rt");
  rt.innerHTML = `<div class="rt-name">${IC.wrench} ${esc(s.name||"tool")}</div>`;
  const a = el("div","rt-step model");
  a.innerHTML = `<div class="lab"><span class="mk">in</span> input</div>` + codeBlock(s.input);
  const o = el("div","rt-step res"+(s.error?" bad":""));
  o.innerHTML = `<div class="lab"><span class="mk">out</span> ${s.error?"error":"output"}</div>` + codeBlock(s.error??s.output);
  rt.append(a,o); turn.appendChild(rt); return turn;
}

/* ---------- the run, as numbered steps ---------- */
/* `feed` is the container to fill; `d` is a trace document as written by
   src/utils/telemetry.py. Options: {primer:true} prepends the explainer,
   {question:true} prepends the user's query bubble. */
function renderConversationInto(feed, d, opts={}){
  if (opts.primer !== false) feed.appendChild(primer());

  if (opts.question !== false){
    const u = el("div","turn");
    u.innerHTML = `<div class="turn-label">${IC.msg}<span class="who">You asked</span><span>${clock(d.started_at)}</span></div>`;
    u.appendChild(el("div","bubble", d.query || "(no query)"));
    feed.appendChild(u);
  }

  const spans = d.spans || [];
  const toolById = {}, toolQ = {};
  spans.forEach(s => { if (s.type === "tool"){
    if (s.tool_call_id) toolById[s.tool_call_id] = s;
    (toolQ[s.name] = toolQ[s.name] || []).push(s);
  }});
  const used = new Set();
  const priorOut = [];   // [{step, name, json}] for every tool output seen so far
  let step = 0;

  spans.forEach(s => {
    if (s.type === "tool"){
      if (used.has(s)) return;              // already shown inside its LLM step
      feed.appendChild(orphanTool(s));      // safety net: unmatched tool run
      if (s.output != null) priorOut.push({step, name:s.name, json:JSON.stringify(s.output)});
      return;
    }
    step++;
    const out = s.output || {}, txt = textOf(out.text), tcs = out.tool_calls || [];
    const turn = el("div","turn");
    turn.appendChild(stepHeader(step, s, txt, tcs));

    if (s.error){ turn.appendChild(card("err","LLM error",{open:true,bodyHTML:codeBlock(s.error)})); }
    else {
      if (txt.trim())
        turn.appendChild(card("text","What the model said",{open:true,bodyHTML:`<div class="msgtext">${mdInline(txt)}</div>`}));

      if (tcs.length){
        turn.appendChild(callout(
          `The model can't run tools itself. It wrote <b>${tcs.length===1?"one request":tcs.length+" requests"}</b> ` +
          `(a tool name + arguments as JSON); your app runs ${tcs.length===1?"it":"them"} and feeds the result back.`));
        tcs.forEach(tc => {
          let tool = tc.id && toolById[tc.id];
          if (!tool){ const q = toolQ[tc.name] || []; tool = q.find(x => !used.has(x)); }
          if (tool) used.add(tool);
          turn.appendChild(roundTrip(tc, tool, priorOut));
          if (tool && tool.output != null) priorOut.push({step, name:tc.name, json:JSON.stringify(tool.output)});
        });
      }
      turn.appendChild(telemBlock(s));
    }
    feed.appendChild(turn);
  });
  return feed;
}

/* The metadata line above a run: mode/model chips, call counts, tokens, cost. */
function traceMetaline(d){
  const t = d.totals || {};
  return `<div class="metaline">
      ${modeModelChips(d.mode, d.models)}
      ${d.session_id?`<span class="chip">sess ${esc(String(d.session_id).slice(0,12))}</span>`:""}
      <span>·</span><span>${t.llm_calls||0} LLM · ${t.tool_calls||0} tool</span>
      <span>·</span><span>${fmt(t.input_tokens)} in / ${fmt(t.output_tokens)} out${t.cache_read_tokens?` / ${fmt(t.cache_read_tokens)} cache`:""}</span>
      ${t.est_cost_usd?`<span>·</span><span>$${t.est_cost_usd.toFixed(4)}</span>`:""}
      <span>·</span><span>${d.ended_at? (t.duration_s??"?")+"s" : "running…"}</span>
      ${d.error?`<span class="chip bad">trace error</span>`:""}
    </div>`;
}
