'use strict';

/* ============================================================
   文澜 WenLan — frontend application
   ============================================================ */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const icon = (name, cls = '') => `<svg class="icon ${cls}"><use href="#i-${name}"></use></svg>`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PAGE_META = {
  overview: { eyebrow: 'HOME', title: '总览' },
  chat: { eyebrow: 'LAB Q&A', title: '科研问答' },
  library: { eyebrow: 'KNOWLEDGE BASE', title: '文献库' },
  discover: { eyebrow: 'PAPER SEARCH', title: '找论文' },
  evaluation: { eyebrow: 'EVALUATION LAB', title: '评测实验' },
};

const TOOL_LABELS = {
  search_papers: '检索论文',
  get_paper_detail: '论文详情',
  search_library: '原文检索',
  list_library_papers: '列出文献',
  search_online_papers: '在线检索',
};

const state = {
  info: null,
  page: 'overview',
  pendingQuestion: null,
  papers: [],
  conversations: [],
  currentConversationId: null,
  conversation: null,
  messages: [],
  citations: [],
  trace: [],
  streaming: false,
  abortController: null,
  discoverItems: [],
  discoverQuery: '',
  evalRuns: [],
  activeRunId: null,
  activeRun: null,
  pollTimer: null,
  dataset: null,
  theme: localStorage.getItem('wenlan-theme') || 'dark',
  libraryQuery: '',
};

/* ------------------------------------------------------------------ */
/* utilities                                                          */
/* ------------------------------------------------------------------ */
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function fmtNum(value) {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return '—';
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 10000) return (n / 1000).toFixed(1) + 'k';
  return n.toLocaleString('zh-CN');
}

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return (Number(value) * 100).toFixed(digits) + '%';
}

function scoreText(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : '—';
}

function fmtDate(value, withTime = false) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '—';
  const opts = withTime
    ? { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }
    : { year: 'numeric', month: '2-digit', day: '2-digit' };
  return new Intl.DateTimeFormat('zh-CN', opts).format(d);
}

function shortAuthors(authors = [], max = 3) {
  if (!authors || !authors.length) return '作者未知';
  return authors.slice(0, max).join('、') + (authors.length > max ? ' 等' : '');
}

function truncate(value, len = 200) {
  const s = String(value ?? '');
  return s.length > len ? s.slice(0, len).trim() + '…' : s;
}

function safeUrl(value) {
  const url = String(value || '').trim();
  return /^https?:\/\//i.test(url) ? url : '';
}

async function api(path, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}) } };
  if (opts.body && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
    if (typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  const ctype = res.headers.get('content-type') || '';
  let data;
  if (ctype.includes('application/json')) data = await res.json();
  else data = await res.text();
  if (!res.ok) {
    const detail = data && typeof data === 'object' ? data.detail || data.message : data;
    throw new Error(detail || `请求失败 (${res.status})`);
  }
  return data;
}

function toast(message, type = 'info', timeout = 4200) {
  const stack = $('#toast-stack');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const ic = type === 'success' ? 'check' : type === 'error' ? 'close' : 'spark';
  el.innerHTML = `${icon(ic)}<div>${escapeHtml(message)}</div>`;
  stack.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(8px)';
    setTimeout(() => el.remove(), 220);
  }, timeout);
}

function showLoading(text = '正在处理…') {
  $('#loading-text').textContent = text;
  $('#loading-overlay').classList.remove('hidden');
}
function hideLoading() {
  $('#loading-overlay').classList.add('hidden');
}

function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}
function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}
function closeAllModals() {
  $$('.modal-layer').forEach((m) => m.classList.add('hidden'));
}

/* ------------------------------------------------------------------ */
/* markdown                                                           */
/* ------------------------------------------------------------------ */
function formatInline(text) {
  let out = text;
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {
    return `<a href="${url}" target="_blank" rel="noopener">${label}</a>`;
  });
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,，。]|$)/g, '$1<em>$2</em>');
  out = out.replace(/\[(\d{1,2})\]/g, (m, n) => {
    return `<button class="cite-chip" data-cite="${n}" title="查看引用 ${n}">${n}</button>`;
  });
  return out;
}

function renderMarkdown(input) {
  let text = escapeHtml(input || '');
  const codeBlocks = [];
  text = text.replace(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${code.replace(/\n$/, '')}</code></pre>`);
    return `\n\n@@CODEBLOCK${idx}@@\n\n`;
  });
  const lines = text.split('\n');
  const blocks = [];
  let i = 0;
  const inline = (s) => formatInline(s);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    const codeMatch = line.match(/^@@CODEBLOCK(\d+)@@$/);
    if (codeMatch) {
      blocks.push(codeBlocks[Number(codeMatch[1])]);
      i += 1;
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = Math.min(4, h[1].length + 1);
      blocks.push(`<h${level}>${inline(h[2])}</h${level}>`);
      i += 1;
      continue;
    }
    if (/^---+$/.test(line.trim())) {
      blocks.push('<hr />');
      i += 1;
      continue;
    }
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quote.push(inline(lines[i].replace(/^>\s?/, '')));
        i += 1;
      }
      blocks.push(`<blockquote>${quote.join('<br/>')}</blockquote>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ''))}</li>`);
        i += 1;
      }
      blocks.push(`<ul>${items.join('')}</ul>`);
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*\d+[.)]\s+/, ''))}</li>`);
        i += 1;
      }
      blocks.push(`<ol>${items.join('')}</ol>`);
      continue;
    }
    const para = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|>\s?|\s*[-*]\s+|\s*\d+[.)]\s+|@@CODEBLOCK)/.test(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push(`<p>${para.map(inline).join('<br/>')}</p>`);
  }
  return blocks.join('') || '<p></p>';
}

/* ------------------------------------------------------------------ */
/* theme + system                                                     */
/* ------------------------------------------------------------------ */
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('wenlan-theme', theme);
  const use = $('#theme-toggle use');
  if (use) use.setAttribute('href', theme === 'dark' ? '#i-sun' : '#i-moon');
}

function updateSystemUI() {
  const info = state.info;
  if (!info) return;
  $('#system-card').classList.add('ready');
  const demo = info.demo_mode;
  $('#system-mode').textContent = demo ? '离线演示模式' : '模型已连接';
  $('#system-detail').textContent = demo ? '配置 API Key 后可启用真实 Tool Calling' : `${info.llm.label}`;
  $('#status-dot').className = `status-dot ${info.database.ok ? '' : 'bad'}`;
  $('#system-db').textContent = info.database.dialect === 'postgresql'
    ? `PG ${info.database.server_version || ''} · vector ${info.database.pgvector_version || ''}`
    : 'SQLite fallback';
  $('#system-embed').textContent = info.embedding.model || 'local';
  $('#demo-pill').classList.toggle('hidden', !demo);
  $('#top-status').innerHTML = `<span class="status-dot ${info.database.ok ? '' : 'bad'}"></span><span>${info.database.ok ? '就绪' : '异常'}</span>`;
  $('#about-version').textContent = 'v' + info.version;
  $('#nav-paper-count').textContent = fmtNum(info.counts.papers);
  $('#chat-model-label').textContent = info.llm.label || '—';
  $('#chat-embed-label').textContent = info.embedding.label || info.embedding.model || '—';
  $('#upload-limit').textContent = info.limits.max_upload_mb;
  const stats = {
    papers: info.counts.papers,
    chunks: info.counts.chunks,
    conversations: info.counts.conversations,
    evaluations: info.counts.evaluations,
  };
  $$('[data-stat]').forEach((el) => {
    if (stats[el.dataset.stat] !== undefined) el.textContent = fmtNum(stats[el.dataset.stat]);
  });
  renderSettingsStatus();
}

async function loadInfo() {
  try {
    const info = await api('/api/system/info');
    state.info = info;
    updateSystemUI();
    return info;
  } catch (err) {
    $('#system-mode').textContent = '连接失败';
    $('#system-detail').textContent = err.message;
    $('#status-dot').className = 'status-dot bad';
    $('#top-status').innerHTML = '<span class="status-dot bad"></span><span>离线</span>';
    throw err;
  }
}

function renderSettingsStatus() {
  const info = state.info;
  if (!info) return;
  const cards = [
    {
      title: '数据库 / 向量',
      badge: info.database.ok ? '在线' : '异常',
      ok: info.database.ok,
      lines: `${info.database.dialect} ${info.database.server_version || ''}<br/>pgvector ${info.database.pgvector_version || '—'} · ${info.database.embedded ? 'embedded' : 'external'}`,
    },
    {
      title: '对话模型',
      badge: info.demo_mode ? '演示' : '已配置',
      ok: !info.demo_mode,
      lines: `${escapeHtml(info.llm.label || '—')}<br/><code>${info.demo_mode ? 'LLM_PROVIDER=mock' : 'OpenAI-compatible'}</code>`,
    },
    {
      title: '向量模型',
      badge: info.embedding.local ? '本地' : '远端',
      ok: true,
      lines: `${escapeHtml(info.embedding.label || '—')}<br/>${info.embedding.dimension} dimensions`,
    },
    {
      title: 'RAG 参数',
      badge: '生效中',
      ok: true,
      lines: `chunk ${info.limits.chunk_size} · overlap ${info.limits.chunk_overlap}<br/>top-k ${info.limits.top_k} · max ${info.limits.max_pdf_pages} pages`,
    },
  ];
  const el = $('#settings-status');
  if (!el) return;
  el.innerHTML = cards.map((c) => `
    <div class="setting-card">
      <div class="sc-head"><span>${c.title}</span><span class="badge ${c.ok ? 'success' : 'warn'}">${c.badge}</span></div>
      <p>${c.lines}</p>
    </div>`).join('');
  const providers = [
    ['DeepSeek', 'DEEPSEEK_API_KEY', 'deepseek-chat', '性价比高，工具调用稳定'],
    ['通义千问 / 百炼', 'DASHSCOPE_API_KEY', 'qwen-plus / text-embedding-v4', 'LLM 与 Embedding 一站式'],
    ['硅基流动', 'SILICONFLOW_API_KEY', 'BAAI/bge-m3', '推荐作为 Embedding 服务'],
    ['Kimi / 月之暗面', 'MOONSHOT_API_KEY', 'moonshot-v1-8k', '长文本阅读'],
    ['智谱 GLM', 'ZHIPU_API_KEY', 'glm-4-plus', '中文科研分析'],
  ];
  const grid = $('#provider-grid');
  if (grid) {
    grid.innerHTML = providers.map(([name, key, model, desc]) => `
      <div class="provider-card">
        <strong>${name}<span>可选</span></strong>
        <p>${desc}</p>
        <p style="margin-top:5px"><em>${key}</em> · ${model}</p>
      </div>`).join('');
  }
}

/* ------------------------------------------------------------------ */
/* routing                                                            */
/* ------------------------------------------------------------------ */
function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [route, ...rest] = raw.split('/');
  return { route: PAGE_META[route] ? route : 'overview', param: decodeURIComponent(rest.join('/') || '') };
}

function navigate(route, param = '') {
  const hash = `#/${route}${param ? '/' + encodeURIComponent(param) : ''}`;
  if (location.hash === hash) {
    void renderRoute();
  } else {
    location.hash = hash;
  }
}

async function renderRoute() {
  const { route, param } = parseHash();
  state.page = route;
  $$('.page').forEach((p) => p.classList.toggle('active', p.dataset.page === route));
  $$('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.route === route));
  const meta = PAGE_META[route];
  $('#page-title').textContent = meta.title;
  $('#page-eyebrow').textContent = meta.eyebrow;
  $('#app-content').classList.toggle('chat-mode', route === 'chat');
  $('#sidebar').classList.remove('open');

  if (route === 'overview') void loadOverview();
  if (route === 'chat') await enterChat(param);
  if (route === 'library') void loadLibrary();
  if (route === 'discover') void loadDiscoveryDefaults();
  if (route === 'evaluation') void loadEvaluationPage();
}

/* ------------------------------------------------------------------ */
/* overview                                                           */
/* ------------------------------------------------------------------ */
async function loadOverview() {
  void loadInfo().catch(() => {});
  try {
    const data = await api('/api/papers?limit=5');
    state.papers = data.items || [];
    renderRecentPapers();
  } catch (err) {
    $('#recent-papers').innerHTML = `<div class="empty-inline">${escapeHtml(err.message)}</div>`;
  }
  try {
    const runs = await api('/api/evaluations/runs?limit=1');
    const run = runs.items?.[0];
    renderOverviewEval(run);
  } catch {
    renderOverviewEval(null);
  }
}

function renderRecentPapers() {
  const box = $('#recent-papers');
  if (!state.papers.length) {
    box.innerHTML = `<div class="empty-inline">文献库为空。点击左侧“示例文献库”即可开始。</div>`;
    return;
  }
  box.innerHTML = state.papers.map((p) => `
    <div class="recent-item" data-paper="${p.id}">
      <div class="recent-icon">${escapeHtml((p.source || 'PDF').slice(0, 6).toUpperCase())}</div>
      <div class="recent-main">
        <div class="recent-title">${escapeHtml(p.title)}</div>
        <div class="recent-meta">${escapeHtml(shortAuthors(p.authors))} · ${p.year || '年份未知'} · ${escapeHtml(p.venue || '')}</div>
      </div>
      <span class="recent-chunks">${p.chunk_count || 0} chunks</span>
    </div>`).join('');
}

function renderOverviewEval(run) {
  const el = $('#overview-eval-body');
  if (!run) {
    el.innerHTML = `<div class="empty-inline">还没有评测记录，运行一次评测来量化系统效果。</div>`;
    return;
  }
  const m = run.metrics || {};
  const items = [
    ['Recall@K', m.retrieval_recall_at_k, '检索命中率'],
    ['Tool Accuracy', m.tool_calling_accuracy, '工具选择'],
    ['Task Success', m.task_success_rate, '端到端任务'],
  ];
  el.innerHTML = `
    <div class="eval-mini-head"><span>${escapeHtml(run.name)} · ${fmtDate(run.created_at, true)}</span><span class="badge ${run.status === 'completed' ? 'success' : run.status === 'failed' ? 'danger' : 'warn'}">${run.status}</span></div>
    <div class="eval-mini">
      ${items.map(([label, val, sub]) => `<div class="eval-mini-item"><span>${label}</span><strong>${pct(val)}</strong><small>${sub}</small></div>`).join('')}
    </div>`;
}

/* ------------------------------------------------------------------ */
/* chat                                                               */
/* ------------------------------------------------------------------ */
function setChatHeader() {
  const conv = state.conversation;
  $('#chat-title').textContent = conv?.title || '新的科研对话';
  const llm = state.info?.llm;
  $('#chat-subtitle').textContent = state.info?.demo_mode
    ? '离线演示模型 · 基于本地文献库的引用式回答'
    : `${llm?.label || '模型'} · 基于本地文献库的引用式回答`;
  $('#chat-mode-badge').textContent = state.info?.demo_mode ? 'DEMO' : 'RAG';
}

async function loadConversations() {
  try {
    const data = await api('/api/conversations?limit=50');
    state.conversations = data.items || [];
    renderConversationList();
  } catch (err) {
    $('#conversation-list').innerHTML = `<div class="empty-inline">${escapeHtml(err.message)}</div>`;
  }
}

function renderConversationList() {
  const box = $('#conversation-list');
  if (!state.conversations.length) {
    box.innerHTML = `<div class="empty-inline" style="padding:30px 8px;font-size:10px">还没有研究对话</div>`;
    return;
  }
  box.innerHTML = state.conversations.map((c) => `
    <button class="conv-item ${c.id === state.currentConversationId ? 'active' : ''}" data-conv="${c.id}">
      <button class="conv-del" data-del-conv="${c.id}" title="删除对话">×</button>
      <strong>${escapeHtml(c.title)}</strong>
      <span>${fmtDate(c.updated_at, true)} · ${c.message_count || 0} 条</span>
    </button>`).join('');
}

async function enterChat(conversationId) {
  setChatHeader();
  await loadConversations();
  if (state.pendingQuestion) {
    const q = state.pendingQuestion;
    state.pendingQuestion = null;
    startNewChat();
    await sleep(30);
    void sendChat(q);
    return;
  }
  if (conversationId && conversationId !== state.currentConversationId) {
    await loadConversation(conversationId);
  } else if (!conversationId && state.currentConversationId) {
    // keep current conversation in memory when switching pages
    renderChat();
  } else if (!conversationId) {
    startNewChat(false);
  }
  setChatHeader();
}

async function loadConversation(id) {
  try {
    const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
    state.conversation = data.conversation;
    state.currentConversationId = id;
    state.messages = data.messages || [];
    state.citations = [];
    state.trace = [];
    for (let i = state.messages.length - 1; i >= 0; i -= 1) {
      const m = state.messages[i];
      if (m.role === 'assistant' && (m.citations?.length || m.tool_trace?.length)) {
        state.citations = m.citations || [];
        state.trace = m.tool_trace || [];
        break;
      }
    }
    renderChat();
    setChatHeader();
    if (location.hash !== `#/chat/${id}`) history.replaceState(null, '', `#/chat/${id}`);
  } catch (err) {
    toast(err.message, 'error');
    startNewChat(false);
  }
}

function startNewChat(updateHash = true) {
  state.currentConversationId = null;
  state.conversation = null;
  state.messages = [];
  state.citations = [];
  state.trace = [];
  state.streaming = false;
  renderChat();
  setChatHeader();
  if (updateHash) history.replaceState(null, '', '#/chat');
  setTimeout(() => $('#chat-input')?.focus(), 60);
}

function renderChat() {
  const list = $('#message-list');
  const empty = $('#chat-empty');
  empty.classList.toggle('hidden', state.messages.length > 0);
  list.innerHTML = '';
  state.messages.forEach((msg) => list.appendChild(messageNode(msg)));
  renderEvidence(state.citations);
  renderTrace(state.trace);
  scrollChatToBottom(true);
}

function messageNode(msg) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${msg.role}`;
  wrap.dataset.msgId = msg.id || '';
  const isUser = msg.role === 'user';
  const avatar = isUser ? '你' : '澜';
  const content = isUser
    ? `<p>${escapeHtml(msg.content).replaceAll('\n', '<br/>')}</p>`
    : renderMarkdown(msg.content || '');
  const time = msg.created_at ? new Date(msg.created_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) : '';
  const toolTrace = msg.tool_trace || [];
  const toolPills = toolTrace.map((t) => `
    <span class="tool-pill ${t.status === 'running' ? 'running' : ''}">${icon('tool')}${escapeHtml(TOOL_LABELS[t.name] || t.name)}${t.latency_ms ? ` · ${t.latency_ms}ms` : ''}</span>`).join('');
  const typing = msg._streaming && !msg.content ? '<span class="typing"><span></span><span></span><span></span></span>' : '';
  wrap.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-meta"><span>${isUser ? '你' : '文澜'}</span>${time ? `<span>·</span><span>${time}</span>` : ''}${msg.latency_ms ? `<span>·</span><span>${(msg.latency_ms / 1000).toFixed(1)}s</span>` : ''}</div>
      <div class="msg-bubble">${content}${typing}</div>
      ${toolPills ? `<div class="msg-tools">${toolPills}</div>` : ''}
      ${!isUser && msg.content ? `<div class="msg-actions"><button class="msg-action" data-copy-msg="${wrap.dataset.msgId}">${icon('file')}复制</button></div>` : ''}
    </div>`;
  return wrap;
}

function updateStreamingMessage(msg) {
  const node = $(`[data-msg-id="${msg.id}"]`);
  if (!node) {
    renderChat();
    return;
  }
  const bubble = $('.msg-bubble', node);
  if (bubble) bubble.innerHTML = renderMarkdown(msg.content || '') + (msg._streaming && !msg.content ? '<span class="typing"><span></span><span></span><span></span></span>' : '');
  const tools = $('.msg-tools', node);
  const pills = (msg.tool_trace || []).map((t) => `
    <span class="tool-pill ${t.status === 'running' ? 'running' : ''}">${icon('tool')}${escapeHtml(TOOL_LABELS[t.name] || t.name)}${t.latency_ms ? ` · ${t.latency_ms}ms` : ''}</span>`).join('');
  if (tools) tools.innerHTML = pills;
  else if (pills) {
    const div = document.createElement('div');
    div.className = 'msg-tools';
    div.innerHTML = pills;
    $('.msg-body', node)?.appendChild(div);
  }
  scrollChatToBottom();
}

function scrollChatToBottom(force = false) {
  const box = $('#chat-scroll');
  if (!box) return;
  const near = box.scrollHeight - box.scrollTop - box.clientHeight < 180;
  if (force || near) box.scrollTop = box.scrollHeight;
}

function renderEvidence(citations) {
  const body = $('#evidence-body');
  const count = $('#evidence-count');
  const list = citations || [];
  count.textContent = list.length;
  if (!list.length) {
    body.innerHTML = `<div class="evidence-empty">${icon('quote')}<p>回答中引用的原文片段会显示在这里，可直达页码与 PDF。</p></div>`;
    return;
  }
  body.innerHTML = list.map((c) => `
    <div class="citation-card ${c.cited ? 'cited' : ''}" data-cite-card="${c.index}" data-paper="${c.paper_id || ''}">
      <div class="citation-top">
        <span class="citation-index">${c.index}</span>
        <div class="citation-title">${escapeHtml(c.paper_title || '未知论文')}</div>
      </div>
      <div class="citation-meta">${escapeHtml(c.section || '正文')} · p.${c.page_start || '?'}${c.page_end && c.page_end !== c.page_start ? '–' + c.page_end : ''}${c.cited ? ' · 已引用' : ''}</div>
      <div class="citation-snippet">${escapeHtml(c.snippet || '')}</div>
      <div class="citation-actions">
        <span class="citation-score">score ${scoreText(c.score)}</span>
        <button class="citation-open" data-open-paper="${c.paper_id || ''}">查看论文 ${icon('arrow')}</button>
      </div>
    </div>`).join('');
}

function renderTrace(trace) {
  const box = $('#trace-list');
  const list = trace || [];
  $('#trace-count').textContent = `${list.length} calls`;
  if (!list.length) {
    box.innerHTML = `<p class="muted-small">本轮工具调用将在这里展示。</p>`;
    return;
  }
  box.innerHTML = list.map((t) => `
    <div class="trace-item">
      ${icon('tool')}
      <span>${escapeHtml(TOOL_LABELS[t.name] || t.name)}</span>
      <span class="t-status ${t.status}">${t.status === 'running' ? 'running…' : t.status === 'error' ? 'error' : (t.latency_ms || 0) + 'ms'}</span>
    </div>`).join('');
}

async function sendChat(message) {
  if (state.streaming) return;
  const text = String(message || '').trim();
  if (!text) return;
  state.streaming = true;
  $('#send-btn').disabled = true;
  $('#chat-empty').classList.add('hidden');

  const userMsg = {
    id: 'u' + Date.now(),
    role: 'user',
    content: text,
    citations: [],
    tool_trace: [],
    created_at: new Date().toISOString(),
  };
  const draft = {
    id: 'a' + Date.now(),
    role: 'assistant',
    content: '',
    citations: [],
    tool_trace: [],
    _streaming: true,
    created_at: new Date().toISOString(),
  };
  state.messages.push(userMsg, draft);
  state.citations = [];
  state.trace = [];
  renderChat();
  scrollChatToBottom(true);
  $('#chat-input').value = '';
  autoResizeTextarea($('#chat-input'));

  state.abortController = new AbortController();
  let finished = false;
  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        conversation_id: state.currentConversationId,
        top_k: 6,
        prompt_variant: $('#chat-prompt-variant')?.value || 'citation_strict',
      }),
      signal: state.abortController.signal,
    });
    if (!res.ok) {
      let detail = `请求失败 (${res.status})`;
      try { const body = await res.json(); detail = body.detail || detail; } catch {}
      throw new Error(detail);
    }
    await consumeSSE(res, (event, data) => {
      if (event === 'start') {
        state.currentConversationId = data.conversation_id;
        if (!state.conversation || state.conversation.id !== data.conversation_id) {
          state.conversation = { id: data.conversation_id, title: '新的科研对话' };
        }
        history.replaceState(null, '', `#/chat/${data.conversation_id}`);
        setChatHeader();
      } else if (event === 'tool_start') {
        state.trace.push({ name: data.name, status: 'running', arguments: data.arguments, latency_ms: 0 });
        draft.tool_trace = state.trace;
        renderTrace(state.trace);
        updateStreamingMessage(draft);
      } else if (event === 'tool_result') {
        const item = [...state.trace].reverse().find((t) => t.name === data.name && t.status === 'running')
          || [...state.trace].reverse().find((t) => t.name === data.name);
        if (item) {
          item.status = data.status;
          item.latency_ms = data.latency_ms;
          item.summary = data.summary;
        }
        draft.tool_trace = state.trace;
        renderTrace(state.trace);
        updateStreamingMessage(draft);
      } else if (event === 'token') {
        draft.content += data.text || '';
        updateStreamingMessage(draft);
      } else if (event === 'citations') {
        state.citations = data.citations || [];
        draft.citations = state.citations;
        renderEvidence(state.citations);
      } else if (event === 'done') {
        finished = true;
        state.currentConversationId = data.conversation_id;
        if (state.conversation) {
          state.conversation.title = data.conversation_title || state.conversation.title;
          if (!state.conversation.title) state.conversation.title = '新的科研对话';
        }
        draft.id = data.message_id || draft.id;
        draft.content = draft.content || '';
        draft.citations = data.citations || state.citations;
        draft.tool_trace = data.tool_trace || state.trace;
        draft.latency_ms = data.latency_ms;
        draft.usage = data.usage;
        state.citations = draft.citations;
        state.trace = draft.tool_trace;
        state.conversation = {
          ...(state.conversation || {}),
          id: data.conversation_id,
          title: data.conversation_title || state.conversation?.title || '新的科研对话',
        };
        history.replaceState(null, '', `#/chat/${data.conversation_id}`);
        setChatHeader();
        renderEvidence(state.citations);
        renderTrace(state.trace);
        renderChat();
        void loadConversations();
        void refreshCounts();
      } else if (event === 'error') {
        draft.content = `> ⚠️ ${data.message || '模型调用失败'}\n\n${data.hint ? `> ${data.hint}` : ''}`;
        draft._streaming = false;
        toast(data.message || '对话失败', 'error', 7000);
      }
    });
    if (!finished && !draft.content) {
      draft.content = '没有收到模型的回答。请检查模型配置，或稍后重试。';
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      draft.content = draft.content || '已停止生成。';
    } else {
      draft.content = `> ⚠️ ${err.message}`;
      toast(err.message, 'error', 7000);
    }
  } finally {
    state.streaming = false;
    state.abortController = null;
    draft._streaming = false;
    $('#send-btn').disabled = false;
    renderChat();
    void loadInfo().catch(() => {});
  }
}

async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  const dispatch = (block) => {
    if (!block.trim()) return;
    let event = 'message';
    const dataLines = [];
    block.split('\n').forEach((line) => {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    });
    if (!dataLines.length) return;
    try {
      onEvent(event, JSON.parse(dataLines.join('\n')));
    } catch {
      // ignore malformed chunks
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, idx).replace(/\r/g, '');
      buffer = buffer.slice(idx + 2);
      dispatch(block);
    }
  }
  buffer += decoder.decode();
  dispatch(buffer.replace(/\r/g, ''));
}

function autoResizeTextarea(el) {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(130, el.scrollHeight) + 'px';
}

/* ------------------------------------------------------------------ */
/* library                                                            */
/* ------------------------------------------------------------------ */
async function loadLibrary() {
  try {
    const data = await api(`/api/papers?limit=200${state.libraryQuery ? `&q=${encodeURIComponent(state.libraryQuery)}` : ''}`);
    state.papers = data.items || [];
    renderLibrary(data);
    void refreshCounts();
  } catch (err) {
    $('#library-tbody').innerHTML = `<tr><td colspan="6"><div class="empty-inline">${escapeHtml(err.message)}</div></td></tr>`;
  }
}

function renderLibrary(data) {
  const tbody = $('#library-tbody');
  const empty = $('#library-empty');
  const count = $('#library-count');
  count.textContent = `${data.total || 0} 篇文献`;
  if (!state.papers.length) {
    tbody.innerHTML = '';
    empty.classList.remove('hidden');
    $('.table-wrap').classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  $('.table-wrap').classList.remove('hidden');
  const statusMap = {
    indexed: ['success', '已索引'],
    parsed: ['warn', '已解析'],
    parse_failed: ['danger', '解析失败'],
    metadata: ['neutral', '仅元数据'],
  };
  tbody.innerHTML = state.papers.map((p) => {
    const st = statusMap[p.parse_status] || ['neutral', p.parse_status || '未知'];
    return `<tr>
      <td><div class="paper-cell"><div class="paper-mini">PDF</div><div><strong title="${escapeHtml(p.title)}">${escapeHtml(p.title)}</strong><small>${escapeHtml((p.keywords || []).slice(0, 4).join(' · ') || p.doi || '')}</small></div></div></td>
      <td><div style="font-size:10px;color:var(--text-soft)">${p.year || '—'} · ${escapeHtml(p.venue || '—')}</div><div style="font-size:8.5px;color:var(--muted-2)">${escapeHtml(p.source || '')}</div></td>
      <td><span class="badge ${st[0]}"><span class="dot"></span>${st[1]}</span></td>
      <td style="font-family:var(--font-mono);font-size:10px">${p.chunk_count || 0}</td>
      <td style="font-family:var(--font-mono);font-size:10px">${p.page_count || 0}</td>
      <td><div class="row-actions">
        <button class="row-action" data-action="view" data-id="${p.id}" title="详情">${icon('file')}</button>
        <button class="row-action" data-action="pdf" data-id="${p.id}" title="PDF">${icon('external')}</button>
        <button class="row-action" data-action="reindex" data-id="${p.id}" title="重建索引">${icon('refresh')}</button>
        <button class="row-action danger" data-action="delete" data-id="${p.id}" title="删除">${icon('trash')}</button>
      </div></td>
    </tr>`;
  }).join('');
}

async function handleUpload(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    toast('请选择 PDF 文件', 'error');
    return;
  }
  const maxMb = state.info?.limits?.max_upload_mb || 30;
  if (file.size > maxMb * 1024 * 1024) {
    toast(`PDF 超过 ${maxMb}MB 限制`, 'error');
    return;
  }
  $('#upload-progress').classList.remove('hidden');
  $('#upload-bar').style.width = '6%';
  $('#upload-status').textContent = `正在上传 ${file.name}…`;
  const form = new FormData();
  form.append('file', file);
  try {
    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/papers/upload');
      xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const p = 6 + (e.loaded / e.total) * 44;
        $('#upload-bar').style.width = p + '%';
        $('#upload-status').textContent = `上传中 ${Math.round((e.loaded / e.total) * 100)}% · 随后解析并向量化…`;
      };
      xhr.onload = () => {
        let body;
        try { body = JSON.parse(xhr.responseText); } catch { body = { detail: xhr.responseText }; }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.detail || `上传失败 (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error('网络错误'));
      xhr.send(form);
    });
    $('#upload-bar').style.width = '100%';
    $('#upload-status').textContent = data.duplicate ? '文件已存在' : `索引完成 · ${data.chunk_count} chunks`;
    toast(data.message || '上传成功', 'success');
    await sleep(400);
    await loadLibrary();
    await loadInfo().catch(() => {});
  } catch (err) {
    $('#upload-bar').style.width = '0';
    $('#upload-status').textContent = '处理失败';
    toast(err.message, 'error', 7000);
  } finally {
    setTimeout(() => $('#upload-progress').classList.add('hidden'), 1800);
    $('#pdf-input').value = '';
  }
}

async function seedLibrary(download = false) {
  if (!state.info) await loadInfo();
  if (state.info?.counts?.papers > 0) {
    if (!window.confirm('文献库已有内容。仍要导入示例论文吗？已存在的论文会自动跳过。')) return;
  }
  showLoading('正在导入示例论文，并建立 pgvector 索引…');
  try {
    const result = await api(`/api/library/seed?download=${download ? 'true' : 'false'}`, { method: 'POST' });
    hideLoading();
    const msg = result.created > 0
      ? `已导入 ${result.created} 篇论文${result.skipped ? `，跳过 ${result.skipped} 篇` : ''}`
      : '示例文献库已存在，无需重复导入';
    toast(msg, result.failed ? 'error' : 'success', 6000);
    await loadLibrary();
    await loadInfo().catch(() => {});
  } catch (err) {
    hideLoading();
    toast(err.message, 'error', 7000);
  }
}

async function openPaperModal(paperId) {
  if (!paperId) return;
  openModal('paper-modal');
  $('#paper-modal-title').textContent = '加载中…';
  $('#paper-modal-body').innerHTML = '<div class="skeleton line"></div><div class="skeleton line"></div>';
  try {
    const p = await api(`/api/papers/${paperId}`);
    $('#paper-modal-title').textContent = '论文详情';
    const statusMap = {
      indexed: ['success', '已索引'],
      parsed: ['warn', '已解析'],
      parse_failed: ['danger', '解析失败'],
      metadata: ['neutral', '仅元数据'],
    };
    const st = statusMap[p.parse_status] || ['neutral', p.parse_status];
    $('#paper-modal-body').innerHTML = `
      <h2 class="pm-title">${escapeHtml(p.title)}</h2>
      <div class="pm-meta">${escapeHtml(shortAuthors(p.authors, 8))}</div>
      <div class="pm-meta">${p.year || '年份未知'} · ${escapeHtml(p.venue || '来源未知')} · ${escapeHtml(p.source || '')}</div>
      <div class="pm-tags">
        <span class="badge ${st[0]}"><span class="dot"></span>${st[1]}</span>
        <span class="badge neutral">${p.chunk_count || 0} chunks</span>
        <span class="badge neutral">${p.page_count || 0} pages</span>
        ${p.doi ? `<span class="badge neutral">DOI ${escapeHtml(p.doi)}</span>` : ''}
      </div>
      <div class="pm-actions">
        ${p.has_pdf ? `<button class="btn primary small" data-pdf="${p.id}">${icon('external')}打开 PDF</button>` : ''}
        ${safeUrl(p.url) ? `<a class="btn ghost small" href="${escapeHtml(safeUrl(p.url))}" target="_blank" rel="noopener">${icon('external')}原文链接</a>` : ''}
        <button class="btn ghost small" data-reindex-one="${p.id}">${icon('refresh')}重建索引</button>
      </div>
      <div class="pm-section"><h4>摘要</h4><p>${escapeHtml(p.abstract || '暂无摘要')}</p></div>
      <div class="pm-section"><h4>章节</h4><p>${escapeHtml((p.sections || []).join(' · ') || '尚未识别章节')}</p></div>
      <div class="pm-section"><h4>文本块预览</h4>
        ${(p.chunk_preview || []).map((c) => `<div class="chunk-preview"><div class="cp-head"><span>${escapeHtml(c.section || '正文')} · p.${c.page_start}</span><span>${c.chars} chars</span></div><p>${escapeHtml(c.preview)}</p></div>`).join('') || '<p>暂无文本块</p>'}
      </div>`;
  } catch (err) {
    $('#paper-modal-body').innerHTML = `<div class="empty-inline">${escapeHtml(err.message)}</div>`;
  }
}

async function deletePaper(paperId) {
  if (!window.confirm('确定删除这篇论文及其全部向量索引吗？此操作不可撤销。')) return;
  try {
    await api(`/api/papers/${paperId}`, { method: 'DELETE' });
    toast('论文已删除', 'success');
    await loadLibrary();
    await loadInfo().catch(() => {});
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function reindexPaper(paperId) {
  try {
    showLoading('正在重建文本块与向量索引…');
    const result = await api(`/api/papers/${paperId}/reindex`, { method: 'POST', body: {} });
    hideLoading();
    toast(`索引已重建 · ${result.chunk_count} chunks`, 'success');
    await loadLibrary();
  } catch (err) {
    hideLoading();
    toast(err.message, 'error');
  }
}

async function reindexAll() {
  if (!window.confirm('将使用当前 Chunk 配置重建全部论文的索引，可能耗时较长。继续吗？')) return;
  showLoading('正在重建全部文献索引…');
  try {
    const result = await api('/api/library/reindex', { method: 'POST', body: {} });
    hideLoading();
    toast(`重建完成：${result.succeeded}/${result.total} 成功`, result.failed ? 'error' : 'success');
    await loadLibrary();
  } catch (err) {
    hideLoading();
    toast(err.message, 'error');
  }
}

/* ------------------------------------------------------------------ */
/* discovery                                                          */
/* ------------------------------------------------------------------ */
function loadDiscoveryDefaults() {
  if (!$('#discover-input').value && state.discoverQuery) $('#discover-input').value = state.discoverQuery;
  if (state.discoverItems.length) renderDiscover(state.discoverItems);
}

async function runDiscovery() {
  const q = $('#discover-input').value.trim();
  if (!q) { toast('请输入检索关键词', 'error'); return; }
  state.discoverQuery = q;
  const year = $('#discover-year').value;
  const scope = $('#discover-scope').value;
  const btn = $('#discover-form button[type="submit"]');
  btn.disabled = true;
  btn.textContent = '检索中…';
  $('#discover-count').textContent = '正在聚合 OpenAlex / arXiv / 本地库…';
  $('#discover-errors').classList.add('hidden');
  $('#discover-results').innerHTML = Array.from({ length: 4 }).map(() => `
    <div class="paper-card"><div class="skeleton line"></div><div class="skeleton line"></div><div class="skeleton line"></div><div class="skeleton line"></div></div>`).join('');
  try {
    const params = new URLSearchParams({ q, limit: '8', scope });
    if (year) params.set('year_from', year);
    const data = await api(`/api/discovery/search?${params}`);
    state.discoverItems = data.items || [];
    renderDiscover(state.discoverItems, data);
    $('#discover-count').textContent = `本地 ${data.local_count} 篇 · 在线 ${data.online_count} 篇`;
    $('#discover-refresh').classList.remove('hidden');
    if (data.errors?.length) {
      const box = $('#discover-errors');
      box.classList.remove('hidden');
      box.innerHTML = data.errors.map((e) => `${icon('spark')} ${escapeHtml(e)}`).join('<br/>');
    }
  } catch (err) {
    $('#discover-results').innerHTML = `<div class="discover-placeholder"><h3>检索失败</h3><p>${escapeHtml(err.message)}</p></div>`;
    $('#discover-count').textContent = '检索失败';
    toast(err.message, 'error', 7000);
  } finally {
    btn.disabled = false;
    btn.textContent = '检索';
  }
}

function renderDiscover(items, data = null) {
  const grid = $('#discover-results');
  if (!items.length) {
    grid.innerHTML = `<div class="discover-placeholder"><div class="radar"><span></span><span></span><span></span><i></i></div><h3>没有找到结果</h3><p>尝试更宽泛的英文关键词，或切换为“仅本地库”。</p></div>`;
    return;
  }
  grid.innerHTML = items.map((p, i) => {
    const local = p.in_library;
    return `<article class="paper-card" data-index="${i}">
      <div class="paper-card-top">
        <span class="source-badge ${local ? 'local' : ''}">${local ? 'IN LIBRARY' : escapeHtml(p.source || 'ONLINE')}</span>
        <span class="paper-meta">${p.citation_count ? p.citation_count + ' citations' : ''}</span>
      </div>
      <h3>${escapeHtml(p.title)}</h3>
      <div class="authors">${escapeHtml(shortAuthors(p.authors, 5))}</div>
      <p class="abstract ${/[\u4e00-\u9fff]/.test(p.abstract || '') ? 'zh' : ''}">${escapeHtml(truncate(p.abstract || '暂无摘要', 420))}</p>
      <div class="paper-card-foot">
        <span class="paper-meta"><strong>${p.year || '—'}</strong> · ${escapeHtml(truncate(p.venue || '—', 40))}</span>
        <div class="card-actions">
          ${safeUrl(p.url) ? `<a class="mini-btn" href="${escapeHtml(safeUrl(p.url))}" target="_blank" rel="noopener">${icon('external')}原文</a>` : ''}
          <button class="mini-btn ${local ? '' : 'primary'}" data-import="${i}" ${local ? 'disabled' : ''}>${local ? '已在库' : '导入'}</button>
        </div>
      </div>
    </article>`;
  }).join('');
}

async function importCandidate(index, button) {
  const p = state.discoverItems[index];
  if (!p) return;
  button.disabled = true;
  const original = button.textContent;
  button.textContent = p.arxiv_id ? '下载并索引…' : '导入中…';
  try {
    const data = await api('/api/discovery/import', {
      method: 'POST',
      body: {
        title: p.title,
        authors: p.authors || [],
        abstract: p.abstract || '',
        year: p.year || null,
        venue: p.venue || '',
        doi: p.doi || '',
        arxiv_id: p.arxiv_id || '',
        url: p.url || '',
        pdf_url: p.pdf_url || '',
        source: p.source || 'import',
        external_id: p.external_id || '',
        citation_count: p.citation_count || 0,
        fetch_pdf: Boolean(p.arxiv_id),
      },
    });
    p.in_library = true;
    button.textContent = '已导入';
    toast(data.message || '导入成功', 'success', 5000);
    await refreshCounts();
    if (state.info?.counts?.papers !== undefined) $('#nav-paper-count').textContent = fmtNum(state.info.counts.papers);
  } catch (err) {
    button.disabled = false;
    button.textContent = original;
    toast(err.message, 'error', 7000);
  }
}

/* ------------------------------------------------------------------ */
/* evaluation                                                         */
/* ------------------------------------------------------------------ */
async function loadEvaluationPage() {
  void loadInfo().catch(() => {});
  await Promise.all([loadDataset(), loadRuns()]);
  if (state.activeRunId) {
    try {
      const data = await api(`/api/evaluations/runs/${state.activeRunId}`);
      state.activeRun = data.run;
      renderRun(data.run);
    } catch {
      state.activeRunId = null;
    }
  }
}

async function loadDataset() {
  try {
    const data = await api('/api/evaluations/dataset?limit=50');
    state.dataset = data;
    $('#dataset-mini strong').textContent = `${data.count} cases`;
    $('#eval-limit').max = Math.max(1, data.count);
    if (Number($('#eval-limit').value) > data.count) $('#eval-limit').value = data.count;
  } catch (err) {
    $('#dataset-mini strong').textContent = '不可用';
  }
}

async function loadRuns() {
  try {
    const data = await api('/api/evaluations/runs?limit=30');
    state.evalRuns = data.items || [];
    renderRunHistory();
    const completed = state.evalRuns.filter((r) => !['queued', 'running'].includes(r.status));
    const preferred = completed.find((r) => r.kind === 'full') || completed[0];
    if (!state.activeRun && preferred) {
      state.activeRunId = preferred.id;
      const full = await api(`/api/evaluations/runs/${preferred.id}`);
      state.activeRun = full.run;
      renderRun(full.run);
    }
    const live = state.evalRuns.find((r) => ['queued', 'running'].includes(r.status));
    if (live) {
      state.activeRunId = live.id;
      pollRun(live.id);
    }
  } catch (err) {
    $('#eval-history').innerHTML = `<div class="empty-inline">${escapeHtml(err.message)}</div>`;
  }
}

function renderRunHistory() {
  const box = $('#eval-history');
  if (!state.evalRuns.length) {
    box.innerHTML = `<div class="empty-inline">暂无记录。运行第一个实验来建立基线。</div>`;
    return;
  }
  const statusClass = { completed: 'success', failed: 'danger', running: 'warn', queued: 'neutral' };
  box.innerHTML = state.evalRuns.map((r) => `
    <div class="run-card" data-run="${r.id}">
      <div class="run-card-top">
        <strong>${escapeHtml(r.name)}</strong>
        <span class="badge ${statusClass[r.status] || 'neutral'}">${r.status}</span>
      </div>
      <p>${escapeHtml(r.summary || r.metrics?.summary || r.error_message || summarizeRunClient(r))}</p>
      <div class="run-card-foot"><span>${escapeHtml(r.kind)}</span><span>${fmtDate(r.created_at, true)}</span></div>
    </div>`).join('');
}

function summarizeRunClient(run) {
  const m = run.metrics || {};
  const parts = [];
  if (m.retrieval_recall_at_k !== undefined) parts.push(`R@K ${pct(m.retrieval_recall_at_k)}`);
  if (m.tool_calling_accuracy !== undefined) parts.push(`Tool ${pct(m.tool_calling_accuracy)}`);
  if (m.task_success_rate !== undefined) parts.push(`Task ${pct(m.task_success_rate)}`);
  if (m.best_config?.recall_at_k !== undefined) {
    const b = m.best_config;
    parts.push(`best ${b.chunk_size}/${b.overlap}/K${b.top_k} · R@K ${pct(b.recall_at_k)}`);
  }
  if (m.best_variant?.task_success_rate !== undefined) {
    parts.push(`best ${m.best_variant.label} · Task ${pct(m.best_variant.task_success_rate)}`);
  }
  return parts.join(' · ') || '暂无指标';
}

async function submitEvaluation(event) {
  event.preventDefault();
  const kind = $('#eval-kind').value;
  const body = {
    kind,
    limit: Number($('#eval-limit').value) || 10,
    top_k: Number($('#eval-topk').value) || 6,
    prompt_variant: $('#eval-prompt').value,
    chunk_sizes: ($('#grid-chunks').value || '300,500,700').split(',').map((x) => parseInt(x.trim(), 10)).filter(Boolean),
    overlaps: ($('#grid-overlaps').value || '50,100').split(',').map((x) => parseInt(x.trim(), 10)).filter((x) => x !== undefined),
    top_ks: ($('#grid-topks').value || '3,5,8').split(',').map((x) => parseInt(x.trim(), 10)).filter(Boolean),
  };
  const btn = $('#run-eval-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>任务已提交…`;
  try {
    const data = await api('/api/evaluations/run', { method: 'POST', body });
    state.activeRunId = data.run.id;
    state.activeRun = data.run;
    renderRun(data.run);
    toast('评测任务已启动', 'success');
    pollRun(data.run.id);
    void loadRuns();
  } catch (err) {
    toast(err.message, 'error', 7000);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${icon('chart')}开始实验`;
  }
}

function pollRun(runId) {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  const tick = async () => {
    if (state.activeRunId !== runId) return;
    try {
      const data = await api(`/api/evaluations/runs/${runId}`);
      const run = data.run;
      state.activeRun = run;
      renderRun(run);
      if (['queued', 'running'].includes(run.status)) {
        state.pollTimer = setTimeout(tick, 1600);
      } else {
        state.pollTimer = null;
        void loadRuns();
        void loadInfo().catch(() => {});
        toast(run.status === 'completed' ? '评测完成' : '评测失败', run.status === 'completed' ? 'success' : 'error', 5000);
      }
    } catch {
      state.pollTimer = setTimeout(tick, 2600);
    }
  };
  state.pollTimer = setTimeout(tick, 700);
}

function renderRun(run) {
  if (!run) return;
  $('#eval-result-title').textContent = run.name;
  const badge = $('#eval-status-badge');
  badge.textContent = run.status;
  badge.className = `badge ${run.status === 'completed' ? 'success' : run.status === 'failed' ? 'danger' : run.status === 'running' ? 'warn' : 'neutral'}`;
  const body = $('#eval-result-body');
  if (run.status === 'queued' || run.status === 'running') {
    const progress = run.metrics?.progress;
    body.innerHTML = `
      <div class="run-progress"><span class="spinner"></span><div><strong>正在执行 · ${escapeHtml(run.kind)}</strong><br/><span style="color:var(--muted)">${progress ? escapeHtml(JSON.stringify(progress)) : 'Agent 正在逐条跑评测用例，请稍候…'}</span></div></div>
      <div class="skeleton line"></div><div class="skeleton line"></div><div class="skeleton line"></div><div class="skeleton line"></div>`;
    return;
  }
  if (run.status === 'failed') {
    body.innerHTML = `<div class="empty-inline" style="color:var(--danger)">${escapeHtml(run.error_message || '评测失败')}</div>`;
    return;
  }
  if (run.kind === 'retrieval_grid') return renderGridRun(run, body);
  if (run.kind === 'prompt_grid') return renderPromptRun(run, body);
  return renderAgentRun(run, body);
}

function metricBox(label, value, note, cls = '') {
  const n = Math.max(0, Math.min(1, Number(value) || 0));
  return `<div class="metric-box ${cls}">
    <div class="m-head"><span>${label}</span><span>${note}</span></div>
    <div class="m-val">${pct(value)}</div>
    <div class="metric-bar"><i style="width:${(n * 100).toFixed(1)}%"></i></div>
  </div>`;
}

function rawMetricBox(label, value, note, cls = '') {
  return `<div class="metric-box ${cls}">
    <div class="m-head"><span>${label}</span><span>${note}</span></div>
    <div class="m-val" style="font-size:18px">${value}</div>
    <div class="metric-bar"><i style="width:100%"></i></div>
  </div>`;
}

function renderAgentRun(run, body) {
  const m = run.metrics || {};
  updateEvalKPIs(m);
  const bars = [
    ['Recall@K', m.retrieval_recall_at_k, '检索'],
    ['Tool Acc.', m.tool_calling_accuracy, '工具'],
    ['Task Success', m.task_success_rate, '任务'],
    ['Citation Hit', m.citation_hit_rate, '引用'],
  ].filter((b) => b[1] !== undefined);
  const cases = (run.details || []).filter((d) => d.stage === 'agent');
  const retrieval = (run.details || []).filter((d) => d.stage === 'retrieval');
  body.innerHTML = `
    <div class="metric-panels">
      ${metricBox('Recall@K', m.retrieval_recall_at_k, `Top-${m.k || run.config?.top_k || 6}`)}
      ${metricBox('Tool Calling Accuracy', m.tool_calling_accuracy, 'coverage', 'violet')}
      ${metricBox('Task Success Rate', m.task_success_rate, `${run.case_count || cases.length} cases`, 'amber')}
      ${metricBox('Citation Hit Rate', m.citation_hit_rate, 'grounded')}
    </div>
    <div class="chart-block">
      <div class="chart-head"><span>指标对比</span><span>${escapeHtml(run.llm_provider || '')}</span></div>
      <div class="bar-chart">
        ${bars.map(([label, value]) => `<div class="bar-row"><span>${label}</span><div class="bar-track"><i style="width:${(Math.max(0, Math.min(1, Number(value) || 0)) * 100).toFixed(1)}%"></i></div><span>${pct(value)}</span></div>`).join('') || '<div class="empty-inline">暂无指标</div>'}
      </div>
    </div>
    <div class="chart-head"><span>逐用例结果</span><span>${cases.filter((c) => c.success).length}/${cases.length} 成功</span></div>
    <div style="overflow-x:auto">
    <table class="case-table">
      <thead><tr><th>用例</th><th>问题</th><th>工具</th><th>状态</th><th>耗时</th></tr></thead>
      <tbody>
        ${cases.map((c, idx) => `
          <tr class="case-expand" data-case="${idx}">
            <td style="font-family:var(--font-mono);font-size:8px">${escapeHtml(c.id)}</td>
            <td class="case-q">${escapeHtml(truncate(c.question, 52))}</td>
            <td style="font-size:8px">${escapeHtml((c.actual_tools || []).join(', ') || '—')}</td>
            <td><span class="case-status ${c.success ? 'ok' : 'fail'}">${c.success ? 'PASS' : 'FAIL'}</span></td>
            <td style="font-family:var(--font-mono);font-size:8px">${c.latency_ms || 0}ms</td>
          </tr>
          <tr class="case-detail hidden" data-case-detail="${idx}"><td colspan="5">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
              <div><strong style="color:var(--text-soft)">模型回答</strong><div class="answer-box">${escapeHtml(c.answer || '(空)')}</div></div>
              <div><strong style="color:var(--text-soft)">评测诊断</strong>
                <p>期望工具：${escapeHtml((c.expected_tools || []).join(', ') || '—')}</p>
                <p>实际工具：${escapeHtml((c.actual_tools || []).join(', ') || '—')}</p>
                <p>关键词：${c.keyword_ok ? '通过' : '缺失 ' + escapeHtml((c.missing_keywords || []).join('; '))}</p>
                <p>引用命中：${c.citation_hit ? '是' : '否'} · 引用数 ${c.citation_count || 0}</p>
                <p>参考：${escapeHtml(truncate(c.reference_answer || '', 260))}</p>
              </div>
            </div>
          </td></tr>`).join('') || '<tr><td colspan="5"><div class="empty-inline">本次运行没有 Agent 用例</div></td></tr>'}
      </tbody>
    </table>
    </div>
    ${retrieval.length ? `<div class="chart-head" style="margin-top:14px"><span>检索明细</span><span>${retrieval.filter((r) => r.hit).length}/${retrieval.length} 命中</span></div>
      <div style="overflow-x:auto"><table class="grid-table"><thead><tr><th>用例</th><th>Recall</th><th>MRR</th><th>首个命中</th></tr></thead><tbody>${retrieval.map((r) => `<tr><td>${escapeHtml(r.id)}</td><td>${pct(r.recall)}</td><td>${Number(r.mrr || 0).toFixed(3)}</td><td>${escapeHtml((r.retrieved || []).find((x) => x.paper_id && (r.relevant_papers || []).includes(x.title))?.title || '—')}</td></tr>`).join('')}</tbody></table></div>` : ''}
    <p class="muted-small" style="margin:12px 0 0">提示：规则评测以关键词与引用落地为准；接入真实模型后，Tool Calling Accuracy 更有参考价值。</p>`;
}

function renderGridRun(run, body) {
  const m = run.metrics || {};
  const rows = (run.details || []).slice().sort((a, b) => (b.recall_at_k || 0) - (a.recall_at_k || 0));
  const best = m.best_config || rows[0] || {};
  updateEvalKPIs({ retrieval_recall_at_k: best.recall_at_k, retrieval_mrr: best.mrr });
  body.innerHTML = `
    <div class="run-progress" style="background:var(--accent-soft)"><span class="badge success">BEST</span><div><strong>chunk=${best.chunk_size || '—'} · overlap=${best.overlap || '—'} · Top-K=${best.top_k || '—'}</strong><br/><span style="color:var(--muted)">Recall@K ${pct(best.recall_at_k)} · MRR ${Number(best.mrr || 0).toFixed(3)} · ${best.chunk_count || 0} chunks</span></div></div>
    <div class="metric-panels">
      ${metricBox('Best Recall@K', best.recall_at_k, 'chunk × top-k')}
      ${metricBox('Best MRR', best.mrr, 'first relevant')}
      ${rawMetricBox('Chunk Count', `${best.chunk_count || 0}`, '本次内存索引')}
      ${rawMetricBox('Avg Chunks/Paper', `${best.avg_chunks_per_paper || 0}`, '平均分块数')}
    </div>
    <div class="chart-head"><span>推荐组合与全部结果</span><span>${rows.length} 组</span></div>
    <div style="overflow-x:auto"><table class="grid-table">
      <thead><tr><th>Chunk</th><th>Overlap</th><th>Top-K</th><th>Recall@K</th><th>MRR</th><th>nDCG@K</th><th>Chunks</th></tr></thead>
      <tbody>${rows.slice(0, 24).map((r) => `<tr class="${r === best ? 'best' : ''}"><td>${r.chunk_size}</td><td>${r.overlap}</td><td>${r.top_k}</td><td>${pct(r.recall_at_k)}</td><td>${Number(r.mrr || 0).toFixed(3)}</td><td>${Number(r.ndcg_at_k || 0).toFixed(3)}</td><td>${r.chunk_count}</td></tr>`).join('')}</tbody>
    </table></div>
    <p class="muted-small" style="margin:12px 0 0">网格实验在内存中重新分块并向量化，不修改正式索引；适合寻找当前语料下的较优 Chunk 与 Top-K 组合。</p>`;
}

function renderPromptRun(run, body) {
  const m = run.metrics || {};
  const variants = (m.variants || []).slice().sort((a, b) => (b.task_success_rate || 0) - (a.task_success_rate || 0));
  const best = m.best_variant || variants[0] || {};
  updateEvalKPIs({
    retrieval_recall_at_k: best.retrieval_recall_at_k,
    tool_calling_accuracy: best.tool_calling_accuracy,
    task_success_rate: best.task_success_rate,
    citation_hit_rate: best.citation_hit_rate,
  });
  body.innerHTML = `
    <div class="run-progress" style="background:var(--accent-soft)"><span class="badge success">BEST</span><div><strong>${escapeHtml(best.label || '—')}</strong><br/><span style="color:var(--muted)">任务成功率 ${pct(best.task_success_rate)} · 工具准确率 ${pct(best.tool_calling_accuracy)} · 引用命中 ${pct(best.citation_hit_rate)}</span></div></div>
    <div style="overflow-x:auto"><table class="grid-table">
      <thead><tr><th>Prompt</th><th>Task Success</th><th>Tool Acc.</th><th>Citation</th><th>Tool F1</th><th>平均耗时</th></tr></thead>
      <tbody>${variants.map((v) => `<tr class="${v === best ? 'best' : ''}"><td>${escapeHtml(v.label)}<br/><span style="font-size:8px;color:var(--muted-2)">${escapeHtml(v.variant)}</span></td><td>${pct(v.task_success_rate)}</td><td>${pct(v.tool_calling_accuracy)}</td><td>${pct(v.citation_hit_rate)}</td><td>${Number(v.tool_f1 || 0).toFixed(3)}</td><td>${v.agent_latency_ms || 0}ms</td></tr>`).join('')}</tbody>
    </table></div>
    <div class="chart-block" style="margin-top:14px">
      <div class="chart-head"><span>Prompt 任务成功率</span><span>${m.case_count || 0} cases</span></div>
      <div class="bar-chart">${variants.map((v) => `<div class="bar-row"><span>${escapeHtml(v.label)}</span><div class="bar-track"><i style="width:${(Math.max(0, Math.min(1, Number(v.task_success_rate) || 0)) * 100).toFixed(1)}%"></i></div><span>${pct(v.task_success_rate)}</span></div>`).join('')}</div>
    </div>`;
}

function updateEvalKPIs(m) {
  const map = {
    recall: m.retrieval_recall_at_k ?? m.recall_at_k,
    tool: m.tool_calling_accuracy,
    success: m.task_success_rate,
    citation: m.citation_hit_rate,
  };
  Object.entries(map).forEach(([key, value]) => {
    const el = $(`[data-kpi="${key}"]`);
    if (el) el.textContent = value === undefined ? '—' : pct(value);
  });
}

async function openDatasetModal() {
  openModal('dataset-modal');
  $('#dataset-body').innerHTML = '<div class="skeleton line"></div><div class="skeleton line"></div>';
  try {
    const data = state.dataset || await api('/api/evaluations/dataset?limit=100');
    state.dataset = data;
    const types = Object.entries(data.by_type || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
    $('#dataset-body').innerHTML = `
      <div class="dataset-summary"><span>用例 <strong>${data.count}</strong></span><span>${escapeHtml(types)}</span><span>${escapeHtml(data.path)}</span></div>
      ${(data.cases || []).map((c) => `<div class="dataset-case">${escapeHtml(c.question)}<span>${escapeHtml(c.id)} · ${escapeHtml(c.type)} · 期望工具：${escapeHtml((c.expected_tools || []).join(', '))}</span></div>`).join('')}`;
  } catch (err) {
    $('#dataset-body').innerHTML = `<div class="empty-inline">${escapeHtml(err.message)}</div>`;
  }
}

/* ------------------------------------------------------------------ */
/* misc                                                               */
/* ------------------------------------------------------------------ */
async function refreshCounts() {
  try {
    const data = await api('/api/library/stats');
    if (state.info) {
      state.info.counts.papers = data.paper_count;
      state.info.counts.chunks = data.chunk_count;
      updateSystemUI();
    }
  } catch {}
}

function openSettings() {
  if (state.info) renderSettingsStatus();
  openModal('settings-modal');
}

function bindEvents() {
  document.addEventListener('click', (event) => {
    const routeEl = event.target.closest('[data-route]');
    if (routeEl) {
      navigate(routeEl.dataset.route);
      return;
    }
    const askEl = event.target.closest('[data-ask]');
    if (askEl) {
      queueQuestion(askEl.dataset.ask);
      return;
    }
    const closeEl = event.target.closest('[data-close]');
    if (closeEl) {
      closeModal(closeEl.dataset.close);
      return;
    }
    const cite = event.target.closest('[data-cite]');
    if (cite) {
      const idx = cite.dataset.cite;
      const card = $(`[data-cite-card="${idx}"]`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card.style.borderColor = 'var(--accent)';
        setTimeout(() => { card.style.borderColor = ''; }, 1600);
      }
      if (window.innerWidth < 1120) $('#evidence-pane').classList.add('open');
      return;
    }
    const conv = event.target.closest('[data-conv]');
    if (conv && !event.target.closest('[data-del-conv]')) {
      navigate('chat', conv.dataset.conv);
      return;
    }
    const delConv = event.target.closest('[data-del-conv]');
    if (delConv) {
      event.stopPropagation();
      deleteConversation(delConv.dataset.delConv);
      return;
    }
    const paper = event.target.closest('[data-paper]');
    if (paper && !event.target.closest('[data-open-paper]')) {
      if (paper.dataset.paper) openPaperModal(paper.dataset.paper);
      return;
    }
    const openPaper = event.target.closest('[data-open-paper]');
    if (openPaper) {
      if (openPaper.dataset.openPaper) openPaperModal(openPaper.dataset.openPaper);
      return;
    }
    const pdf = event.target.closest('[data-pdf]');
    if (pdf) {
      window.open(`/api/papers/${pdf.dataset.pdf}/pdf`, '_blank', 'noopener');
      return;
    }
    const reindexOne = event.target.closest('[data-reindex-one]');
    if (reindexOne) {
      closeModal('paper-modal');
      reindexPaper(reindexOne.dataset.reindexOne);
      return;
    }
    const action = event.target.closest('[data-action]');
    if (action) {
      const { action: act, id } = action.dataset;
      if (act === 'view') openPaperModal(id);
      if (act === 'pdf') window.open(`/api/papers/${id}/pdf`, '_blank', 'noopener');
      if (act === 'reindex') reindexPaper(id);
      if (act === 'delete') deletePaper(id);
      return;
    }
    const imp = event.target.closest('[data-import]');
    if (imp) {
      importCandidate(Number(imp.dataset.import), imp);
      return;
    }
    const run = event.target.closest('[data-run]');
    if (run) {
      loadRunById(run.dataset.run);
      return;
    }
    const copy = event.target.closest('[data-copy-msg]');
    if (copy) {
      const msg = state.messages.find((m) => m.id === copy.dataset.copyMsg);
      if (msg) navigator.clipboard.writeText(msg.content).then(() => toast('已复制回答', 'success', 1800)).catch(() => toast('复制失败', 'error'));
      return;
    }
    const caseRow = event.target.closest('[data-case]');
    if (caseRow && !event.target.closest('button')) {
      const detail = $(`[data-case-detail="${caseRow.dataset.case}"]`);
      if (detail) detail.classList.toggle('hidden');
      return;
    }
    const tab = event.target.closest('[data-tab]');
    if (tab) {
      $$('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      $$('[data-tab-panel]').forEach((p) => p.classList.toggle('active', p.dataset.tabPanel === tab.dataset.tab));
      return;
    }
  });

  $$('.nav-item').forEach((btn) => btn.addEventListener('click', () => navigate(btn.dataset.route)));
  $('#theme-toggle').addEventListener('click', () => applyTheme(state.theme === 'dark' ? 'light' : 'dark'));
  $('#open-settings').addEventListener('click', openSettings);
  $('#avatar-btn').addEventListener('click', openSettings);
  $('#menu-toggle').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
  $('#toggle-evidence').addEventListener('click', () => $('#evidence-pane').classList.toggle('open'));

  $('#overview-ask').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('#overview-ask-input').value.trim();
    if (q) queueQuestion(q);
  });

  $('#new-chat').addEventListener('click', () => startNewChat());
  $('#chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('#chat-input');
    const text = input.value.trim();
    if (!text) return;
    sendChat(text);
  });
  $('#chat-input').addEventListener('input', (e) => autoResizeTextarea(e.target));
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      $('#chat-form').requestSubmit();
    }
  });

  const input = $('#pdf-input');
  $('#upload-trigger').addEventListener('click', () => input.click());
  $('#quick-upload').addEventListener('click', () => { navigate('library'); setTimeout(() => input.click(), 300); });
  $('#upload-zone').addEventListener('click', () => input.click());
  input.addEventListener('change', () => handleUpload(input.files?.[0]));
  ['dragenter', 'dragover'].forEach((ev) => $('#upload-zone').addEventListener(ev, (e) => { e.preventDefault(); $('#upload-zone').classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => $('#upload-zone').addEventListener(ev, (e) => { e.preventDefault(); $('#upload-zone').classList.remove('drag'); }));
  $('#upload-zone').addEventListener('drop', (e) => handleUpload(e.dataTransfer.files?.[0]));
  $('#seed-library').addEventListener('click', () => seedLibrary(false));
  $('#empty-seed').addEventListener('click', () => seedLibrary(false));
  $('#quick-seed').addEventListener('click', () => { navigate('library'); setTimeout(() => seedLibrary(false), 300); });
  $('#refresh-library').addEventListener('click', () => loadLibrary());
  $('#reindex-library').addEventListener('click', reindexAll);
  $('#library-search').addEventListener('input', debounce((e) => { state.libraryQuery = e.target.value.trim(); loadLibrary(); }, 320));

  $('#discover-form').addEventListener('submit', (e) => { e.preventDefault(); runDiscovery(); });
  $('#discover-refresh').addEventListener('click', runDiscovery);

  $('#eval-form').addEventListener('submit', submitEvaluation);
  $('#eval-kind').addEventListener('change', (e) => {
    const grid = ['retrieval_grid', 'prompt_grid'].includes(e.target.value);
    $('#grid-options').classList.toggle('hidden', !grid);
    $('#run-note').innerHTML = grid
      ? `${icon('spark')}参数实验会在内存中重新分块与向量化，不修改正式索引；使用远端 Embedding 时会消耗额度。`
      : `${icon('spark')}离线 Mock 可以快速跑通指标链路；接入真实模型后，工具调用和任务成功指标更有参考价值。`;
  });
  $('#eval-dataset-btn').addEventListener('click', openDatasetModal);
  $('#refresh-runs').addEventListener('click', () => loadRuns());
  $('#refresh-eval').addEventListener('click', () => state.activeRunId && loadRunById(state.activeRunId));

  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === ',') { e.preventDefault(); openSettings(); }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (state.page === 'overview') $('#overview-ask-input').focus();
      else { navigate('chat'); setTimeout(() => $('#chat-input').focus(), 250); }
    }
    if (e.key === 'Escape') {
      closeAllModals();
      $('#sidebar').classList.remove('open');
      $('#evidence-pane').classList.remove('open');
    }
  });

  window.addEventListener('hashchange', () => { void renderRoute(); });
  window.addEventListener('resize', debounce(() => {
    if (window.innerWidth > 1120) $('#evidence-pane').classList.remove('open');
    autoResizeTextarea($('#chat-input'));
  }, 200));
}

function debounce(fn, wait) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function queueQuestion(question) {
  state.pendingQuestion = question;
  if (state.page === 'chat') {
    startNewChat();
    const q = state.pendingQuestion;
    state.pendingQuestion = null;
    setTimeout(() => sendChat(q), 40);
  } else {
    navigate('chat');
  }
}

async function deleteConversation(id) {
  if (!window.confirm('删除这段研究对话？')) return;
  try {
    await api(`/api/conversations/${id}`, { method: 'DELETE' });
    if (state.currentConversationId === id) startNewChat();
    await loadConversations();
    toast('对话已删除', 'success');
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function loadRunById(id) {
  state.activeRunId = id;
  try {
    const data = await api(`/api/evaluations/runs/${id}`);
    state.activeRun = data.run;
    renderRun(data.run);
    if (['queued', 'running'].includes(data.run.status)) pollRun(id);
  } catch (err) {
    toast(err.message, 'error');
  }
}

function init() {
  applyTheme(state.theme);
  bindEvents();
  void (async () => {
    try {
      await loadInfo();
    } catch (err) {
      toast('无法连接后端：' + err.message, 'error', 8000);
    }
    if (!location.hash) history.replaceState(null, '', '#/overview');
    await renderRoute();
    void loadConversations();
    setInterval(() => { if (state.page === 'overview') void refreshCounts(); }, 45000);
  })();
}

document.addEventListener('DOMContentLoaded', init);
