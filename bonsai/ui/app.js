/* 前端逻辑：轮询状态、流式对话、设置面板。
   这里刻意不做任何构建步骤 —— 直接改就能生效。 */

const $ = (id) => document.getElementById(id);
const TOKEN_KEY = 'bonsai.token';
let TOKEN = (window.__BONSAI_TOKEN__ || '').trim();
let state = {};
let convs = [];
let current = null;
let streaming = false;
let booted = false;

/* ─────────────────────────────── 小工具 ─────────────────────────────── */
function toast(msg, ms = 1900) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
  });
  const text = await res.text();
  try { return { ok: res.ok, status: res.status, data: JSON.parse(text) }; }
  catch { return { ok: res.ok, status: res.status, data: { raw: text } }; }
}

function esc(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* 极简 Markdown：代码块、行内代码、粗体、列表、段落。
   故意的 —— 引入一个 Markdown 库会让这个文件大 20 倍，收益很小。 */
function md(src) {
  const blocks = [];
  let s = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code>${esc(code.replace(/\n$/, ''))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  s = esc(s);
  s = s.replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  s = s.split(/\n{2,}/).map((para) => {
    const lines = para.split('\n');
    if (lines.every((l) => /^\s*[-*·]\s+/.test(l))) {
      return '<ul>' + lines.map((l) => `<li>${l.replace(/^\s*[-*·]\s+/, '')}</li>`).join('') + '</ul>';
    }
    if (/^\u0000\d+\u0000$/.test(para.trim())) return para;
    return '<p>' + lines.join('<br>') + '</p>';
  }).join('');
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
  return s;
}

/* ─────────────────────────────── 会话存储 ───────────────────────────── */
function loadConvs() {
  try { convs = JSON.parse(localStorage.getItem('bonsai.convs') || '[]'); }
  catch { convs = []; }
  current = convs[0] || null;
}
function saveConvs() {
  localStorage.setItem('bonsai.convs', JSON.stringify(convs.slice(0, 60)));
}
function newConv() {
  const c = { id: 'c' + Date.now(), title: '新对话', messages: [] };
  convs.unshift(c);
  current = c;
  saveConvs();
  renderConvList();
  renderMessages();
  $('input').focus();
}

function renderConvList() {
  const box = $('conv-list');
  box.innerHTML = '';
  convs.forEach((c) => {
    const b = document.createElement('button');
    b.className = 'conv' + (current && c.id === current.id ? ' active' : '');
    b.textContent = c.title || '新对话';
    b.onclick = () => { current = c; renderConvList(); renderMessages(); };
    box.appendChild(b);
  });
}

function renderMessages() {
  const stream = $('stream');
  const msgs = current ? current.messages : [];
  if (!msgs.length) {
    stream.innerHTML = $('welcome') ? '' : '';
    stream.innerHTML = welcomeHTML();
    bindStarters();
    return;
  }
  stream.innerHTML = '';
  msgs.forEach((m, i) => stream.appendChild(msgNode(m, i)));
  scrollDown();
}

function welcomeHTML() {
  return `<div class="welcome">
    <div class="mark small"><svg viewBox="0 0 48 48" fill="none"><path d="M24 42V20" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><path d="M24 26c0-7 5-12 12-12 0 7-5 12-12 12Z" fill="currentColor" opacity=".85"/><path d="M24 21c0-6-4-11-11-11 0 6 4 11 11 11Z" fill="currentColor" opacity=".5"/><path d="M15 42h18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg></div>
    <h2>有什么可以帮你的？</h2>
    <p class="muted">全部在这台电脑上运行，对话不会发到任何服务器。</p>
    <div class="starters">
      <button class="starter" data-p="用三句话解释量子纠缠">用三句话解释量子纠缠</button>
      <button class="starter" data-p="帮我把下面这段文字改得更简洁：">帮我润色一段文字</button>
      <button class="starter" data-p="写一个 Python 函数，判断一个字符串是不是回文">写一个小函数</button>
    </div></div>`;
}

function bindStarters() {
  document.querySelectorAll('.starter').forEach((b) => {
    b.onclick = () => { $('input').value = b.dataset.p; autoGrow(); $('input').focus(); };
  });
}

function msgNode(m, idx) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (m.role === 'user' ? 'user' : 'ai');
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = m.role === 'user' ? '你' : 'AI';
  const body = document.createElement('div');
  body.className = 'body';
  body.innerHTML = m.role === 'user' ? `<p>${esc(m.content).replace(/\n/g, '<br>')}</p>`
                                     : md(m.content || '');
  wrap.append(who, body);
  if (m.role !== 'user' && m.content) {
    const tools = document.createElement('div');
    tools.className = 'tools';
    const cp = document.createElement('button');
    cp.className = 'icon-btn';
    cp.title = '复制';
    cp.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>';
    cp.onclick = () => { navigator.clipboard.writeText(m.content); toast('已复制'); };
    tools.appendChild(cp);
    body.appendChild(tools);
  }
  return wrap;
}

function scrollDown() {
  const s = $('stream');
  s.scrollTop = s.scrollHeight;
}

/* ─────────────────────────────── 发送消息 ───────────────────────────── */
async function send() {
  const input = $('input');
  const text = input.value.trim();
  if (!text || streaming) return;
  if (!current) newConv();

  current.messages.push({ role: 'user', content: text });
  if (current.title === '新对话') {
    current.title = text.slice(0, 18) + (text.length > 18 ? '…' : '');
    renderConvList();
  }
  input.value = '';
  autoGrow();
  renderMessages();

  const holder = current.messages;
  const ai = { role: 'assistant', content: '' };
  holder.push(ai);
  const stream = $('stream');
  if (holder.length === 2) renderMessages();
  const node = msgNode(ai, holder.length - 1);
  stream.appendChild(node);
  const body = node.querySelector('.body');
  body.innerHTML = '<span class="caret"></span>';
  scrollDown();

  streaming = true;
  $('send').disabled = true;
  $('status-text').textContent = '正在回答';
  $('status-pill').classList.add('busy');

  const payload = {
    model: 'bonsai',
    messages: holder.slice(0, -1).map((m) => ({ role: m.role, content: m.content })),
    stream: true,
    temperature: 0.7,
    max_tokens: 2048,
    // 直接回答，不要先输出一段思考过程
    chat_template_kwargs: { enable_thinking: false },
  };

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + TOKEN },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.text()).slice(0, 300) || ('HTTP ' + res.status));

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        for (const line of part.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          try {
            const j = JSON.parse(data);
            const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
            if (d.content) {
              ai.content += d.content;
              body.innerHTML = md(ai.content) + '<span class="caret"></span>';
              scrollDown();
            }
          } catch { /* 不完整的一行，等下一块 */ }
        }
      }
    }
  } catch (e) {
    ai.content = ai.content || '';
    body.innerHTML = md(ai.content) + `<p style="color:#f87171">出错了：${esc(String(e.message || e))}</p>`;
  } finally {
    streaming = false;
    $('send').disabled = false;
    $('status-text').textContent = '就绪';
    $('status-pill').classList.remove('busy');
    saveConvs();
    renderMessages();
  }
}

function autoGrow() {
  const t = $('input');
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 190) + 'px';
}

/* ─────────────────────────────── 状态轮询 ───────────────────────────── */
function setBoot(p) {
  const bar = $('boot-bar');
  if (p.total) {
    bar.style.width = p.pct + '%';
    $('boot-detail').textContent = `${p.mb_done} / ${p.mb_total} MB` + (p.detail ? ' · ' + p.detail : '');
  } else {
    bar.style.width = p.stage === 'ready' ? '100%' : '35%';
    $('boot-detail').textContent = p.detail || '';
  }
  $('boot-label').textContent = p.label || '';
}

async function poll() {
  const r = await api('/app/state');
  if (!r.ok) return;
  state = r.data;

  const p = state.progress || {};
  if (p.stage === 'error' || state.error) {
    $('boot-title').textContent = '没能启动';
    $('boot-sub').textContent = '下面是具体原因。';
    $('boot-progress').classList.add('hidden');
    $('boot-error').classList.remove('hidden');
    $('boot-error-text').textContent = p.error || state.error || '未知错误';
    return;
  }

  const ready = state.engine && state.engine.running && p.stage === 'ready';
  if (ready && !booted) {
    booted = true;
    $('boot').classList.add('hidden');
    $('app').classList.remove('hidden');
    $('chip-model').textContent = state.gpu ? state.gpu.name : 'CPU 运行';
    updateRemoteUI();
    updateEnv();
    $('input').focus();
  } else if (!ready) {
    $('boot').classList.remove('hidden');
    $('app').classList.add('hidden');
    $('boot-title').textContent = '正在准备';
    $('boot-sub').textContent = state.model_present
      ? '模型已就绪，正在载入。'
      : '第一次打开需要下载模型（约 5.5 GB），只需一次。';
    $('boot-progress').classList.remove('hidden');
    $('boot-error').classList.add('hidden');
    setBoot(p);
  }
}

/* ──────────────────────────────── 设置 ─────────────────────────────── */
function openSettings() {
  $('settings').classList.remove('hidden');
  renderSettings();
}
function closeSettings() { $('settings').classList.add('hidden'); }

function renderSettings() {
  const s = state.settings || {};
  // 记忆容量
  const box = $('tiles-memory');
  box.innerHTML = '';
  Object.entries(state.tiers || {}).forEach(([key, t]) => {
    const b = document.createElement('button');
    b.className = 'tile' + (s.memory_tier === key ? ' active' : '');
    b.innerHTML = `<b>${t.label}</b><span>${t.hint}<br>${t.ctx / 1024}K 上下文</span>`;
    b.onclick = () => saveSettings({ memory_tier: key });
    box.appendChild(b);
  });
  $('path-value').textContent = s.data_dir || '—';
  $('token-value').textContent = TOKEN ? TOKEN.slice(0, 10) + '••••••••••••••••••••••' : '（远程访问时请输入令牌）';
  updateRemoteUI();
  updateEnv();
}

function updateEnv() {
  const e = state.engine || {};
  const rows = [
    ['运行方式', state.gpu ? `GPU · ${state.gpu.name}` : 'CPU（没有检测到 NVIDIA 显卡）'],
    ['显存', state.gpu ? `${(state.gpu.vram_mb / 1024).toFixed(0)} GB` : '—'],
    ['记忆容量', `${(state.context || 0) / 1024}K`],
    ['版本', state.version || '—'],
  ];
  $('env-kv').innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${esc(String(v))}</b></div>`).join('');
}

function updateRemoteUI() {
  const t = (state.tunnel) || {};
  const on = !!t.running;
  $('remote-toggle').checked = on;
  $('remote-state').textContent = on ? '已开启' : '已关闭';
  $('remote-panel').classList.toggle('hidden', !on);
  $('chip-remote').classList.toggle('hidden', !on);
  if (on) {
    $('remote-url').textContent = t.url || '正在启动…';
    $('remote-note').textContent = '这个网址是临时的，关掉开关就会失效。转发给谁，谁就能用你的模型，请连同令牌一起发。';
    loadQR(t.url);
  }
}

let qrFor = '';
function loadQR(url) {
  if (!url || url === qrFor) return;
  qrFor = url;
  $('qr').innerHTML = '<svg viewBox="0 0 1 1"></svg>';
  fetch('/app/qr?data=' + encodeURIComponent(url))
    .then((r) => (r.ok ? r.text() : Promise.reject()))
    .then((svg) => { $('qr').innerHTML = svg; })
    .catch(() => { $('qr').innerHTML = '<span style="color:#333;font-size:11px">无法生成二维码</span>'; });
}

async function saveSettings(patch) {
  const r = await api('/app/settings', { method: 'POST', body: JSON.stringify(patch) });
  if (r.ok) {
    state = r.data.state || state;
    toast('已保存');
    if (r.data.restarting) { booted = false; }
    renderSettings();
  } else {
    toast(r.data.error || '保存失败');
  }
}

/* ──────────────────────────────── 绑定 ─────────────────────────────── */
function bind() {
  $('new-chat').onclick = newConv;
  $('send').onclick = send;
  $('input').addEventListener('input', autoGrow);
  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  $('open-settings').onclick = openSettings;
  $('close-settings').onclick = closeSettings;
  $('settings').addEventListener('click', (e) => { if (e.target.id === 'settings') closeSettings(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSettings(); });

  $('boot-retry').onclick = async () => {
    $('boot-error').classList.add('hidden');
    $('boot-progress').classList.remove('hidden');
    await api('/app/restart', { method: 'POST', body: '{}' });
  };

  $('pick-folder').onclick = async () => {
    toast('请在弹出的窗口里选择文件夹…', 3200);
    const r = await api('/app/pick-folder', { method: 'POST', body: '{}' });
    if (r.ok && r.data.path) await saveSettings({ data_dir: r.data.path });
    else if (r.data && r.data.error) toast(r.data.error);
  };

  $('remote-toggle').onchange = async (e) => {
    const want = e.target.checked;
    $('remote-state').textContent = want ? '正在启动…' : '正在关闭…';
    const r = await api('/app/remote', { method: 'POST', body: JSON.stringify({ enabled: want }) });
    if (!r.ok) { toast(r.data.error || '操作失败'); e.target.checked = !want; }
    else if (want && r.data.url) { toast('已开启，可以扫码了'); }
    await poll();
    updateRemoteUI();
  };

  $('copy-url').onclick = () => {
    const u = $('remote-url').textContent;
    if (u && u.startsWith('http')) { navigator.clipboard.writeText(u); toast('链接已复制'); }
  };
  $('copy-token').onclick = () => {
    if (TOKEN) { navigator.clipboard.writeText(TOKEN); toast('令牌已复制'); }
  };
  $('rotate-token').onclick = async () => {
    if (!confirm('重新生成后，所有已经在用旧令牌的程序都会失效。继续？')) return;
    const r = await api('/app/token/rotate', { method: 'POST', body: '{}' });
    if (r.ok) { TOKEN = r.data.token; localStorage.setItem(TOKEN_KEY, TOKEN); renderSettings(); toast('已生成新令牌'); }
  };
  $('restart-engine').onclick = async () => {
    toast('正在重启模型…');
    booted = false;
    await api('/app/restart', { method: 'POST', body: '{}' });
  };
}

/* ──────────────────────────────── 启动 ─────────────────────────────── */
(async function main() {
  if (!TOKEN) TOKEN = localStorage.getItem(TOKEN_KEY) || '';
  if (!TOKEN) {
    const entered = prompt('请输入访问令牌（在运行这个程序的那台电脑上，打开设置即可看到）');
    if (entered) { TOKEN = entered.trim(); localStorage.setItem(TOKEN_KEY, TOKEN); }
  } else {
    localStorage.setItem(TOKEN_KEY, TOKEN);
  }
  loadConvs();
  renderConvList();
  renderMessages();
  bind();
  await poll();
  setInterval(poll, 2500);
})();
