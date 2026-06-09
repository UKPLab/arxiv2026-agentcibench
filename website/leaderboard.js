function copyInstall() {
  const t = document.querySelector('.install-pill code').textContent;
  navigator.clipboard.writeText(t).catch(()=>{});
  const btn = document.querySelector('.install-pill .copy');
  const orig = btn.textContent;
  btn.textContent = 'copied';
  setTimeout(() => btn.textContent = orig, 1200);
}

const lbBody = document.querySelector('#lb .lb-body');
const lbState = { sortKey:'eng', dir:'asc', group:'all' };

function logoPath(family){ return `../figures/logos/${family}.png`; }

function renderLB() {
  const rows = AGENTS
    .filter(a => lbState.group === 'all' ? true : a.group === lbState.group)
    .slice()
    .sort((a,b) => {
      const k = lbState.sortKey;
      if (k === 'model') {
        return lbState.dir === 'asc' ? a.model.localeCompare(b.model) : b.model.localeCompare(a.model);
      }
      if (k === 'group') {
        return lbState.dir === 'asc' ? a.group.localeCompare(b.group) : b.group.localeCompare(a.group);
      }
      const va = a[k], vb = b[k];
      return lbState.dir === 'asc' ? va - vb : vb - va;
    });

  lbBody.innerHTML = rows.map((r,i) => `
    <div class="lb-row">
      <div class="rank">${i+1}</div>
      <div class="model-cell">
        <img class="mlogo" src="${logoPath(r.family)}" alt="" onerror="this.style.display='none'"/>
        <code>${r.model}</code>
      </div>
      <div><span class="${r.group==='prop'?'tag-prop':'tag-open'}">${r.group==='prop'?'Proprietary':'Open'}</span></div>
      <div class="metric-cell"><span class="bar blue"><span class="b" style="width:${r.util}%"></span></span><span class="value">${r.util.toFixed(1)}%</span></div>
      <div class="metric-cell leak-cell"><span class="bar red"><span class="b" style="width:${r.leak}%"></span></span><span class="value">${r.leak.toFixed(1)}%</span><span class="ci">[${r.ci[0].toFixed(0)}-${r.ci[1].toFixed(0)}]</span></div>
      <div class="metric-cell refusal-cell"><span class="value">${r.refusal.toFixed(1)}%</span></div>
      <div class="metric-cell"><span class="bar red"><span class="b" style="width:${r.eng}%"></span></span><span class="value strong">${r.eng.toFixed(1)}%</span></div>
    </div>
  `).join('');

  document.querySelectorAll('#lb .lb-head button').forEach(th => {
    th.classList.remove('sorted','asc','desc');
    if (th.dataset.key === lbState.sortKey) {
      th.classList.add('sorted', lbState.dir);
    }
  });
}

document.querySelectorAll('#lb .lb-head button').forEach(th => {
  th.addEventListener('click', () => {
    const k = th.dataset.key;
    if (!k || k === 'rank') return;
    if (lbState.sortKey === k) {
      lbState.dir = lbState.dir === 'asc' ? 'desc' : 'asc';
    } else {
      lbState.sortKey = k;
      lbState.dir = 'asc';
    }
    renderLB();
  });
});
document.querySelectorAll('.filters .chip').forEach(c => {
  c.addEventListener('click', () => {
    document.querySelectorAll('.filters .chip').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    lbState.group = c.dataset.group;
    renderLB();
  });
});
renderLB();
