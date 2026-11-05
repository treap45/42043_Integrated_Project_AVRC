// Small SVG charts for the project page, drawn from window.SITE_DATA (written by website/build.py).
(function () {
  "use strict";
  const data = window.SITE_DATA;
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent, text) {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo || 1;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step) ticks.push(+t.toFixed(10));
    return ticks;
  }

  function makeTip(host) {
    const tip = document.createElement("div");
    tip.className = "tip";
    host.appendChild(tip);
    return {
      show(html, x, y) {
        tip.innerHTML = html;
        tip.style.display = "block";
        const box = host.getBoundingClientRect();
        const w = tip.offsetWidth;
        const left = Math.min(Math.max(0, x + 12), box.width - w);
        tip.style.left = (x + 12 + w > box.width && x - w - 12 > 0 ? x - w - 12 : left) + "px";
        tip.style.top = Math.max(0, y - 10) + "px";
      },
      hide() { tip.style.display = "none"; },
    };
  }

  function legend(host, items) {
    if (items.length < 2) return;
    const box = document.createElement("div");
    box.className = "legend";
    for (const item of items) {
      const span = document.createElement("span");
      const swatch = document.createElement("i");
      if (item.line) swatch.className = "line";
      swatch.style.background = item.color;
      if (item.opacity) swatch.style.opacity = item.opacity;
      span.append(swatch, item.name);
      box.appendChild(span);
    }
    host.appendChild(box);
  }

  function dataTable(host, head, rows) {
    const details = document.createElement("details");
    details.className = "data";
    const summary = document.createElement("summary");
    summary.textContent = "Show the data as a table";
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    const thead = table.createTHead().insertRow();
    head.forEach((h, i) => { const th = document.createElement("th"); th.textContent = h; if (i) th.className = "n"; thead.appendChild(th); });
    const body = table.createTBody();
    rows.forEach((r) => { const tr = body.insertRow(); r.forEach((v, i) => { const td = tr.insertCell(); td.textContent = v; if (i) td.className = "n"; }); });
    wrap.appendChild(table);
    details.append(summary, wrap);
    host.appendChild(details);
  }

  // Time series with an optional uncertainty band, reference line and vertical event markers.
  function lineChart(host, spec) {
    const W = 800, H = spec.height || 210, m = { l: 54, r: 18, t: 10, b: 38 };
    if (spec.title) { const t = document.createElement("p"); t.className = "chart-title"; t.textContent = spec.title; host.appendChild(t); }
    legend(host, spec.series.map((s) => ({ name: s.name, color: s.color, line: true }))
      .concat(spec.band ? [{ name: spec.band.name, color: spec.band.color, opacity: 0.28 }] : []));
    const holder = document.createElement("div");
    holder.className = "chart";
    host.appendChild(holder);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.alt }, holder);
    const t = spec.t;
    const all = spec.series.flatMap((s) => s.y).concat(spec.band ? spec.band.lo.concat(spec.band.hi) : []).concat(spec.refY ? [spec.refY.y] : []).concat(spec.yExtra || []).filter(Number.isFinite);
    let yLo = Math.min(...all), yHi = Math.max(...all);
    const pad = (yHi - yLo) * 0.06 || 1;
    yLo -= pad; yHi += pad;
    if (spec.yMin !== undefined) yLo = spec.yMin;
    if (spec.yMax !== undefined) yHi = spec.yMax;
    const xs = (v) => m.l + ((v - t[0]) / (t[t.length - 1] - t[0])) * (W - m.l - m.r);
    const ys = (v) => H - m.b - ((v - yLo) / (yHi - yLo)) * (H - m.t - m.b);
    niceTicks(yLo, yHi, 4).forEach((v) => {
      el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v) }, svg);
      el("text", { x: m.l - 8, y: ys(v) + 4, "text-anchor": "end" }, svg, v);
    });
    niceTicks(t[0], t[t.length - 1], 8).forEach((v) => el("text", { x: xs(v), y: H - m.b + 18, "text-anchor": "middle" }, svg, v));
    el("line", { class: "base", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, svg);
    el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 3, "text-anchor": "middle" }, svg, spec.xLabel || "time (s)");
    el("text", { class: "axis-title", transform: `translate(13 ${(H - m.b + m.t) / 2}) rotate(-90)`, "text-anchor": "middle" }, svg, spec.yLabel);
    if (spec.band) {
      const up = t.map((v, i) => `${xs(v).toFixed(1)},${ys(spec.band.hi[i]).toFixed(1)}`);
      const dn = t.map((v, i) => `${xs(v).toFixed(1)},${ys(spec.band.lo[i]).toFixed(1)}`).reverse();
      el("polygon", { points: up.concat(dn).join(" "), fill: spec.band.color, opacity: 0.22 }, svg);
    }
    if (spec.refY) {
      el("line", { class: "marker-line", x1: m.l, x2: W - m.r, y1: ys(spec.refY.y), y2: ys(spec.refY.y) }, svg);
      el("text", { class: "direct", x: (m.l + W - m.r) / 2, y: ys(spec.refY.y) - 5, "text-anchor": "middle" }, svg, spec.refY.label);
    }
    (spec.events || []).forEach((ev, i) => {
      el("line", { class: "marker-line", x1: xs(ev.t), x2: xs(ev.t), y1: m.t, y2: H - m.b }, svg);
      el("text", { class: "direct", x: xs(ev.t) + (ev.side === "left" ? -5 : 5), y: m.t + 12 + i * 15, "text-anchor": ev.side === "left" ? "end" : "start" }, svg, ev.label);
    });
    spec.series.forEach((s) => {
      const parts = [];
      let open = false;
      s.y.forEach((v, i) => {
        if (!Number.isFinite(v)) { open = false; return; }
        parts.push(`${open ? "L" : "M"}${xs(t[i]).toFixed(1)} ${ys(v).toFixed(1)}`);
        open = true;
      });
      el("path", { d: parts.join(""), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    });
    const cross = el("line", { class: "marker-line", y1: m.t, y2: H - m.b, visibility: "hidden" }, svg);
    const dots = spec.series.map((s) => el("circle", { r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
    const tip = makeTip(holder);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
    function moveTo(i, px, py) {
      cross.setAttribute("x1", xs(t[i])); cross.setAttribute("x2", xs(t[i])); cross.setAttribute("visibility", "visible");
      let rows = "";
      spec.series.forEach((s, k) => {
        const ok = Number.isFinite(s.y[i]);
        dots[k].setAttribute("visibility", ok ? "visible" : "hidden");
        if (ok) { dots[k].setAttribute("cx", xs(t[i])); dots[k].setAttribute("cy", ys(s.y[i])); rows += `<br>${s.name}: <b>${s.y[i].toFixed(spec.digits ?? 2)}</b>`; }
      });
      if (spec.band) rows += `<br>${spec.band.name}: <b>${spec.band.lo[i].toFixed(spec.digits ?? 2)} to ${spec.band.hi[i].toFixed(spec.digits ?? 2)}</b>`;
      tip.show(`t = <b>${t[i].toFixed(1)} s</b>${rows}`, px, py);
    }
    function nearest(evt) {
      const box = svg.getBoundingClientRect();
      const x = ((evt.clientX - box.left) / box.width) * W;
      const value = t[0] + ((x - m.l) / (W - m.l - m.r)) * (t[t.length - 1] - t[0]);
      let best = 0;
      t.forEach((v, i) => { if (Math.abs(v - value) < Math.abs(t[best] - value)) best = i; });
      moveTo(best, evt.clientX - box.left, evt.clientY - box.top);
    }
    function clear() {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      tip.hide();
    }
    hit.addEventListener("pointermove", nearest);
    hit.addEventListener("pointerleave", clear);
    let cursor = 0;
    hit.addEventListener("keydown", (evt) => {
      if (evt.key !== "ArrowLeft" && evt.key !== "ArrowRight") return;
      evt.preventDefault();
      cursor = Math.max(0, Math.min(t.length - 1, cursor + (evt.key === "ArrowRight" ? 1 : -1)));
      moveTo(cursor, (xs(t[cursor]) / W) * svg.clientWidth, 20);
    });
    hit.addEventListener("blur", clear);
  }

  // Horizontal strips of per-trial values, one row per condition and one colour per method.
  function stripPlot(host, spec) {
    legend(host, spec.methods.map((k) => ({ name: k.name, color: k.color })));
    const holder = document.createElement("div");
    holder.className = "chart";
    host.appendChild(holder);
    const rowH = 46, W = 800, m = { l: 118, r: 18, t: 8, b: 38 };
    const rows = spec.rows.length * spec.methods.length;
    const H = m.t + m.b + spec.rows.length * (rowH * spec.methods.length * 0.62 + 12);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.alt }, holder);
    const xs = (v) => m.l + (v / spec.xMax) * (W - m.l - m.r);
    niceTicks(0, spec.xMax, 6).forEach((v) => {
      el("line", { class: "grid", x1: xs(v), x2: xs(v), y1: m.t, y2: H - m.b }, svg);
      el("text", { x: xs(v), y: H - m.b + 18, "text-anchor": "middle" }, svg, v);
    });
    el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 3, "text-anchor": "middle" }, svg, spec.xLabel);
    const tip = makeTip(holder);
    let y = m.t;
    const table = [];
    spec.rows.forEach((row) => {
      const block = rowH * spec.methods.length * 0.62;
      el("text", { class: "axis-title", x: m.l - 12, y: y + block / 2 + 4, "text-anchor": "end" }, svg, row.name);
      spec.methods.forEach((method, k) => {
        const cy = y + ((k + 0.5) / spec.methods.length) * block;
        const values = row.values[method.key] || [];
        values.forEach((v, seed) => {
          if (v === null) return;
          const dot = el("circle", { cx: xs(v), cy, r: 4.5, fill: method.color, stroke: "var(--surface)", "stroke-width": 1.5, opacity: 0.85 }, svg);
          const hitDot = el("circle", { cx: xs(v), cy, r: 11, fill: "transparent" }, svg);
          hitDot.addEventListener("pointerenter", () => {
            const box = svg.getBoundingClientRect();
            tip.show(`${row.name}, ${method.name}<br>seed ${seed}: <b>${v.toFixed(1)} mm</b>`, (xs(v) / W) * box.width, (cy / H) * box.height);
            dot.setAttribute("r", 6);
          });
          hitDot.addEventListener("pointerleave", () => { tip.hide(); dot.setAttribute("r", 4.5); });
        });
        const good = values.filter((v) => v !== null).sort((a, b) => a - b);
        table.push([row.name + ", " + method.name, good.length, good.length ? good[Math.floor(good.length / 2)].toFixed(1) : "-", good.length ? good[good.length - 1].toFixed(1) : "-"]);
      });
      y += block + 12;
    });
    if (spec.limit) {
      el("line", { class: "marker-line", x1: xs(spec.limit.x), x2: xs(spec.limit.x), y1: m.t, y2: H - m.b }, svg);
      el("text", { class: "direct", x: xs(spec.limit.x) + 5, y: m.t + 11 }, svg, spec.limit.label);
    }
    dataTable(host, ["Condition", "Trials with a grasp offset", "Median (mm)", "Max (mm)"], table);
    void rows;
  }

  // Stacked horizontal bars of trial outcomes, with a 2 px gap between segments.
  function outcomeBars(host, spec) {
    legend(host, spec.keys.map((k) => ({ name: k.name, color: k.color })));
    const holder = document.createElement("div");
    holder.className = "chart";
    host.appendChild(holder);
    const W = 800, barH = 22, gap = 18, m = { l: 118, r: 18, t: 4, b: 34 };
    const H = m.t + m.b + spec.rows.length * (barH + gap);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.alt }, holder);
    const xs = (v) => m.l + (v / spec.total) * (W - m.l - m.r);
    niceTicks(0, spec.total, 5).forEach((v) => {
      el("line", { class: "grid", x1: xs(v), x2: xs(v), y1: m.t, y2: H - m.b }, svg);
      el("text", { x: xs(v), y: H - m.b + 16, "text-anchor": "middle" }, svg, v);
    });
    el("text", { class: "axis-title", x: (W + m.l) / 2, y: H - 2, "text-anchor": "middle" }, svg, spec.xLabel);
    const tip = makeTip(holder);
    spec.rows.forEach((row, i) => {
      const y = m.t + gap / 2 + i * (barH + gap);
      el("text", { class: "axis-title", x: m.l - 12, y: y + barH / 2 + 4, "text-anchor": "end" }, svg, row.name);
      let acc = 0;
      spec.keys.forEach((key) => {
        const v = row[key.key] || 0;
        if (!v) return;
        const x0 = xs(acc), x1 = xs(acc + v);
        const r = el("rect", { x: x0 + 1, y, width: Math.max(1, x1 - x0 - 2), height: barH, rx: 3, fill: key.color }, svg);
        if (x1 - x0 > 26) el("text", { x: (x0 + x1) / 2, y: y + barH / 2 + 4, "text-anchor": "middle", style: "fill:#fff;font-weight:600;pointer-events:none" }, svg, v);
        r.addEventListener("pointerenter", () => {
          const box = svg.getBoundingClientRect();
          tip.show(`${row.name}<br>${key.name}: <b>${v}/${spec.total}</b>`, (x0 / W) * box.width, (y / H) * box.height);
        });
        r.addEventListener("pointerleave", () => tip.hide());
        acc += v;
      });
    });
    dataTable(host, ["Mode"].concat(spec.keys.map((k) => k.name)), spec.rows.map((r) => [r.name].concat(spec.keys.map((k) => r[k.key] || 0))));
  }

  const C = { a: "var(--series-1)", b: "var(--series-2)", c: "var(--series-3)", n: "var(--series-neutral)" };
  const mount = (id) => document.getElementById(id);

  if (mount("chart-ekf-x")) {
    const e = data.ekf_trace;
    const ev = [{ t: e.true_stop, label: "cube stops", side: "left" }, { t: e.declared_stop, label: "stop declared", side: "left" }];
    lineChart(mount("chart-ekf-x"), {
      title: "Cube x position: estimate, 2σ band and truth", t: e.t, yLabel: "x (mm)", digits: 1, height: 230, events: ev,
      alt: "EKF estimate of the cube x position with its two sigma band, next to the true position",
      series: [{ name: "EKF estimate", color: C.a, y: e.x_hat }, { name: "True state", color: C.b, y: e.x_true }],
      band: { name: "Estimate ±2σ", color: C.a, lo: e.x_lo, hi: e.x_hi },
    });
    lineChart(mount("chart-ekf-err"), {
      title: "Position error against the filter's own 2σ", t: e.t, yLabel: "error (mm)", digits: 2, height: 190, yMin: 0, events: ev,
      alt: "EKF position error magnitude against the two sigma bound the filter reports",
      series: [{ name: "Position error", color: C.a, y: e.err }, { name: "Filter's 2σ (largest axis)", color: C.b, y: e.sigma2 }],
    });
    lineChart(mount("chart-ekf-v"), {
      title: "Cube speed: estimate and truth", t: e.t, yLabel: "speed (mm/s)", digits: 1, height: 190, yMin: 0, events: ev,
      refY: { y: 5, label: "stop threshold 5 mm/s" },
      alt: "Estimated and true cube speed, with the 5 millimetre per second stop threshold",
      series: [{ name: "EKF speed", color: C.a, y: e.v_hat }, { name: "True speed", color: C.b, y: e.v_true }],
    });
  }

  if (mount("chart-push")) {
    stripPlot(mount("chart-push"), {
      alt: "Grasp offset per seed for look-then-move and PBVS with the cube pushed 0, 15 and 25 millimetres",
      xLabel: "grasp offset: cube centre to TCP when lifted (mm)", xMax: 32,
      methods: [{ key: "open", name: "Look-then-move", color: C.b }, { key: "pbvs", name: "PBVS", color: C.a }],
      rows: data.push_rows,
    });
  }

  if (mount("chart-occluders")) {
    outcomeBars(mount("chart-occluders"), {
      alt: "Outcome of 50 occluded trials for three grasp selection modes",
      xLabel: "trials (seeds 0-49, static cube, 1-3 occluders)", total: 50,
      keys: [{ key: "placed", name: "Placed", color: C.a }, { key: "collided", name: "Failed in motion", color: C.b }, { key: "refused", name: "Refused the grasp", color: C.n }, { key: "other", name: "Other", color: C.c }],
      rows: data.occluder_rows,
    });
  }

  // Theme toggle and the section highlight in the side navigation.
  const toggle = document.getElementById("theme-toggle");
  if (toggle) {
    let stored = null;
    try { stored = localStorage.getItem("theme"); } catch (e) { /* storage may be blocked */ }
    if (stored) document.documentElement.setAttribute("data-theme", stored);
    toggle.addEventListener("click", () => {
      const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked */ }
    });
  }
  const links = [...document.querySelectorAll(".toc a[href^='#']")];
  const sections = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id));
      });
    }, { rootMargin: "-15% 0px -75% 0px" });
    sections.forEach((s) => observer.observe(s));
  }
})();
