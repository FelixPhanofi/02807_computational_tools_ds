import "./vendor/d3-hierarchy-3.1.2.min.js"
import { action, fmt, make } from "./atlas-data.js"

// Two linked scales keep every address visible while a pointer or arrow key explores nearby boxes.
export function addressRuler(parent, choose) {
  const full = make("canvas"),
    lens = make("canvas"),
    readout = make("output"),
    slider = make("input")
  full.setAttribute("aria-label", "All 1,024 boxes in this map")
  lens.setAttribute("aria-label", "Magnified view of 17 nearby boxes")
  Object.assign(slider, { type: "range", min: 0, max: 1023, step: 1 })
  slider.setAttribute("aria-label", "Open a box by number")
  const axis = make("div", null, "atlas-axis")
  axis.append(make("span", "0"), make("span", "512"), make("span", "1,023"))
  const guide = make(
    "p",
    "Move across the top ruler to magnify 17 boxes below. Tap a box to open it. Taller marks hold more pieces.",
    "atlas-ruler-guide",
  )
  parent.append(guide, full, axis, lens, slider, readout)
  let counts,
    probes = [],
    selected = 0,
    center = 0,
    begin = 0
  function canvasCtx(c) {
    const w = c.clientWidth,
      h = c.clientHeight,
      dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = Math.round(w * dpr)
    c.height = Math.round(h * dpr)
    const ctx = c.getContext("2d")
    ctx.scale(dpr, dpr)
    return { w, h, ctx }
  }
  function draw() {
    if (!counts) return
    const { w, h, ctx } = canvasCtx(full)
    const max = Math.max(...counts)
    begin = Math.max(0, Math.min(1007, center - 8))
    counts.forEach((v, i) => {
      ctx.fillStyle = "#95afa0"
      const y = (v / max) * (h - 8)
      ctx.fillRect((i * w) / 1024, h - y, Math.max(0.7, w / 1024), y)
    })
    ctx.fillStyle = "#d7fa803b"
    ctx.fillRect((begin * w) / 1024, 0, (17 * w) / 1024, h)
    for (const i of probes) {
      ctx.fillStyle = i === probes[0] ? "#d6f783" : "#f7aa7b"
      ctx.fillRect((i * w) / 1024 - 1, 3, 2, h)
    }
    ctx.fillStyle = "white"
    ctx.fillRect((selected * w) / 1024, 0, 1.4, h)
    const { w: lw, h: lh, ctx: l } = canvasCtx(lens),
      step = lw / 17
    const localMax = Math.max(1, ...counts.slice(begin, begin + 17))
    for (let j = 0; j < 17; j++) {
      const i = begin + j,
        y = (counts[i] / localMax) * (lh - 19)
      l.fillStyle = i === selected ? "#eff5dc" : probes.includes(i) ? "#f8a979" : "#829e89"
      l.fillRect(j * step + 1, lh - 14 - y, Math.max(1, step - 2), y)
      if (j === 0 || j === 16 || j === 8) {
        l.fillStyle = "#dee9dc"
        l.font = "10px system-ui"
        l.textAlign = "center"
        l.fillText(i, (j + 0.5) * step, lh - 2)
      }
    }
  }
  function at(c, event, zoom) {
    const r = c.getBoundingClientRect()
    return Math.max(
      0,
      Math.min(
        1023,
        (zoom ? begin : 0) +
          Math.min(zoom ? 16 : 1023, Math.floor(((event.clientX - r.left) / r.width) * (zoom ? 17 : 1024))),
      ),
    )
  }
  function label(i) {
    readout.textContent =
      "Box " + i + " · " + fmt(counts[i]) + " pieces" + (probes.includes(i) ? " · searched" : "")
  }
  full.onpointermove = (e) => {
    if (counts && e.pointerType !== "touch") {
      center = at(full, e, false)
      draw()
      label(center)
    }
  }
  full.onclick = (e) => {
    if (counts) choose(at(full, e, false))
  }
  lens.onpointermove = (e) => {
    if (counts) label(at(lens, e, true))
  }
  lens.onclick = (e) => {
    if (counts) choose(at(lens, e, true))
  }
  slider.oninput = () => choose(Number(slider.value))
  new ResizeObserver(draw).observe(parent)
  return (c, p, s) => {
    counts = c
    probes = p
    selected = center = s
    slider.value = s
    slider.setAttribute("aria-valuetext", "Box " + s)
    draw()
    label(s)
  }
}

const palette = ["#bdd695", "#8cb9a3", "#dcaa7f", "#a9b9d2", "#d7c488", "#9ebfc0", "#c5b3c6", "#b4c3a1"]
export function groupMap(parent, library, pin) {
  const allGenres = [...new Set(library.songs.map((s) => s.genre || "Other"))].sort()
  const board = make("div", null, "atlas-board"),
    note = make("div", null, "atlas-map-note"),
    hint = make("span")
  const zoom = action("Zoom into genre", () => {
    zoomed = zoomed ? null : hovered?.song.genre
    draw()
  })
  const tell = make("div", null, "atlas-tooltip")
  tell.hidden = true
  board.append(tell)
  note.append(hint, zoom)
  parent.append(board, note)
  let groups = [],
    selected,
    hovered,
    zoomed,
    genreList = [],
    cells = [],
    width = 0
  function draw() {
    width = board.clientWidth
    if (!width || !groups.length) return
    const height = board.clientHeight
    tell.hidden = true
    board.replaceChildren(tell)
    cells = []
    const byGenre = Map.groupBy
      ? Map.groupBy(groups, (g) => g.song.genre || "Other")
      : groups.reduce((m, g) => {
          let k = g.song.genre || "Other"
          if (!m.has(k)) m.set(k, [])
          m.get(k).push(g)
          return m
        }, new Map())
    genreList = [...byGenre.keys()].sort()
    if (!byGenre.has(zoomed)) zoomed = null
    const tree = d3
      .hierarchy({
        children: genreList
          .filter((k) => !zoomed || k === zoomed)
          .map((genre) => ({ genre, children: byGenre.get(genre) })),
      })
      .sum((d) => d.rows?.length || 0)
      .sort((a, b) => b.value - a.value)
    d3
      .treemap()
      .size([width, height])
      .round(true)
      .paddingInner(1)
      .paddingTop((n) => (n.depth === 1 && n.value / tree.value > 0.075 ? 16 : 0))(tree)
    for (const n of tree.children || []) {
      if (n.value / tree.value > 0.075 && n.x1 - n.x0 > 35) {
        const name = make("small", n.data.genre, "atlas-genre")
        place(name, n.x0, n.y0, n.x1 - n.x0, 16)
        board.append(name)
      }
    }
    for (const [i, n] of tree.leaves().entries()) {
      const g = n.data,
        b = action(
          "",
          () => {
            pin(g)
            describe(g, b, false)
          },
          "atlas-cell",
        )
      b.dataset.id = g.song.id
      const title =
        g.song.title +
        " · " +
        g.song.artist +
        " · " +
        g.rows.length +
        (g.rows.length === 1 ? " piece" : " pieces")
      b.setAttribute("aria-label", title)
      b.setAttribute("aria-pressed", String(g.song.id === selected))
      b.tabIndex = i === 0 || g.song.id === selected ? 0 : -1
      b.style.background = palette[allGenres.indexOf(n.parent.data.genre) % palette.length]
      const marks = make("span", null, "atlas-dots")
      for (const r of g.rows) marks.append(make("i"))
      b.append(marks)
      if (n.x1 - n.x0 > 45 && n.y1 - n.y0 > 22) b.append(make("span", g.song.title, "atlas-cell-title"))
      place(b, n.x0, n.y0, n.x1 - n.x0, n.y1 - n.y0)
      b.onpointerenter = (e) => {
        if (e.pointerType !== "touch") describe(g, b, true)
      }
      b.onfocus = () => describe(g, b, true)
      b.onkeydown = (e) => {
        const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -1, ArrowDown: 1 }[e.key]
        if (delta) {
          e.preventDefault()
          cells[(i + delta + cells.length) % cells.length].focus()
        }
      }
      board.append(b)
      cells.push(b)
    }
    const preferred =
      groups.find((g) => g.song.id === selected && (!zoomed || g.song.genre === zoomed)) ||
      tree.leaves()[0]?.data
    if (preferred) describe(preferred, null, false)
    if (zoomed) zoom.textContent = "Show all genres"
  }
  function place(el, x, y, w, h) {
    Object.assign(el.style, {
      left: x + "px",
      top: y + "px",
      width: Math.max(0, w) + "px",
      height: Math.max(0, h) + "px",
    })
  }
  function describe(g, b, tooltip) {
    hovered = g
    hint.textContent =
      g.song.title +
      " · " +
      g.song.artist +
      " · " +
      g.rows.length +
      (g.rows.length === 1 ? " piece" : " pieces")
    if (!zoomed) zoom.textContent = "Zoom into " + (g.song.genre || "Other")
    if (tooltip && b) {
      tell.textContent =
        hint.textContent +
        " · " +
        g.rows
          .map((r) => g.song.starts[r - g.song.first] + "–" + (g.song.starts[r - g.song.first] + 10) + " s")
          .join(", ")
      tell.hidden = false
      tell.style.left = Math.max(2, Math.min(board.clientWidth - tell.offsetWidth - 2, b.offsetLeft)) + "px"
      tell.style.top =
        Math.max(0, b.offsetTop > 38 ? b.offsetTop - tell.offsetHeight : b.offsetTop + b.offsetHeight) + "px"
    } else tell.hidden = true
  }
  board.onpointerleave = () => {
    tell.hidden = true
  }
  new ResizeObserver(() => {
    if (board.clientWidth !== width) draw()
  }).observe(board)
  return {
    show(rows, id) {
      const grouped = new Map()
      for (const row of rows) {
        const s = library.songAt(row)
        if (!grouped.has(s.id)) grouped.set(s.id, { song: s, rows: [] })
        grouped.get(s.id).rows.push(row)
      }
      groups = [...grouped.values()]
      selected = id
      zoomed = null
      draw()
    },
    pin(id) {
      selected = id
      for (const b of cells) b.setAttribute("aria-pressed", String(Number(b.dataset.id) === id))
    },
  }
}
