// Every mark maps back to a row in the held-out test. Numeric axes stay fixed while filtering.
export function scorePlot(canvas, tip, choose) {
  let rows = [],
    selected,
    points = [],
    hover
  const threshold = { score: 0.749896646, gap: 0.117484212 }
  const ink = "#192c27",
    grid = "#d8dfd8",
    known = "#08695e",
    hidden = "#b55430"
  function paint() {
    const width = canvas.clientWidth,
      height = canvas.clientHeight
    if (!width || !height) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext("2d")
    ctx.scale(dpr, dpr)
    const left = 36,
      right = width - 14,
      top = 18,
      bottom = height - 29
    const x = (v) => left + ((v - 0.4) / 0.6) * (right - left),
      y = (v) => bottom - (v / 0.55) * (bottom - top)
    ctx.fillStyle = "#e7f1cf"
    ctx.fillRect(x(threshold.score), top, right - x(threshold.score), y(threshold.gap) - top)
    ctx.lineWidth = 1
    ctx.font = "11px system-ui"
    ctx.textBaseline = "middle"
    function line(x1, y1, x2, y2) {
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    }
    ctx.strokeStyle = grid
    ctx.fillStyle = "#51615b"
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) {
      line(x(v), top, x(v), bottom)
      ctx.textAlign = "center"
      ctx.fillText(v.toFixed(1), x(v), bottom + 15)
    }
    for (const v of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) {
      line(left, y(v), right, y(v))
      ctx.textAlign = "right"
      ctx.fillText(v.toFixed(1), left - 8, y(v))
    }
    ctx.strokeStyle = "#698052"
    ctx.setLineDash([4, 4])
    line(x(threshold.score), top, x(threshold.score), bottom)
    line(left, y(threshold.gap), right, y(threshold.gap))
    ctx.setLineDash([])
    ctx.fillStyle = "#3c5230"
    ctx.textAlign = "right"
    ctx.font = "600 12px system-ui"
    ctx.fillText("Give a title", right - 7, top + 13)
    points = rows.map((q) => ({ q, x: x(q.score), y: y(q.gap) }))
    ctx.globalAlpha = 0.58
    for (const p of points) {
      ctx.fillStyle = p.q.known ? known : hidden
      ctx.beginPath()
      if (p.q.known) ctx.arc(p.x, p.y, 2.8, 0, Math.PI * 2)
      else {
        ctx.moveTo(p.x, p.y - 3.1)
        ctx.lineTo(p.x + 3.1, p.y)
        ctx.lineTo(p.x, p.y + 3.1)
        ctx.lineTo(p.x - 3.1, p.y)
        ctx.closePath()
      }
      ctx.fill()
    }
    ctx.globalAlpha = 1
    for (const q of [selected, hover])
      if (q) {
        ctx.beginPath()
        ctx.arc(x(q.score), y(q.gap), 6, 0, Math.PI * 2)
        ctx.strokeStyle = "white"
        ctx.lineWidth = 4
        ctx.stroke()
        ctx.strokeStyle = ink
        ctx.lineWidth = 1.5
        ctx.stroke()
      }
  }
  function nearest(event) {
    const box = canvas.getBoundingClientRect(),
      x = event.clientX - box.left,
      y = event.clientY - box.top
    let hit,
      best = 14 ** 2
    for (const p of points) {
      const d = (p.x - x) ** 2 + (p.y - y) ** 2
      if (d < best) {
        hit = p
        best = d
      }
    }
    return hit
  }
  canvas.onpointermove = (event) => {
    if (event.pointerType === "touch") return
    const hit = nearest(event)
    if (hit?.q === hover) return
    hover = hit?.q
    tip.hidden = !hit
    if (hit) {
      tip.textContent =
        (hover.title || "Track " + hover.track) + " · " + (hover.known ? "in library" : "hidden")
      tip.style.left = Math.max(6, Math.min(canvas.clientWidth - tip.offsetWidth - 6, hit.x + 10)) + "px"
      tip.style.top = canvas.offsetTop + Math.max(2, hit.y - 36) + "px"
    }
    paint()
  }
  canvas.onpointerleave = () => {
    hover = null
    tip.hidden = true
    paint()
  }
  canvas.onclick = (event) => {
    const hit = nearest(event)
    if (hit) choose(hit.q)
  }
  new ResizeObserver(paint).observe(canvas)
  return (filtered, current) => {
    rows = filtered
    selected = current
    hover = null
    tip.hidden = true
    paint()
  }
}
