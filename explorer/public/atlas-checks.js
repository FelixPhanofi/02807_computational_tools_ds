import { action, loadPlanes, make } from "./atlas-data.js"

export function fingerprintChecks(parent, open) {
  const title = make("h3", "What tips an answer to yes?"),
    intro = make(
      "p",
      "The fingerprint is a row of 256 numbers. Each check gives those numbers different weights. Some push the total up and others pull it down. See where the line ends.",
    )
  const bits = make("div", null, "atlas-checks"),
    info = make("p"),
    canvas = make("canvas"),
    slider = make("input")
  Object.assign(slider, { type: "range", min: 0, max: 256, value: 256 })
  slider.setAttribute("aria-label", "Add the fingerprint numbers one by one")
  canvas.setAttribute(
    "aria-label",
    "Running total as the check adds each weighted number. The middle line is zero.",
  )
  const note = make("output")
  const sound = make("div", null, "atlas-transform")
  const jump = action("", () => open(trace.boxes[table][0] ^ (1 << bit)), "atlas-jump")
  parent.append(title, sound, intro, bits, info, canvas, slider, note, jump)
  let trace,
    table,
    bit = 0,
    planes,
    totals,
    epoch = 0
  function chart() {
    if (!totals) return
    const w = canvas.clientWidth,
      h = canvas.clientHeight,
      d = Math.min(devicePixelRatio || 1, 2)
    canvas.width = w * d
    canvas.height = h * d
    const c = canvas.getContext("2d")
    c.scale(d, d)
    const middle = h / 2,
      max = Math.max(0.001, ...totals.map(Math.abs)),
      top = 12
    const x = (i) => 5 + (i / 256) * (w - 10),
      y = (v) => middle - (v / max) * (middle - top)
    c.strokeStyle = "#8f9f90"
    c.setLineDash([4, 3])
    c.beginPath()
    c.moveTo(0, middle)
    c.lineTo(w, middle)
    c.stroke()
    c.setLineDash([])
    const stop = Number(slider.value)
    for (const [limit, color] of [
      [256, "#c3ccbe"],
      [stop, "#20634f"],
    ]) {
      c.strokeStyle = color
      c.lineWidth = 2
      c.beginPath()
      c.moveTo(x(0), y(0))
      for (let i = 1; i <= limit; i++) c.lineTo(x(i), y(totals[i]))
      c.stroke()
    }
    c.fillStyle = "#192a24"
    c.beginPath()
    c.arc(x(stop), y(totals[stop]), 4, 0, Math.PI * 2)
    c.fill()
    note.textContent =
      stop +
      " of 256 numbers added · total " +
      totals[stop].toFixed(4) +
      (stop === 256 ? " · final answer " + (trace.projections[table][bit] >= 0 ? "1 (yes)" : "0 (no)") : "")
  }
  function select(b) {
    bit = b
    slider.value = 256
    ;[...bits.children].forEach((n, i) => n.setAttribute("aria-pressed", String(9 - i === bit)))
    const final = trace.projections[table][b]
    const close = trace.boxes[table].slice(1).some((v) => ((v ^ trace.boxes[table][0]) & (1 << b)) !== 0)
    jump.hidden = !close
    jump.textContent = "See the box this one answer changes → " + (trace.boxes[table][0] ^ (1 << b))
    info.textContent =
      "Check " +
      (10 - b) +
      " ended " +
      (final >= 0 ? "above" : "below") +
      " zero at " +
      final.toFixed(4) +
      ". " +
      (close
        ? "It was one of the two closest calls, so the search also tried the other answer."
        : "Drag to watch the weighted numbers add up.")
    totals = [0]
    const offset = (table * 10 + b) * 256
    for (let i = 0; i < 256; i++) totals.push(totals[i] + planes[offset + i] * trace.vector[i])
    chart()
  }
  slider.oninput = chart
  new ResizeObserver(chart).observe(canvas)
  return async (t, map, result) => {
    const token = ++epoch
    trace = t
    table = map
    parent.hidden = !t
    if (!t) {
      totals = null
      return
    }
    const strips = []
    for (const [label, values] of [
      [result.source ? "Original recording" : "Your ten-second clip", result.wave],
      ["Its 256-number fingerprint", t.vector],
    ]) {
      if (!values) continue
      const card = make("figure"),
        graph = make("div", null, "atlas-strip")
      const peak = Math.max(0.0001, ...values.map(Math.abs))
      for (const val of values) {
        const bar = make("i")
        bar.style.height = Math.max(1, (Math.abs(val) / peak) * 100) + "%"
        if (val < 0) bar.className = "negative"
        graph.append(bar)
      }
      graph.setAttribute("role", "img")
      graph.setAttribute("aria-label", label)
      card.append(make("figcaption", label), graph)
      strips.push(card)
    }
    sound.replaceChildren(...strips)
    intro.textContent =
      "The sound model describes a clip with 1,024 numbers. Training taught it to keep a shorter pattern of 256. Each check weights those numbers differently. Drag the slider to see which ones tip the total above or below zero."
    try {
      planes = await loadPlanes()
    } catch {
      if (token === epoch) {
        bits.replaceChildren()
        info.textContent = "Couldn’t load the check weights."
        totals = null
      }
      return
    }
    if (token !== epoch) return
    bits.replaceChildren()
    for (let i = 9; i >= 0; i--) {
      const b = action("", () => select(i))
      b.setAttribute("aria-label", "Inspect check " + (10 - i))
      b.append(make("small", "Check " + (10 - i)), make("strong", String(Number(t.projections[map][i] >= 0))))
      bits.append(b)
    }
    const closest = t.projections[map].map(Math.abs)
    select(closest.indexOf(Math.min(...closest)))
  }
}
