import { scorePlot } from "./scatter.js"
import { mountRuler } from "./ruler.js"
import "./audio.js"

const get = (id) => document.getElementById(id)
const count = (value) => value.toLocaleString("en")
function el(tag, text, className) {
  const node = document.createElement(tag)
  if (text != null) node.textContent = text
  if (className) node.className = className
  return node
}
const outcome = (q) => (!q.accepted ? "No guess" : q.correct ? "Right song" : "Wrong song")
const title = (q) => q.title || "Track " + q.track
const methods = {
  exact: ["Check every clip", "Exact search"],
  lsh: ["Look in matching boxes", "LSH"],
  ivf: ["Look in similar groups", "IVF"],
}
const effects = {
  voice_over: "a voice talking over the music",
  speed_up: "faster playback",
  compression: "compressed audio",
  reverb: "room reflections",
  time_stretch: "a change in tempo",
  pitch_shift: "a change in pitch",
  low_pass: "muffled high notes",
  high_pass: "quieter low notes",
  noise: "background noise",
  band_pass: "only the middle notes",
  echo: "an echo",
}

function benchmarks(measurements) {
  const state = { catalogue: "large", family: "standard", metric: "accuracy" }
  const headings = {
    accuracy: "How many clips got the right title?",
    ms: "How long did one lookup take?",
    qps: "How many clips in one second?",
  }
  function draw() {
    const { catalogue, family, metric } = state
    const timed = metric !== "accuracy"
    get("family").disabled = timed
    const rows = measurements.filter((r) => r.catalogue === catalogue && r.family === family)
    const value = (r) => (timed ? r[metric] : (100 * r.correct) / r.total)
    const max = timed ? Math.max(...rows.map(value)) : 100
    get("chart-title").textContent = headings[metric]
    get("chart").replaceChildren(
      ...rows.map((row) => {
        const line = el("div", null, "barrow")
        const head = el("div", null, "barhead"),
          name = el("span", methods[row.method][0])
        name.append(el("small", methods[row.method][1]))
        const v = value(row)
        head.append(
          name,
          el(
            "b",
            metric === "accuracy"
              ? v.toFixed(1) + "%"
              : metric === "ms"
                ? v.toFixed(3) + " ms"
                : count(Math.round(v)),
          ),
        )
        const track = el("div", null, "bar-track"),
          bar = el("div", null, "bar-fill")
        bar.dataset.method = row.method
        bar.style.width = (100 * v) / max + "%"
        track.append(bar)
        line.append(head, track)
        return line
      }),
    )
    const boxes = rows.find((r) => r.method === "lsh")
    const exact = rows.find((r) => r.method === "exact")
    const changed =
      family === "standard" ? "such as noise and echo" : "such as faster playback and talking over the music"
    get("chart-note").textContent =
      metric === "accuracy"
        ? "The box search named " +
          count(boxes.correct) +
          " of " +
          count(boxes.total) +
          " clips with changes " +
          changed +
          ". Each bar counts the first guess."
        : metric === "ms"
          ? "The box search was " +
            (exact.ms / boxes.ms).toFixed(2) +
            "× faster than checking every clip. We timed one lookup at a time on the same graphics card, with the fingerprints ready."
          : "Longer is faster. Each method searched 128 ready-made fingerprints at a time on the same graphics card."
  }
  for (const key of Object.keys(state)) {
    const buttons = [...get(key).querySelectorAll("button")]
    function mark() {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.value === state[key])))
    }
    buttons.forEach(
      (b) =>
        (b.onclick = () => {
          state[key] = b.value
          mark()
          draw()
        }),
    )
    mark()
  }
  draw()
}

function queryExplorer(queries) {
  let filtered = [],
    page = 0,
    selected
  const pageSize = 4
  const plot = scorePlot(get("score-plot"), get("plot-tip"), (q) => {
    selected = q
    page = Math.floor(filtered.indexOf(q) / pageSize)
    show()
  })
  for (const genre of [...new Set(queries.map((q) => q.genre).filter(Boolean))].sort())
    get("genre").add(new Option(genre, genre))
  function select(q) {
    selected = q
    plot(filtered, q)
    for (const button of get("rows").children)
      button.setAttribute("aria-pressed", String(button.dataset.id === q.id))
    const who = el("div", null, "inspect-song"),
      name = el("h3", title(q))
    who.append(
      el("span", outcome(q), "verdict"),
      name,
      el("p", [q.artist, q.genre].filter(Boolean).join(" · ")),
    )
    const story = el("div", null, "inspect-story")
    const changes = q.condition
      .split("+")
      .map((k) => effects[k] || k.replaceAll("_", " "))
      .join(", ")
    story.append(
      el(
        "p",
        (q.known ? "The original was in the library." : "We hid the original song.") +
          " We tested it with " +
          changes +
          ".",
      ),
    )
    const dl = el("dl")
    for (const [key, value] of [
      ["Best guess", q.predicted_title || "Track " + q.predicted],
      ["Next closest", q.runner_up_title || "Unknown title"],
      ["Match strength", q.score.toFixed(3)],
      ["Lead", q.gap.toFixed(3)],
    ]) {
      const cell = el("div")
      cell.append(el("dt", key), el("dd", value))
      dl.append(cell)
    }
    story.append(dl)
    get("inspect").replaceChildren(who, story)
  }
  function show() {
    const rows = get("rows"),
      start = page * pageSize,
      visible = filtered.slice(start, start + pageSize)
    rows.replaceChildren(
      ...visible.map((q) => {
        const b = el("button", null, "queryrow"),
          t = el("span")
        b.dataset.id = q.id
        t.append(el("b", title(q)), el("small", q.artist || "FMA " + q.track))
        b.append(t, el("span", outcome(q), "tag"))
        b.onclick = () => select(q)
        return b
      }),
    )
    if (visible.length) select(visible.includes(selected) ? selected : visible[0])
    else {
      plot([], null)
      get("inspect").textContent = "No clips match these filters."
    }
    get("count").textContent = visible.length
      ? start + 1 + "–" + (start + visible.length) + " of " + count(filtered.length) + " clips"
      : "0 clips"
    get("prev").disabled = page === 0
    get("next").disabled = start + pageSize >= filtered.length
  }
  function filter() {
    const term = get("search").value.trim().toLowerCase()
    filtered = queries.filter(
      (q) =>
        (!get("outcome").value || outcome(q) === get("outcome").value) &&
        (!get("genre").value || q.genre === get("genre").value) &&
        [q.id, q.track, q.title, q.artist, q.predicted_title, q.runner_up_title].some(
          (v) => v != null && String(v).toLowerCase().includes(term),
        ),
    )
    page = 0
    show()
  }
  get("search").oninput = filter
  get("outcome").onchange = filter
  get("genre").onchange = filter
  get("prev").onclick = () => {
    page--
    show()
  }
  get("next").onclick = () => {
    page++
    show()
  }
  filter()
}
function coverage(genres) {
  const rows = get("genre-rows"),
    sorted = [...genres].sort((a, b) => b.tracks - a.tracks)
  const panel = get("genre-detail"),
    layout = panel.parentElement
  const small = matchMedia("(max-width:640px)")
  let active
  function inspect(g) {
    active = g
    const selectedButton = [...rows.querySelectorAll(".genre-row")].find((b) => b.value === g.genre)
    if (small.matches) {
      if (selectedButton.nextElementSibling !== panel) selectedButton.after(panel)
    } else if (panel.parentElement !== layout) layout.append(panel)
    for (const b of rows.querySelectorAll(".genre-row"))
      b.setAttribute("aria-pressed", String(b.value === g.genre))
    const link = el("a", "Explore " + g.genre + " clips →", "genre-link")
    link.href = "#queries"
    link.onclick = () => {
      get("search").value = ""
      get("outcome").value = ""
      get("genre").value = g.genre
      get("genre").dispatchEvent(new Event("change"))
    }
    const fraction = el("div", null, "genre-fraction")
    fraction.append(el("strong", count(g.correct)), el("span", " / " + count(g.clips)))
    get("genre-detail").replaceChildren(
      el("h3", g.genre),
      fraction,
      el("p", "clips got the right title when the song was in the library."),
      el(
        "p",
        "We tested " + count(g.songs) + " songs from this genre. The library holds " + count(g.tracks) + ".",
        "genre-context",
      ),
      link,
    )
  }
  for (const g of sorted) {
    const b = el("button", null, "genre-row")
    b.value = g.genre
    const rail = el("span", null, "genre-rail"),
      fill = el("i")
    fill.style.width = (100 * g.tracks) / sorted[0].tracks + "%"
    rail.append(fill)
    b.append(el("span", g.genre), rail, el("b", count(g.tracks)))
    b.onfocus = b.onclick = () => inspect(g)
    b.onpointerenter = (e) => {
      if (e.pointerType === "mouse") inspect(g)
    }
    rows.append(b)
  }
  small.addEventListener("change", () => inspect(active))
  inspect(sorted[0])
}
mountRuler(get("ruler")).catch(
  () => (get("ruler").textContent = "The song map couldn’t load. Refresh to try again."),
)
try {
  const response = await fetch("/evidence.json")
  if (!response.ok) throw Error()
  const data = await response.json()
  benchmarks(data.methods)
  queryExplorer(data.queries)
  coverage(data.genres)
} catch {
  for (const id of ["chart", "rows", "genre-rows"])
    get(id).textContent = "The results couldn’t load. Refresh to try again."
}
