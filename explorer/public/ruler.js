import { action, clock, fmt, loadLibrary, loadMap, loadSearch, make, scoreAt } from "./atlas-data.js"
import { addressRuler, groupMap } from "./atlas-map.js"
import { fingerprintChecks } from "./atlas-checks.js"
import { excerptPlayer } from "./atlas-audio.js"
import { selectedSearch } from "./search-session.js"

const examples = [
  [2, "Hello Central, Give Me Heaven"],
  [5, "1978"],
  [6, "Coucou"],
]
const safeLink = (url) => {
  try {
    const u = new URL(url)
    if (["http:", "https:"].includes(u.protocol)) return u.href
  } catch {}
  return null
}
const timeRange = (start) => clock(start) + "–" + clock(start + 10)
export async function mountRuler(root) {
  const library = await loadLibrary()
  root.className = "search-atlas"
  root.innerHTML = `
    <div class="atlas-context"><div><small>Following</small><h3 data-id="title"></h3><p data-id="summary" role="status"></p></div>
    <label>Explore another search<select data-id="example"><option value="">Player selection</option></select></label></div>
    <div data-id="work">
    <p class="atlas-map-intro" data-id="mapIntro"></p>
    <div class="atlas-tools"><label>Map <select data-id="map"></select></label><div data-id="probes" class="atlas-probes" role="group" aria-label="Boxes opened by this search"></div></div>
    <div class="atlas-field" data-id="ruler"></div>
    <div class="atlas-body"><div class="atlas-box"><div class="atlas-subhead"><h3 data-id="box"></h3><span data-id="box-count"></span></div>
    <p>Each tile is a song. Each dot is a ten-second piece. Hover for a name, or tap to open its timeline.</p>
    <div data-id="groups"></div></div>
    <aside class="atlas-inspector"><div class="atlas-subhead"><h3>Song timeline</h3></div><div class="atlas-shortcuts" data-id="shortcuts"></div>
    <h4 data-id="song"></h4><p data-id="artist"></p><div data-id="track-axis"></div><div data-id="windows" class="atlas-windows"></div>
    <p class="atlas-result" data-id="piece"></p><button type="button" data-id="play" class="atlas-play"></button>
    <small data-id="play-state" role="status"></small><div data-id="credit" class="atlas-credit"></div></aside></div>
    <div class="atlas-browser"><h3>Find a song in the library.</h3><label>Song, artist or track number<input type="search" data-id="search" placeholder="Try a song or artist"></label>
    <div class="atlas-directory" data-id="results"></div><div class="atlas-pages"><button data-id="prev" aria-label="Previous songs">←</button><output data-id="count"></output><button data-id="next" aria-label="Next songs">→</button></div></div>
    <div class="atlas-fingerprint" data-id="checks"></div></div>`
  const ui = Object.fromEntries([...root.querySelectorAll("[data-id]")].map((e) => [e.dataset.id, e]))
  let queryEpoch = 0,
    mapEpoch = 0,
    table = 0,
    box = 0,
    map,
    trace,
    scores,
    result,
    song,
    row,
    offset = null,
    audioState = "idle",
    latest = selectedSearch(),
    lastRequest
  const audio = excerptPlayer((state, msg) => {
    audioState = state
    ui["play-state"].textContent =
      msg ||
      (state === "loading"
        ? "Fetching the original recording…"
        : state === "playing"
          ? "Playing the selected part."
          : "")
    ui.play.textContent =
      state === "playing"
        ? "■ Stop"
        : state === "loading"
          ? "Cancel loading"
          : "▶ Play " + (song ? timeRange(offset ?? song.starts[row - song.first]) : "piece")
  })
  const inspectChecks = fingerprintChecks(ui.checks, (value) => {
    openBox(value)
    ui.box.tabIndex = -1
    ui.box.focus({ preventScroll: true })
    ui.groups.scrollIntoView({ block: "center" })
  })
  const drawRuler = addressRuler(ui.ruler, openBox)
  const tiles = groupMap(ui.groups, library, (g) => pin(g.song, bestRow(g.rows), null))
  for (let i = 0; i < 64; i++) ui.map.add(new Option(i + 1 + " of 64", String(i)))
  examples.forEach(([idx, title]) => ui.example.add(new Option(title, String(idx))))
  function bestRow(rows) {
    return rows.reduce(
      (best, r) => ((scoreAt(scores, r) ?? -Infinity) > (scoreAt(scores, best) ?? -Infinity) ? r : best),
      rows[0],
    )
  }
  function rowsOf(s) {
    return s.starts.map((_, i) => s.first + i)
  }
  function pin(next, ref, landmark) {
    if (!next) return
    song = next
    row = ref
    offset = landmark
    audio.stop()
    const start = landmark ?? song.starts[row - song.first]
    ui.song.textContent = song.title
    ui.artist.textContent = [song.artist, song.genre].filter(Boolean).join(" · ")
    ui["track-axis"].textContent = "0:00 — " + clock(Math.round(song.duration)) + " in the archive"
    ui.windows.replaceChildren()
    rowsOf(song).forEach((r) => {
      const s = song.starts[r - song.first],
        lane = make("div")
      const b = action(timeRange(s), () => locate(r, false))
      b.style.left = (100 * s) / song.duration + "%"
      b.style.width = (100 * 10) / song.duration + "%"
      b.setAttribute("aria-pressed", String(offset === null && row === r))
      b.title = timeRange(s) + (r === trace?.bestRow ? " · closest piece" : "")
      if (r === trace?.bestRow) b.className = "best"
      else if (scoreAt(scores, r) !== null) b.className = "compared"
      if (map?.keys[r] === box) b.classList.add("in-box")
      lane.append(b)
      ui.windows.append(lane)
    })
    const sourceId = lastRequest?.sample ?? result?.source
    if (sourceId === song.id) {
      const marker = make("i", null, "atlas-query-start")
      const start = lastRequest.sample ? 4 : result.start
      marker.style.left = (100 * start) / song.duration + "%"
      marker.title = "Test clip started at " + start.toFixed(2) + " seconds"
      ui.windows.append(marker)
      ui["track-axis"].textContent += " · dashed line marks the clip start"
    }
    if (landmark !== null) {
      const lane = make("div"),
        marker = make("span", timeRange(start), "atlas-offset")
      marker.style.left = (100 * start) / song.duration + "%"
      marker.style.width = (100 * 10) / song.duration + "%"
      lane.append(marker)
      ui.windows.append(lane)
    }
    const value = scoreAt(scores, row)
    const found =
      landmark !== null
        ? "Landmarks lined up here."
        : row === trace?.bestRow
          ? result.accepted
            ? "The closest piece."
            : "Closest guess; the search was unsure."
          : row === trace?.runnerRow
            ? "Closest piece from the second song."
            : trace && value === null
              ? "This piece stayed outside the shortlist."
              : "Saved piece."
    ui.piece.textContent =
      timeRange(start) +
      " · " +
      found +
      (landmark === null && map?.keys[row] === box ? " In this box." : "") +
      (landmark === null && value !== null ? " Match " + value.toFixed(3) + " / 1.000." : "")
    const [license, licenseUrl, playable] = song.rights
    ui.play.hidden = !playable
    ui.play.textContent = "▶ Play " + timeRange(start)
    ui.credit.replaceChildren()
    for (const [text, url] of [
      ["Original recording", song.url],
      [license || "Source information", licenseUrl],
    ]) {
      const dest = safeLink(url)
      if (dest) {
        const a = make("a", text)
        a.href = dest
        a.target = "_blank"
        a.rel = "noopener"
        ui.credit.append(a)
      }
    }
    if (!playable) ui["play-state"].textContent = "Open the original recording to listen."
    tiles.pin(song.id)
  }
  ui.play.onclick = () => {
    if (!song) return
    if (audioState === "loading" || audioState === "playing") audio.stop()
    else audio.play(song, offset ?? song.starts[row - song.first])
  }
  function openBox(k, keep = false) {
    if (!map) return
    box = k
    const rows = map.bins[k],
      probes = trace?.boxes[table] || []
    drawRuler(map.counts, probes, k)
    ui.box.textContent = "Box " + fmt(k)
    const ids = new Set(rows.map((r) => library.songAt(r).id))
    ui["box-count"].textContent = fmt(ids.size) + " songs · " + fmt(rows.length) + " pieces"
    ui.probes.replaceChildren(
      ...probes.map((id, i) => {
        const b = action((i ? "Nearby " : "Clip’s box ") + id, () => openBox(id))
        b.setAttribute("aria-pressed", String(id === k))
        return b
      }),
    )
    tiles.show(rows, song?.id)
    if (keep && song) pin(song, row, offset)
    if (!keep && rows.length) {
      const best = bestRow(rows)
      pin(library.songAt(best), best, null)
    }
  }
  async function setMap(n, requested, keep = false) {
    const token = ++mapEpoch
    table = n
    ui.map.value = n
    try {
      const loaded = await loadMap(n)
      if (token !== mapEpoch) return
      if (loaded.keys.length !== library.total) throw Error("Map size mismatch")
      map = loaded
      openBox(
        typeof requested === "function" ? requested(loaded) : (requested ?? trace?.boxes[n][0] ?? box),
        keep,
      )
      inspectChecks(trace, n, result)
    } catch {
      if (token === mapEpoch) {
        ui.summary.textContent = "This map couldn’t load. "
        ui.summary.append(action("Try again", () => setMap(n, requested, keep)))
      }
    }
  }
  ui.map.onchange = () => setMap(Number(ui.map.value))
  async function locate(ref, findSearched) {
    const token = ++mapEpoch,
      qe = queryEpoch
    let n = table,
      first
    try {
      first = await loadMap(n)
      if (findSearched && trace && !trace.boxes[n].includes(first.keys[ref])) {
        for (let i = 0; i < 64; i++) {
          const other = await loadMap(i)
          if (token !== mapEpoch || qe !== queryEpoch) return
          if (trace.boxes[i].includes(other.keys[ref])) {
            n = i
            break
          }
        }
      }
      if (token !== mapEpoch || qe !== queryEpoch) return
      pin(library.songAt(ref), ref, null)
      await setMap(n, (m) => m.keys[ref], true)
    } catch {
      if (token === mapEpoch) {
        ui.summary.textContent = "Couldn’t open the selected piece. "
        ui.summary.append(action("Try again", () => locate(ref, findSearched)))
      }
    }
  }
  const directory = [...library.songs].sort((a, b) => a.title.localeCompare(b.title) || a.id - b.id)
  let filtered = directory,
    page = 0
  function list() {
    const start = page * 5
    ui.results.replaceChildren(
      ...filtered.slice(start, start + 5).map((s) => {
        const b = action("", () => locate(bestRow(rowsOf(s)), false))
        b.append(make("b", s.title), make("small", s.artist))
        return b
      }),
    )
    ui.count.textContent = filtered.length
      ? start + 1 + "–" + Math.min(start + 5, filtered.length) + " of " + fmt(filtered.length)
      : "No songs found"
    ui.prev.disabled = page === 0
    ui.next.disabled = start + 5 >= filtered.length
  }
  ui.search.oninput = () => {
    const term = ui.search.value.toLocaleLowerCase().trim()
    filtered = directory.filter((s) => s.term.includes(term))
    page = 0
    list()
  }
  ui.prev.onclick = () => {
    page--
    list()
  }
  ui.next.onclick = () => {
    page++
    list()
  }
  list()
  async function follow(request) {
    lastRequest = request
    const token = ++queryEpoch
    ++mapEpoch
    audio.stop()
    ui.work.hidden = true
    ui.title.textContent = request.name
    ui.summary.textContent = "Opening this search…"
    try {
      const loaded = await loadSearch(request)
      if (token !== queryEpoch) return
      ;({ trace, scores, result } = loaded)
      if (trace && (scores.length !== library.total || trace.boxes.length !== 64))
        throw Error("Incompatible search")
      ui.work.hidden = false
      ui.mapIntro.textContent = trace
        ? "Each map sorts the same music a different way. The search opens the clip’s own box and three more in case a close answer goes the other way."
        : "Browse the saved pieces in the music library. This map is separate from the sound-peak search."
      ui.shortcuts.replaceChildren()
      ui.example.value = request.example !== undefined ? String(request.example) : ""
      if (trace) {
        const best = library.songAt(trace.bestRow)
        ui.summary.textContent =
          (request.sample ? "Saved sample search. " : "") +
          (result.accepted ? "Found “" + best.title + "”. " : "No clear match. ") +
          fmt(trace.candidates) +
          " pieces compared across 64 maps."
        ui.shortcuts.append(action("Closest piece", () => locate(trace.bestRow, true)))
        if (trace.runnerRow != null)
          ui.shortcuts.append(action("Runner-up", () => locate(trace.runnerRow, true)))
        table = 0
        await locate(trace.bestRow, true)
      } else {
        ui.summary.textContent =
          result.trace?.kind === "landmarks"
            ? "This search compared sound peaks in 24 songs. You can listen to its result and browse the library."
            : "This search returned a title without a saved box path. You can still explore the library."
        await setMap(0, 0, true)
        if (token !== queryEpoch) return
        const candidate = library.byId.get(result.track_id)
        if (candidate && result.accepted && Number.isFinite(result.trace?.start)) {
          const at = Math.max(0, Math.min(candidate.duration - 10, result.trace.start))
          pin(candidate, candidate.first, at)
          ui.shortcuts.append(action("Matched moment", () => pin(candidate, candidate.first, at)))
        } else if (candidate) pin(candidate, candidate.first, null)
        else {
          const first = library.songs[0]
          pin(first, first.first, null)
        }
      }
    } catch {
      if (token === queryEpoch) {
        ui.summary.textContent = "This search couldn’t load. "
        ui.summary.append(action("Try again", () => follow(lastRequest)))
      }
    }
  }
  ui.example.onchange = () => {
    const example = examples.find((e) => String(e[0]) === ui.example.value)
    if (example) follow({ example: example[0], name: example[1] })
    else if (latest) follow(latest)
  }
  window.addEventListener("signal-search", (e) => {
    latest = e.detail
    ui.example.options[0].textContent = latest.name + " · player"
    follow(latest)
  })
  latest = selectedSearch()
  if (latest) ui.example.options[0].textContent = latest.name + " · player"
  follow(latest || { example: 2, name: examples[0][1] })
}
