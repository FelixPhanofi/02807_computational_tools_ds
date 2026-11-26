const base = "/search-atlas/"
export async function fetchData(url, binary = false) {
  const response = await fetch(url)
  if (!response.ok) throw Error("Couldn’t load " + url)
  return binary ? response.arrayBuffer() : response.json()
}
const files = new Map()
function cached(key, read) {
  if (!files.has(key))
    files.set(
      key,
      read().catch((e) => {
        files.delete(key)
        throw e
      }),
    )
  return files.get(key)
}
export function loadPlanes() {
  return cached("planes", async () => new Float32Array(await fetchData(base + "checks.bin", true)))
}
export function loadMap(table) {
  return cached(table, async () => {
    const keys = new Uint16Array(await fetchData(base + "map-" + table + ".bin", true))
    const bins = Array.from({ length: 1024 }, () => [])
    keys.forEach((key, row) => bins[key].push(row))
    return { keys, bins, counts: bins.map((b) => b.length) }
  })
}
export async function loadLibrary() {
  const d = await fetchData(base + "songs.json")
  const owners = new Int32Array(d.windows),
    byId = new Map()
  let first = 0
  const songs = d.songs.map(([id, title, artist, genre, duration, starts, url, license], index) => {
    const s = {
      id,
      title,
      artist,
      genre,
      duration,
      starts,
      url,
      rights: d.licenses[license],
      first,
      term: (title + " " + artist + " " + id).toLocaleLowerCase(),
    }
    owners.fill(index, first, first + starts.length)
    first += starts.length
    byId.set(id, s)
    return s
  })
  if (first !== d.windows) throw Error("Inconsistent library")
  return { songs, byId, owners, total: first, songAt: (row) => songs[owners[row]] }
}
export async function loadSearch(request) {
  const r =
    request.result ||
    (await fetchData(
      request.sample
        ? "/sample-searches/" + request.sample + ".json"
        : base + "example-" + request.example + ".json",
    ))
  const trace = r.trace
  if (trace?.kind !== "lsh") return { result: r, trace: null }
  if (trace.library !== "medium-v1") throw Error("The search uses a different library")
  const raw = Uint8Array.from(atob(trace.scores), (c) => c.charCodeAt(0))
  return { result: r, trace, scores: new Int16Array(raw.buffer) }
}
export const scoreAt = (scores, row) => (scores && scores[row] !== -32768 ? scores[row] / 32767 : null)
export const fmt = (n) => n.toLocaleString("en")
export const clock = (n) => Math.floor(n / 60) + ":" + String(Math.floor(n % 60)).padStart(2, "0")
export function make(tag, text, cls) {
  const el = document.createElement(tag)
  if (text != null) el.textContent = text
  if (cls) el.className = cls
  return el
}
export function action(label, fn, cls) {
  const b = make("button", label, cls)
  b.type = "button"
  b.onclick = fn
  return b
}
