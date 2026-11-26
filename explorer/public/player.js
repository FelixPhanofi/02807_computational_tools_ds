import { publishSearch, stopListening } from "./search-session.js"
const byId = (id) => document.getElementById(id)
const audio = new Audio()
let library = [],
  bag = [],
  song,
  currentURL,
  loading,
  aborter,
  selection = 0
let locked = false,
  ready = false

function buttons() {
  byId("preview").disabled = !song || locked
  byId("shuffle").disabled = !song || locked
  byId("list-toggle").disabled = !song || locked
  byId("try-example").disabled = !song || locked || !ready
  byId("demo-tracks").inert = locked
}
export function setPlayerState(isBusy, isAvailable) {
  locked = isBusy
  ready = isAvailable
  if (locked) audio.pause()
  buttons()
}

function drawPlayer() {
  const playing = !audio.paused && !audio.ended
  byId("player").classList.toggle("playing", playing)
  byId("play-icon").textContent = playing ? "Ⅱ" : "▶"
  byId("preview").setAttribute("aria-label", (playing ? "Pause " : "Play ") + (song?.title || "sample"))
  byId("player-time").textContent =
    "0:" + String(Math.floor(audio.currentTime || 0)).padStart(2, "0") + " / 0:10"
}
;["timeupdate", "play", "pause", "ended"].forEach((event) => audio.addEventListener(event, drawPlayer))

function choose(track) {
  selection++
  aborter?.abort()
  audio.pause()
  bag = bag.filter((t) => t !== track)
  audio.removeAttribute("src")
  audio.load()
  if (currentURL) URL.revokeObjectURL(currentURL)
  currentURL = undefined
  loading = undefined
  song = track
  byId("follow-result").hidden = true
  publishSearch({ name: track.title, sample: track.track_id })
  byId("sample-title").textContent = track.title
  byId("sample-artist").textContent = track.artist
  byId("player-note").textContent = ""
  byId("saved-sample").hidden = true
  byId("demo-tracks").hidden = true
  byId("list-toggle").setAttribute("aria-expanded", "false")
  for (const item of byId("demo-tracks").querySelectorAll("button"))
    item.setAttribute("aria-pressed", String(item.value === String(track.track_id)))
  buttons()
  drawPlayer()
}
function shuffle() {
  if (!bag.length) {
    bag = [...library]
    for (let i = bag.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[bag[i], bag[j]] = [bag[j], bag[i]]
    }
  }
  if (bag.length > 1 && bag.at(-1) === song) [bag[0], bag[bag.length - 1]] = [bag.at(-1), bag[0]]
  choose(bag.pop())
}
function pcmWave(buffer) {
  const values = buffer.getChannelData(0)
  const bytes = new DataView(new ArrayBuffer(44 + 2 * values.length))
  const word = (at, str) => [...str].forEach((c, i) => bytes.setUint8(at + i, c.charCodeAt(0)))
  word(0, "RIFF")
  bytes.setUint32(4, bytes.byteLength - 8, true)
  word(8, "WAVE")
  word(12, "fmt ")
  bytes.setUint32(16, 16, true)
  bytes.setUint16(20, 1, true)
  bytes.setUint16(22, 1, true)
  bytes.setUint32(24, buffer.sampleRate, true)
  bytes.setUint32(28, 2 * buffer.sampleRate, true)
  bytes.setUint16(32, 2, true)
  bytes.setUint16(34, 16, true)
  word(36, "data")
  bytes.setUint32(40, 2 * values.length, true)
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(-1, Math.min(1, values[i]))
    bytes.setInt16(44 + i * 2, Math.round(v * (v < 0 ? 32768 : 32767)), true)
  }
  return new Blob([bytes.buffer], { type: "audio/wav" })
}
export function currentClip() {
  if (loading) return loading
  const token = selection,
    id = song.track_id
  aborter = new AbortController()
  byId("player-note").textContent = "Loading ten seconds…"
  loading = (async () => {
    const response = await fetch(id === 20096 ? "/listen.wav" : "/api/sample?id=" + id, {
      signal: aborter.signal,
    })
    if (!response.ok) throw Error("This song couldn’t load. Try another song or the saved sample.")
    const offline = new OfflineAudioContext(1, 240000, 24000)
    const decoded = await offline.decodeAudioData(await response.arrayBuffer())
    const source = offline.createBufferSource()
    source.buffer = decoded
    source.connect(offline.destination)
    source.start(0, id === 20096 ? 0 : Math.min(4, Math.max(0, decoded.duration - 10)), 10)
    const clip = await offline.startRendering()
    if (token !== selection) throw new DOMException("Song changed", "AbortError")
    currentURL = URL.createObjectURL(pcmWave(clip))
    audio.src = currentURL
    byId("player-note").textContent = ""
    return clip
  })().catch((error) => {
    if (token === selection) {
      loading = undefined
      byId("player-note").textContent = "This song couldn’t load. Pick another or try again."
      byId("saved-sample").hidden = id === 20096
    }
    throw error
  })
  return loading
}
export function sampleName() {
  return song?.title || "Your sample"
}
let playbackEpoch = 0
window.addEventListener("signal-stop-audio", () => {
  playbackEpoch++
  audio.pause()
})
async function play() {
  if (!audio.paused) {
    audio.pause()
    return
  }
  stopListening()
  const token = selection,
    playToken = playbackEpoch
  try {
    await currentClip()
    if (token === selection && playToken === playbackEpoch && !locked) await audio.play()
  } catch (error) {
    if (error.name === "NotAllowedError")
      byId("player-note").textContent = "Your clip is ready. Tap play to listen."
  }
}
byId("preview").onclick = play
byId("shuffle").onclick = shuffle
byId("saved-sample").onclick = () => {
  choose(library.find((t) => t.track_id === 20096))
  play()
}
byId("list-toggle").onclick = () => {
  const list = byId("demo-tracks")
  list.hidden = !list.hidden
  byId("list-toggle").setAttribute("aria-expanded", String(!list.hidden))
}
async function init() {
  const response = await fetch("/listening-catalogue.json")
  if (!response.ok) throw Error()
  library = await response.json()
  const list = byId("demo-tracks")
  for (const track of library) {
    const li = document.createElement("li"),
      button = document.createElement("button")
    button.textContent = track.title
    button.value = track.track_id
    button.onclick = () => {
      choose(track)
      byId("preview").focus()
    }
    li.append(button)
    list.append(li)
  }
  shuffle()
}
init().catch(() => {
  byId("sample-title").textContent = "Samples couldn’t load"
  byId("player-note").textContent = "Reload the page to try again."
})
