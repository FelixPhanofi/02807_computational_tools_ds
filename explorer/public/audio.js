import { publishSearch, stopListening, waveform } from "./search-session.js"
import { currentClip, setPlayerState, sampleName } from "./player.js"
const seconds = 10
const sampleRate = 24000
const samplesNeeded = seconds * sampleRate
const get = (id) => document.getElementById(id)
const canvas = get("scope")
let liveSamples
let busy = false
let available = false
let cancelCapture

function draw() {
  if (!liveSamples || canvas.hidden) return
  const { width, height } = canvas.getBoundingClientRect()
  const dpr = Math.min(devicePixelRatio, 2)
  canvas.width = width * dpr
  canvas.height = height * dpr
  const ctx = canvas.getContext("2d")
  ctx.scale(dpr, dpr)
  ctx.strokeStyle = "#354036"
  ctx.lineWidth = 0.6
  for (let x = 0; x < width; x += 18) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }
  for (let y = 0; y < height; y += 18) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }
  const mid = height / 2
  for (let x = 1; x < width; x += 3) {
    const start = Math.floor((x / width) * liveSamples.length)
    const end = Math.min(liveSamples.length, start + Math.ceil((3 * liveSamples.length) / width))
    let peak = 0
    for (let j = start; j < end; j++) peak = Math.max(peak, Math.abs(liveSamples[j]))
    peak = Math.min(1, peak * 2.5)
    ctx.strokeStyle = "#b7d469"
    ctx.lineWidth = 1.5
    const h = Math.max(1, peak * mid * 0.9)
    ctx.beginPath()
    ctx.moveTo(x, mid - h)
    ctx.lineTo(x, mid + h)
    ctx.stroke()
  }
}
new ResizeObserver(draw).observe(canvas)

function status(title, detail, error = false) {
  get("status").dataset.state = error ? "error" : "info"
  get("status-title").textContent = title
  get("status-detail").textContent = detail
}
function controls() {
  get("record").disabled = (!available || busy) && !cancelCapture
  get("record").textContent = cancelCapture ? "Cancel recording" : "Record 10 seconds"
  get("upload").disabled = !available || busy
  setPlayerState(busy, available)
}

async function checkService() {
  try {
    const reply = await fetch("/api/health", { cache: "no-store", signal: AbortSignal.timeout(10000) })
    const health = await reply.json()
    available = reply.ok && health.recognition === true
  } catch {
    available = false
  }
  if (!busy) {
    status(
      available ? "Ready to listen" : "Recognition is offline",
      available ? "" : "Try again in a moment. You can still play the sample songs.",
      !available,
    )
    controls()
  }
}
controls()
checkService()
window.addEventListener("focus", () => {
  if (!available && !busy) checkService()
})

async function capture() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode)
    throw Error("Microphone capture needs HTTPS and a browser with Web Audio. You can also upload a clip.")
  status("Waiting for microphone", "Allow microphone access to start the ten-second capture.")
  const context = new AudioContext()
  let stream, source, processor, sink, watchdog
  try {
    await context.resume()
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    })
    await context.audioWorklet.addModule("/capture-worklet.js")
    const buffer = context.createBuffer(1, Math.round(context.sampleRate * seconds), context.sampleRate)
    source = context.createMediaStreamSource(stream)
    processor = new AudioWorkletNode(context, "ten-second-capture")
    sink = context.createGain()
    sink.gain.value = 0
    let offset = 0
    await new Promise((resolve, reject) => {
      cancelCapture = () => reject(Error("Capture cancelled."))
      controls()
      watchdog = setTimeout(() => reject(Error("Microphone stopped delivering audio. Try again.")), 20000)
      stream.getAudioTracks()[0].onended = () => reject(Error("Microphone disconnected."))
      status("Listening · 10 s", "Play your excerpt near the microphone.")
      processor.port.onmessage = ({ data }) => {
        buffer.copyToChannel(data.samples, 0, offset)
        offset += data.samples.length
        canvas.hidden = false
        liveSamples = data.samples
        draw()
        status(
          "Listening · " + Math.ceil((buffer.length - offset) / context.sampleRate) + " s",
          "Play your excerpt near the microphone.",
        )
        if (data.done) resolve()
      }
      source.connect(processor).connect(sink).connect(context.destination)
    })
    return buffer
  } finally {
    clearTimeout(watchdog)
    cancelCapture = undefined
    source?.disconnect()
    processor?.disconnect()
    sink?.disconnect()
    stream?.getTracks().forEach((track) => track.stop())
    await context.close()
  }
}
async function decodeUpload(file) {
  if (file.size > 8 * 1024 * 1024 || !file.size) throw Error("Choose a nonempty audio file of up to 8 MiB.")
  const metadata = new Audio()
  const url = URL.createObjectURL(file)
  try {
    const duration = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error("Audio metadata could not be read.")), 8000)
      metadata.onloadedmetadata = () => {
        clearTimeout(timer)
        resolve(metadata.duration)
      }
      metadata.onerror = () => {
        clearTimeout(timer)
        reject(Error("Choose a browser-supported audio file, such as WAV or MP3."))
      }
      metadata.preload = "metadata"
      metadata.src = url
    })
    if (Number.isFinite(duration) && (duration < seconds || duration > 30))
      throw Error("Choose an audio excerpt between 10 and 30 seconds long.")
    if (!Number.isFinite(duration))
      throw Error("This file has no readable duration. Export it as a short WAV or MP3.")
    const ctx = new OfflineAudioContext(1, samplesNeeded, sampleRate)
    const buffer = await ctx.decodeAudioData(await file.arrayBuffer())
    if (buffer.duration < seconds || buffer.duration > 30.01)
      throw Error("Decoded audio must be 10–30 seconds long.")
    return buffer
  } finally {
    metadata.removeAttribute("src")
    metadata.load()
    URL.revokeObjectURL(url)
  }
}
async function recognize(buffer, name) {
  status("Looking for your song…", "The first search after a break can take about two minutes.")
  const offline = new OfflineAudioContext(1, samplesNeeded, sampleRate)
  const source = offline.createBufferSource()
  source.buffer = buffer
  source.connect(offline.destination)
  source.start(0, 0, seconds)
  const mono = (await offline.startRendering()).getChannelData(0)
  liveSamples = mono
  draw()
  const bytes = new DataView(new ArrayBuffer(2 * samplesNeeded))
  for (let i = 0; i < samplesNeeded; i++) {
    const v = Math.max(-1, Math.min(1, mono[i]))
    bytes.setInt16(2 * i, Math.round(v * (v < 0 ? 32768 : 32767)), true)
  }
  const reply = await fetch("/api/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: bytes.buffer,
    signal: AbortSignal.timeout(240000),
  })
  if (!reply.ok) {
    const message = await reply.json().catch(() => ({}))
    throw Error(
      typeof message.error === "string"
        ? message.error
        : "Recognition service returned HTTP " + reply.status + ".",
    )
  }
  const result = await reply.json()
  if (typeof result.accepted !== "boolean") throw Error("Unexpected recognition response.")
  const title = result.accepted ? result.title || "Track " + result.track_id : "No clear match"
  const label = result.accepted
    ? result.artist || "Found in the music library."
    : "We couldn’t tell which song this was. Try a clearer part of the recording."
  status(title, label)
  result.wave = waveform(mono)
  publishSearch({ name, result })
  get("follow-result").hidden = false
}
async function run(getBuffer, name) {
  if (busy || !available) return
  stopListening()
  get("follow-result").hidden = true
  busy = true
  canvas.hidden = true
  controls()
  try {
    await recognize(await getBuffer(), name)
  } catch (error) {
    status("Recognition stopped", error.message, true)
  } finally {
    busy = false
    canvas.hidden = true
    controls()
  }
}
get("record").onclick = () => (cancelCapture ? cancelCapture() : run(capture, "Your recording"))
get("upload").onclick = () => get("file").click()
get("file").onchange = (e) => {
  const file = e.target.files[0]
  e.target.value = ""
  if (file) run(() => decodeUpload(file), "Your upload")
}

get("try-example").onclick = () => run(currentClip, sampleName())
