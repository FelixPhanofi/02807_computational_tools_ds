import { stopListening } from "./search-session.js"
export function excerptPlayer(update) {
  let context,
    source,
    cacheId,
    decoded,
    controller,
    version = 0
  function stop() {
    version++
    controller?.abort()
    if (source) {
      source.onended = null
      try {
        source.stop()
      } catch {}
      source.disconnect()
      source = null
    }
    update("idle")
  }
  window.addEventListener("signal-stop-audio", stop)
  async function play(song, start) {
    stopListening()
    const token = version
    try {
      context ||= new AudioContext()
      await context.resume()
      if (token !== version) return
      if (cacheId !== song.id) {
        update("loading")
        controller = new AbortController()
        const res = await fetch("/api/sample?id=" + song.id, { signal: controller.signal })
        if (!res.ok) throw Error("The archive couldn’t load this recording. Try again or open its source.")
        const buffer = await context.decodeAudioData(await res.arrayBuffer())
        if (token !== version) return
        cacheId = song.id
        decoded = buffer
      }
      if (token !== version) return
      const at = Math.max(0, start),
        length = Math.min(10, decoded.duration - at)
      if (length <= 0) throw Error("This piece lies beyond the recording.")
      source = context.createBufferSource()
      source.buffer = decoded
      source.connect(context.destination)
      source.onended = () => {
        if (version === token) {
          source.disconnect()
          source = null
          update("idle")
        }
      }
      source.start(0, at, length)
      update("playing")
    } catch (e) {
      if (token === version) update("error", e.message)
    }
  }
  return { play, stop }
}
