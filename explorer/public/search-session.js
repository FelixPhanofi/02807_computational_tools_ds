// The player may choose a clip before the map finishes loading.
let current
export function publishSearch(detail) {
  current = detail
  window.dispatchEvent(new CustomEvent("signal-search", { detail }))
}
export function selectedSearch() {
  return current
}
export function stopListening() {
  window.dispatchEvent(new Event("signal-stop-audio"))
}

export function waveform(samples) {
  const bins = 96
  return Array.from({ length: bins }, (_, i) => {
    let peak = 0
    for (
      let j = Math.floor((i * samples.length) / bins);
      j < Math.floor(((i + 1) * samples.length) / bins);
      j++
    )
      peak = Math.max(peak, Math.abs(samples[j]))
    return Math.round(1000 * peak) / 1000
  })
}
