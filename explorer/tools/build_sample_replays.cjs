// Run the local GPU service, then use the real browser decoder and recognition endpoint.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || "playwright")
const fs = require("node:fs/promises")
const path = require("node:path")

async function main() {
  const output = path.resolve(__dirname, "../public/sample-searches")
  await fs.mkdir(output, { recursive: true })
  const browser = await chromium.launch({ channel: "chrome", headless: true })
  try {
    const page = await browser.newPage()
    await page.goto("http://127.0.0.1:8327")
    const choices = page.locator("#demo-tracks button")
    await choices.first().waitFor({ state: "attached" })
    const ids = await choices.evaluateAll((bs) => bs.map((b) => b.value))
    for (const id of ids) {
      await page.locator("#list-toggle").click()
      await page.locator('#demo-tracks button[value="' + id + '"]').click()
      const saved = await page.evaluate(async () => {
        const { currentClip } = await import("/player.js")
        const clip = await currentClip()
        const pcm = Int16Array.from(clip.getChannelData(0), (v) =>
          Math.round(Math.max(-1, Math.min(1, v)) * (v < 0 ? 32768 : 32767)),
        )
        const response = await fetch("/api/recognize", {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: pcm,
        })
        if (!response.ok) throw Error("Recognition failed: " + response.status)
        const result = await response.json()
        if (result.trace?.library !== "medium-v1") throw Error("Expected a Medium search trace")
        const { waveform } = await import("/search-session.js")
        result.wave = waveform(clip.getChannelData(0))
        const hash = await crypto.subtle.digest("SHA-256", pcm.buffer)
        result.pcmHash = Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, "0")).join("")
        return result
      })
      await fs.writeFile(path.join(output, id + ".json"), JSON.stringify(saved))
      console.log(id, saved.track_id, saved.accepted)
    }
  } finally {
    await browser.close()
  }
}
main().catch((e) => {
  console.error(e)
  process.exitCode = 1
})
