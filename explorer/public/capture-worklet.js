class TenSecondCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.remaining = 10 * sampleRate;
    this.chunk = new Float32Array(4096);
    this.used = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels.length) return true;
    const frames = Math.min(this.remaining, channels[0].length);
    for (let i = 0; i < frames; i++) {
      let mono = 0;
      for (const channel of channels) mono += channel[i];
      this.chunk[this.used++] = mono / channels.length;
      this.remaining--;
      if (this.used === this.chunk.length || !this.remaining) {
        const samples = this.chunk.subarray(0, this.used);
        this.port.postMessage({ samples, done: !this.remaining }, [samples.buffer]);
        this.chunk = new Float32Array(4096);
        this.used = 0;
      }
    }
    return this.remaining > 0;
  }
}
registerProcessor("ten-second-capture", TenSecondCapture);

