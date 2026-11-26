module.exports = (_req, res) => {
  res.setHeader("Cache-Control", "no-store");
  res.json({ recognition: true, engine: process.env.SIGNAL_GPU_URL
    ? "GPU when online · 24-track landmark backup"
    : "Spectral landmarks · 24-track demo" });
};

