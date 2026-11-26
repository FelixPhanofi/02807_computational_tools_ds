"""Build the small redistributable demo from Medium tracks labelled CC0 by FMA."""
from pathlib import Path
import json
import sys
import wave
import zipfile
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from cpu_match import peaks

def main():
    """Index 24 tracks and export an attributed ten-second CC0 listening example."""
    root = HERE.parent/"data/medium/data"
    with zipfile.ZipFile(root/"raw/fma_metadata.zip") as archive:
        metadata = pd.read_csv(archive.open("fma_metadata/tracks.csv"),header=[0,1],index_col=0,low_memory=False)
    rows = pd.read_csv(root/"processed/manifest.csv").set_index("track_id")
    eligible = rows.index.intersection(metadata.index[metadata.track.license.eq("CC0 1.0 Universal")])
    rows = rows.loc[eligible].query("duration_s >= 20").sample(n=24,random_state=2807).sort_index()
    features = []
    for i,(tid,row) in enumerate(rows.iterrows()):
        pcm = np.load(root/"processed"/row.path)
        assert pcm.dtype == np.int16
        features.append(peaks(pcm.astype("f4")/32768))
        if i == 0:
            with wave.open(str(HERE/"public/listen.wav"),"wb") as audio:
                audio.setparams((1,2,24000,0,"NONE","not compressed"))
                audio.writeframes(pcm[96000:336000].astype("<i2").tobytes())
    np.savez_compressed(HERE/"landmarks.npz",ids=rows.index, titles=rows.title.to_numpy(dtype=str),
                        artists=rows.artist.to_numpy(dtype=str), features=np.concatenate(features),
                        ends=np.cumsum([len(f) for f in features]))
    (HERE/"public/listening-catalogue.json").write_text(rows.reset_index()[["track_id","title","artist"]].to_json(orient="records"),encoding="utf8")

if __name__=="__main__":
    main()

