import pandas as pd
from pathlib import Path
def make_report(bundles, out_file:Path):
    rows=[{"model":b.name,"rms":b.rms}|b.timings for b in bundles]
    df=pd.DataFrame(rows).set_index("model")
    out_file.write_text("<h1>Overlay Summary</h1>"+df.to_html(float_format="%.3f"))
    print(f"[INFO] report saved: {out_file}")