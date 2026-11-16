# ~30 s in Colab. Builds the 500-prompt decode subset (salted hash, deterministic)
# and uploads it and the decode spec to the a5-data volume.
import hashlib, json, pathlib, subprocess, sys

sys.path.insert(0, "/content/a5_code")   # run.py must already be uploaded here
import importlib.util
spec_mod = importlib.util.spec_from_file_location("a5_run", "/content/a5_code/run.py")
run = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(run)

SALT = "a5-ws2-week7-decode-v1"
N = 500
prompts = run.load_prompt_rows()
ids = sorted({str(p["prompt_id"]) for p in prompts},
             key=lambda pid: hashlib.sha256((SALT + pid).encode()).hexdigest())[:N]
ids = sorted(ids)
subset = {"prompt_ids": ids, "n": len(ids), "salt": SALT,
          "selection": "sha256(salt + prompt_id), first 500, then sorted"}
out = pathlib.Path("/content/llama31-8b_decode_subset.json")
out.write_text(json.dumps(subset, indent=2))
print(f"decode subset: {len(ids)} prompts")

for local in [out, pathlib.Path("/content/a5_code/llama31-8b-decode.yaml")]:
    subprocess.run(["modal", "volume", "put", "--force", "a5-data",
                    str(local), "/" + local.name if local == out else "/code/" + local.name],
                   check=True)
print("uploaded decode subset and decode spec to a5-data")
