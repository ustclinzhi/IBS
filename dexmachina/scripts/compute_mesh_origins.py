import os
import json
from pathlib import Path

import trimesh
import numpy as np

ASSET_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes")
OUT_JSON = ASSET_DIR / "mesh_centroids_suggestions.json"
OUT_SNIPPETS = ASSET_DIR / "mesh_origin_urdf_snippets.txt"

EXTS = [".stl", ".obj", ".ply", ".glb", ".dae", ".off"]

def safe_load_mesh(p):
    try:
        m = trimesh.load_mesh(str(p), force='mesh', skip_materials=True)
        if m.is_empty:
            return None
        return m
    except Exception as e:
        print(f"failed to load {p.name}: {e}")
        return None

def main():
    if not ASSET_DIR.exists():
        print(f"Asset dir not found: {ASSET_DIR}")
        return

    results = {}
    snippets = []

    for f in sorted(ASSET_DIR.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue
        m = safe_load_mesh(f)
        if m is None:
            continue
        centroid = np.asarray(m.centroid).tolist()
        bbox_centroid = np.asarray(m.bounding_box.centroid).tolist()
        # Suggest origin to shift mesh so centroid moves to link origin:
        suggested_origin = [-float(round(c, 6)) for c in centroid]
        suggested_origin_bbox = [-float(round(c, 6)) for c in bbox_centroid]

        results[f.name] = {
            "file": str(f),
            "centroid": centroid,
            "bbox_centroid": bbox_centroid,
            "suggested_origin_centroid": suggested_origin,
            "suggested_origin_bbox_centroid": suggested_origin_bbox,
        }

        # URDF snippet (use meshes/<name> or adjust path as needed)
        snip = f"""<!-- visual snippet for {f.name} -->
<visual>
  <origin xyz="{suggested_origin[0]:.6f} {suggested_origin[1]:.6f} {suggested_origin[2]:.6f}" rpy="0 0 0"/>
  <geometry>
    <mesh filename="meshes/{f.name}" />
  </geometry>
</visual>
<collision>
  <origin xyz="{suggested_origin[0]:.6f} {suggested_origin[1]:.6f} {suggested_origin[2]:.6f}" rpy="0 0 0"/>
  <geometry>
    <mesh filename="meshes/{f.name}" />
  </geometry>
</collision>

"""
        snippets.append(snip)

    # write outputs
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    with open(OUT_SNIPPETS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(snippets))

    # print short summary
    print(f"Processed {len(results)} meshes. JSON -> {OUT_JSON}")
    print(f"URDF snippets -> {OUT_SNIPPETS}")
    print("如果你希望基于 bbox_centroid 而非 centroid，请查看 JSON 字段 'suggested_origin_bbox_centroid'。")

if __name__ == "__main__":
    main()