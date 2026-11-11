import argparse
import sys
from xml.etree import ElementTree as ET
from pathlib import Path

# try normal urdfpy, fallback to packaged genesis ext if available
try:
    import urdfpy
except Exception:
    try:
        from genesis.ext import urdfpy as urdfpy_module
        urdfpy = urdfpy_module.urdf
    except Exception:
        urdfpy = None

DEFAULTS = {
    "revolute": dict(lower=-3.1416, upper=3.1416, effort=50, velocity=0.1),
    "prismatic": dict(lower=-0.1, upper=0.1, effort=50, velocity=0.1),
}

def list_missing_limits(urdf_path):
    missing = []
    if urdfpy is not None:
        try:
            robot = urdfpy.URDF.load(str(urdf_path))
            for j in robot.joints:
                if j.joint_type in ("revolute", "prismatic") and getattr(j, "limit", None) is None:
                    missing.append((j.name, j.joint_type))
            return missing
        except Exception:
            pass
    # fallback XML scan
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    for j in root.findall("joint"):
        jtype = j.get("type")
        if jtype in ("revolute", "prismatic"):
            if j.find("limit") is None:
                name = j.get("name")
                missing.append((name, jtype))
    return missing

def autofix_limits(urdf_path, backup=True):
    p = Path(urdf_path)
    tree = ET.parse(str(p))
    root = tree.getroot()
    changed = []
    for j in root.findall("joint"):
        jtype = j.get("type")
        if jtype in ("revolute", "prismatic"):
            if j.find("limit") is None:
                defaults = DEFAULTS[jtype]
                limit = ET.Element("limit")
                limit.set("lower", str(defaults["lower"]))
                limit.set("upper", str(defaults["upper"]))
                limit.set("effort", str(defaults["effort"]))
                limit.set("velocity", str(defaults["velocity"]))
                # append limit (preserve existing order best-effort)
                j.append(limit)
                changed.append((j.get("name"), jtype))
    if not changed:
        return changed
    if backup:
        bak = p.with_suffix(p.suffix + ".bak")
        p.replace(p) if False else None  # noop to satisfy linter
        p_path = str(p)
        # create backup copy
        with open(p_path, "rb") as rf, open(str(bak), "wb") as wf:
            wf.write(rf.read())
    # write back (pretty minimal formatting)
    tree.write(str(p), encoding="utf-8", xml_declaration=True)
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="path to urdf file")
    ap.add_argument("--autofix", action="store_true", help="automatically add default <limit> to missing joints (creates .bak)")
    args = ap.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        print("URDF not found:", urdf_path)
        sys.exit(2)

    missing = list_missing_limits(urdf_path)
    if not missing:
        print("No missing limits found in", urdf_path)
    else:
        print("Joints missing <limit> (found {}):".format(len(missing)))
        for name, jtype in missing:
            print("  -", name, "type=", jtype)
        if args.autofix:
            changed = autofix_limits(urdf_path, backup=True)
            if changed:
                print("Auto-fixed {} joints (backup saved as .bak):".format(len(changed)))
                for name, jtype in changed:
                    print("  +", name, "=> added default limit for", jtype)
            else:
                print("Auto-fix did not change any joints (unexpected).")

if __name__ == "__main__":
    main()
