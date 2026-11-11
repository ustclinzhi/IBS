import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from sklearn.decomposition import PCA

from scipy.spatial import cKDTree
URDF_PATH = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\mano_hand_left_multi.urdf")
MESH_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes")
OUT_SNIPPET = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\origin_suggestions.txt")

SAMPLE_POINTS = 2000  # 可调：每个网格采样点数量，用于最近点搜索和 PCA
NEAR_RADIUS = 0.02    # m，PCA 时从 joint 附近采样点的半径

def load_mesh(name):
    p = MESH_DIR / name
    if not p.exists():
        return None
    m = trimesh.load_mesh(str(p), force='mesh', skip_materials=True)
    # 如果坐标明显大（如 >1），假设是 mm，转换到 m
    if np.max(np.abs(m.bounds)) > 1.0:
        m.apply_scale(0.001)
    return m

def parse_urdf(urdf_path):
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints = []
    for j in root.findall('joint'):
        name = j.get('name')
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        origin = j.find('origin')
        if origin is None:
            xyz = [0.0,0.0,0.0]; rpy=[0.0,0.0,0.0]
        else:
            xyz = [float(x) for x in origin.get('xyz','0 0 0').split()]
            rpy = [float(x) for x in origin.get('rpy','0 0 0').split()]
        axis = j.find('axis')
        axis_v = np.array([0.0,0.0,1.0]) if axis is None else np.array([float(x) for x in axis.get('xyz').split()])
        joints.append({'name':name,'parent':parent,'child':child,'origin_xyz':xyz,'origin_rpy':rpy,'axis':axis_v})
    return joints

def nearest_point_midpoint(meshA, meshB, samples=SAMPLE_POINTS):
    ptsA = meshA.sample(samples)
    ptsB = meshB.sample(samples)
    # build KD using scipy.spatial.cKDTree
    treeB = cKDTree(ptsB)
    dists, idxs = treeB.query(ptsA, k=1)
    min_idx = int(np.argmin(dists))
    pa = ptsA[min_idx]
    pb = ptsB[int(idxs[min_idx])]
    mid = 0.5*(pa + pb)
    return pa, pb, mid

def estimate_axis_near(mesh, center, radius=NEAR_RADIUS):
    # collect vertices within radius, fallback to sampled points
    verts = np.array(mesh.vertices)
    d = np.linalg.norm(verts - center[np.newaxis,:],axis=1)
    near = verts[d <= radius]
    if len(near) < 20:
        # use samples
        samp = mesh.sample(2000)
        d2 = np.linalg.norm(samp - center[np.newaxis,:],axis=1)
        near = samp[d2 <= radius]
    if len(near) < 10:
        return np.array([0.0,0.0,1.0])  # fallback
    pca = PCA(n_components=3)
    pca.fit(near)
    # first principal component direction
    axis = pca.components_[0]
    # ensure consistent direction (pointing roughly outward)
    return axis / np.linalg.norm(axis)

def write_snippets(snippets):
    with open(OUT_SNIPPET, 'w', encoding='utf-8') as f:
        for s in snippets:
            f.write(s+"\n\n")
    print("wrote", OUT_SNIPPET)

def main():
    joints = parse_urdf(URDF_PATH)
    # map link -> mesh name (assume mesh filename like link_joint_X.stl in URDF)
    # try to infer from URDF by searching <mesh filename="...">
    tree = ET.parse(str(URDF_PATH))
    root = tree.getroot()
    link_mesh = {}
    for link in root.findall('link'):
        ln = link.get('name')
        geo = link.find('.//geometry/mesh')
        if geo is not None:
            fname = os.path.basename(geo.get('filename'))
            link_mesh[ln] = fname

    # load meshes
    meshes = {}
    for ln,fname in link_mesh.items():
        m = load_mesh(fname)
        if m is None:
            print("missing mesh for", ln, fname)
        else:
            meshes[ln] = m

    snippets = []
    # assume root link (no parent) frame is world (identity). we'll maintain parent frame poses
    link_world_pos = {}   # link -> position of its frame in world
    link_world_rot = {}   # link -> rotation matrix (3x3); start root as identity for 'hand'
    # find root(s)
    children = {j['child'] for j in joints}
    parents = {j['parent'] for j in joints}
    roots = list(parents - children)
    if not roots:
        roots = ['hand']
    for r in roots:
        link_world_pos[r] = np.zeros(3)
        link_world_rot[r] = np.eye(3)

    # process joints in given order; for robustness iterate until all processed
    pending = joints.copy()
    processed = set()
    while pending:
        progress = False
        for j in pending[:]:
            parent = j['parent']; child = j['child']
            if parent not in link_world_pos:
                continue
            # need both parent and child meshes available
            if parent not in meshes or child not in meshes:
                # still generate best-effort origin at midpoint between their centroids
                if parent in meshes and child in meshes:
                    pass
                else:
                    pending.remove(j); processed.add(j['name']); progress = True
                    continue
            meshA = meshes[parent]; meshB = meshes[child]
            pa, pb, mid = nearest_point_midpoint(meshA, meshB)
            # estimate axis by PCA on combined near regions
            axisA = estimate_axis_near(meshA, pa)
            axisB = estimate_axis_near(meshB, pb)
            # choose axis as average (if opposite, flip)
            axis = axisA + axisB
            if np.linalg.norm(axis) < 1e-6:
                axis = axisA
            axis = axis / np.linalg.norm(axis)
            joint_world = mid
            # convert joint_world to parent frame coords
            R_parent = link_world_rot[parent]
            p_parent = link_world_pos[parent]
            joint_in_parent = R_parent.T.dot(joint_world - p_parent)
            axis_in_parent = R_parent.T.dot(axis)
            # normalize axis
            axis_in_parent = axis_in_parent / np.linalg.norm(axis_in_parent)
            # produce URDF snippet for this joint
            snip = (f"<!-- joint {j['name']} -->\n"
                    f"<joint name=\"{j['name']}\" type=\"revolute\">\n"
                    f"  <parent link=\"{parent}\"/>\n"
                    f"  <child link=\"{child}\"/>\n"
                    f"  <origin xyz=\"{joint_in_parent[0]:.6f} {joint_in_parent[1]:.6f} {joint_in_parent[2]:.6f}\" rpy=\"0 0 0\"/>\n"
                    f"  <axis xyz=\"{axis_in_parent[0]:.6f} {axis_in_parent[1]:.6f} {axis_in_parent[2]:.6f}\"/>\n"
                    f"</joint>")
            snippets.append(snip)
            # define child link frame = joint frame (world)
            link_world_pos[child] = joint_world.copy()
            link_world_rot[child] = R_parent.copy()  # keep orientation same as parent; you can rotate to align axis if desired
            # compute visual / inertial origin for child: transform mesh center to child frame
            com_world = np.array(meshB.center_mass)
            if np.max(np.abs(com_world)) > 1.0:
                com_world = com_world * 0.001
            com_in_child = link_world_rot[child].T.dot(com_world - link_world_pos[child])
            snip2 = (f"<!-- visual/inertial for link {child} -->\n"
                     f"<link name=\"{child}\">\n"
                     f"  <!-- suggested visual/collision/inertial origin (rpy=0) -->\n"
                     f"  <origin xyz=\"{com_in_child[0]:.6f} {com_in_child[1]:.6f} {com_in_child[2]:.6f}\" rpy=\"0 0 0\" />\n"
                     f"</link>")
            snippets.append(snip2)
            pending.remove(j)
            processed.add(j['name'])
            progress = True
        if not progress:
            print("stalled while processing joints; remaining:", [j['name'] for j in pending])
            break

    write_snippets(snippets)
    print("Done. 手动确认 rpy 及轴方向；如需把 child frame 的朝向与轴对齐可进一步用 PCA 调整 link_world_rot[child].")

if __name__ == "__main__":
    main()