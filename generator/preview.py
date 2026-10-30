"""Preview renders: collect world-space meshes into a scene.json and shoot it with
tools/render (three.js in headless Chromium). Coordinates are Source-style, z up."""
import atexit
import json
import os
import socket
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_DIR = os.path.join(REPO, "tools", "render")

_server = None


def _serve_root():
    """Serve the filesystem root on localhost so pages can fetch any absolute path."""
    global _server
    if _server:
        return _server[1]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", "/"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    _server = (proc, port)
    import time
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    return port


class Scene:
    def __init__(self, width=900, height=900, background=(0.13, 0.135, 0.15), ground_z=None):
        self.desc = {"width": width, "height": height, "background": list(background),
                     "meshes": [], "views": []}
        if ground_z is not None:
            self.desc["ground"] = {"z": float(ground_z), "size": 400, "color": [0.33, 0.33, 0.34]}

    def add(self, mesh, matrix=None, name="", **material):
        """mesh: meshkit.Mesh (uv in image space, y down). material: color, map (abs path),
        opacity, roughness, metalness, doubleSided, emissive, physical, renderOrder."""
        v, n = mesh.v, mesh.n
        if n is None:
            mesh.smooth()
            v, n = mesh.v, mesh.n
        if matrix is not None:
            m = np.asarray(matrix, float)
            v = v @ m[:3, :3].T + m[:3, 3]
            n = n @ np.linalg.inv(m[:3, :3])
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        uv = np.stack([mesh.uv[:, 0], 1 - mesh.uv[:, 1]], 1)
        f = mesh.f
        if matrix is not None and np.linalg.det(np.asarray(matrix)[:3, :3]) < 0:
            f = f[:, ::-1]
        if "map" in material and material["map"] and not material["map"].startswith("http"):
            material["map"] = os.path.abspath(material["map"])
        self.desc["meshes"].append({
            "name": name,
            "positions": np.round(v, 4).ravel().tolist(),
            "normals": np.round(n, 4).ravel().tolist(),
            "uvs": np.round(uv, 5).ravel().tolist(),
            "indices": f.ravel().astype(int).tolist(),
            "material": material,
        })

    def view(self, out, eye, target, fov=30, key=None):
        v = {"out": out, "eye": list(map(float, eye)), "target": list(map(float, target)), "fov": fov}
        if key is not None:
            v["key"] = list(map(float, key))
        self.desc["views"].append(v)

    def render(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        scene_path = os.path.abspath(os.path.join(out_dir, "scene.json"))
        with open(scene_path, "w") as fh:
            json.dump(self.desc, fh)
        port = _serve_root()
        url = f"http://127.0.0.1:{port}{RENDER_DIR}/viewer.html?scene={scene_path}"
        res = subprocess.run(["node", os.path.join(RENDER_DIR, "render.js"), url, os.path.abspath(out_dir)],
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or res.stdout)
        return res.stdout.split()
