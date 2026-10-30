"""Source-preserved joint-name, mimic and limit-sign mapping."""
import re
import xml.etree.ElementTree as ET
import numpy as np

def urdf_limits(path):
    out = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        lim = joint.find("limit")
        if lim is not None and joint.get("type") in ("revolute", "prismatic"):
            out[joint.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out

def build_map(cfg, dof_names, lower, upper, ref_limits):
    """source index -> [(dof index, sign)], plus the evidence for every sign."""
    plan, evidence = {}, {}
    for i, src in enumerate(cfg["source_order"]):
        targets = [src]
        for rule in cfg.get("rename", []):
            targets = [re.sub(rule["pattern"], rule["replace"], t) for t in targets]
        for rule in cfg.get("mimic", []):
            if re.match(rule["pattern"], src):
                targets.append(re.sub(rule["pattern"], rule["also"], src))
        plan[i] = []
        for t in targets:
            if t not in dof_names:
                evidence[t] = {"source": src, "status": "not a DOF of the articulation (mimic consumed?)"}
                continue
            j = dof_names.index(t)
            lo_u, hi_u = ref_limits[t]
            overlap = {}
            for s in (1.0, -1.0):
                a, b = sorted((s * lo_u, s * hi_u))
                overlap[s] = max(0.0, min(b, float(upper[j])) - max(a, float(lower[j])))
            sign = max(overlap, key=overlap.get)
            plan[i].append((j, sign))
            evidence[t] = {"source": src, "sign": sign, "urdf_limits": [lo_u, hi_u],
                           "articulation_limits": [float(lower[j]), float(upper[j])],
                           "overlap_plus": overlap[1.0], "overlap_minus": overlap[-1.0]}
    return plan, evidence
