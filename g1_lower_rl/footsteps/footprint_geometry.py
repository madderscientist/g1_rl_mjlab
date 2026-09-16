"""从 G1 模型读取足底碰撞胶囊，并表达在各自足端参考点的平面坐标系"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from g1_lower_rl import PKG_PATH

MODEL_PATH = PKG_PATH / "assets/g1_gloria/xmls/g1_with_dual_gloria_m.xml"


def load_footprint_geometry(path: Path = MODEL_PATH) -> dict:
  """读取模型足底胶囊的平面投影和长宽，返回单位为米的可序列化几何数据"""
  root = ET.parse(path).getroot()
  collision = root.find(".//default[@class='collision']/geom")
  defaults = root.find(".//default[@class='foot_capsule']/geom")
  if collision is None or collision.get("type") != "capsule" or defaults is None:
    raise ValueError("Expected G1 foot_capsule collision defaults")
  feet = []
  for side in ("left", "right"):
    body = root.find(f".//body[@name='{side}_ankle_roll_link']")
    if body is None:
      raise ValueError(f"Missing {side} ankle roll body")
    site = body.find(f"site[@name='{side}_foot']")
    if site is None or any(name in site.attrib for name in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
      raise ValueError(f"Expected an unrotated {side}_foot site in ankle roll body")
    origin = [float(value) for value in site.attrib["pos"].split()]
    if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
      raise ValueError("Invalid foot site position")
    capsules = []
    for geom in body.findall("geom[@class='foot_capsule']"):
      if geom.get("type", "capsule") != "capsule":
        raise ValueError("Footprint expects capsule collision geometry")
      radius = float(geom.get("size", defaults.attrib["size"]).split()[0])
      endpoints = [float(value) for value in geom.attrib["fromto"].split()]
      if len(endpoints) != 6 or not all(math.isfinite(value) for value in endpoints) or not 0 < radius < math.inf:
        raise ValueError("Invalid foot capsule")
      if not math.isclose(endpoints[2], endpoints[5], abs_tol=1e-9):
        raise ValueError("Footprint expects capsules parallel to the sole plane")
      # 命令位置指向足端参考点，模型胶囊端点却以踝关节连杆为原点
      capsules.append(
        {
          "start": [endpoints[0] - origin[0], endpoints[1] - origin[1]],
          "end": [endpoints[3] - origin[0], endpoints[4] - origin[1]],
          "radius": radius,
        }
      )
    if not capsules:
      raise ValueError(f"No {side} foot collision capsules")
    lower = [
      min(min(capsule["start"][axis], capsule["end"][axis]) - capsule["radius"] for capsule in capsules) for axis in (0, 1)
    ]
    upper = [
      max(max(capsule["start"][axis], capsule["end"][axis]) + capsule["radius"] for capsule in capsules) for axis in (0, 1)
    ]
    feet.append(
      {
        "side": side,
        "site": f"{side}_foot",
        "site_position_in_link": origin,
        "capsules": capsules,
        "bounds": [lower, upper],
        "length": upper[0] - lower[0],
        "width": upper[1] - lower[1],
      }
    )
  return {"source": path.name, "representation": "mjcf_collision_capsules_xy", "units": "m", "feet": feet}
