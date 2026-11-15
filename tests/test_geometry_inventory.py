"""Geometry audit must preserve unresolved source references and malformed URDF failures."""

import pytest

from vlarobot.backends.webots.geometry_inventory import proto_inventory, urdf_inventory


def test_named_solids_and_def_use_are_not_silently_certified(tmp_path):
    p = tmp_path / "robot.proto"
    p.write_text(
        """PROTO Example [] { Solid { name "arm"
      boundingObject Group { children [ Box { size 0.1 0.2 0.3 } USE UNKNOWN ] }
    }}""",
        encoding="utf-8",
    )
    result = proto_inventory(p)
    assert result["bounding_objects"][0]["owner"] == "arm"
    assert result["primitive_counts"] == {"Box": 1}
    assert result["bounding_objects"][0]["unresolved_geometry"]
    assert not result["source_geometry_fully_resolved"]
    p.write_text("PROTO Example [] { Solid { boundingObject USE UNKNOWN }}", encoding="utf-8")
    assert proto_inventory(p)["unparsed_bounding_object_statements"] == 1


def test_urdf_fixed_joints_remain_data_not_automatic_self_collision_exclusions(tmp_path):
    p = tmp_path / "robot.urdf"
    p.write_text(
        """<robot name="test"><link name="a"><collision><geometry>
      <box size="0.1 0.1 0.1"/></geometry></collision></link><link name="b"/>
      <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint></robot>""",
        encoding="utf-8",
    )
    result = urdf_inventory(p)
    assert result["joints"][0]["type"] == "fixed"
    assert "exclusions" not in result
    p.write_text('<robot><joint name="broken" type="fixed"/></robot>', encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete link coverage"):
        urdf_inventory(p)
