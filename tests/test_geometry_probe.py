"""Live-probe serialization handles explicit geometry and rejects empty traversal results."""

from types import SimpleNamespace

import numpy as np
import pytest

from vlarobot.backends.webots.geometry_probe import snapshot


class Field:
    def __init__(self, value):
        self.value = value

    def getSFNode(self):
        return self.value

    def getMFNode(self, i):
        return self.value[i]

    def getCount(self):
        return len(self.value)

    def getSFVec3f(self):
        return self.value

    def getSFRotation(self):
        return self.value

    def getSFString(self):
        return self.value

    def getSFBool(self):
        return self.value


class Node:
    def __init__(self, identifier, kind, **fields):
        self.identifier, self.kind, self.fields = identifier, kind, fields

    def getId(self):
        return self.identifier

    def getTypeName(self):
        return self.kind

    def getBaseTypeName(self):
        return self.kind

    def isProto(self):
        return False

    def getField(self, name):
        return Field(self.fields[name]) if name in self.fields else None

    def getPose(self, *args):
        return np.eye(4).reshape(-1).tolist()

    def getDef(self):
        return "FIXTURE"


def test_probe_serializes_numeric_geometry_without_motor_actions():
    box = Node(3, "Box", size=[0.1, 0.2, 0.3])
    robot = Node(2, "Robot", name="fixture", boundingObject=box, selfCollision=False)
    root = Node(1, "Group", children=[robot])
    supervisor = SimpleNamespace(
        getSelf=lambda: robot,
        getRoot=lambda: root,
        getNumberOfDevices=lambda: 0,
        getTime=lambda: 0.032,
    )
    report = snapshot(supervisor)
    assert report["motor_commands_sent"] == 0
    assert len(report["objects"]) == 1
    assert report["objects"][0]["shapes"][0]["parameters"]["size"] == [0.1, 0.2, 0.3]


def test_probe_does_not_call_empty_traversal_complete():
    robot = Node(2, "Robot", name="fixture", selfCollision=False)
    root = Node(1, "Group", children=[])
    supervisor = SimpleNamespace(
        getSelf=lambda: robot,
        getRoot=lambda: root,
        getNumberOfDevices=lambda: 0,
        getTime=lambda: 0.032,
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        snapshot(supervisor)
