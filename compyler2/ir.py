# ir.py
#
# Copyright (C) 2019 Pauli Kauro
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""This module contains the intermediate representation."""


import logging
from dataclasses import dataclass, field, fields, InitVar
from typing import Set, List


# types will be registered into this dictionary
_types = {}


def get_type(name: str):
    # TODO: try?
    return _types[name]


@dataclass(frozen=True)
class IRType:
    """Base class for IR types"""
    name: str
    size: int

    def __post_init__(self):
        assert self.name not in _types
        _types[self.name] = self

    def __str__(self):
        return self.name


# some basic types
Void = IRType("Void", 0)
Unit = IRType("Unit", 0)
Func = IRType("Func", 0)
Bool = IRType("Bool", 1)


@dataclass(frozen=True)
class IntType(IRType):
    """Base integer type"""
    signed: bool


class PointerType(IntType):
    """Each target should have an instance of this class"""
    def __init__(self, target_name: str, size: int):
        super().__init__(name=f"Ptr_{target_name}", size=size, signed=False)

    def __str__(self):
        return "Ptr"


# integer types
U8 = IntType("U8", 1, False)
I8 = IntType("I8", 1, True)
U16 = IntType("U16", 2, False)
I16 = IntType("I16", 2, True)
U32 = IntType("U32", 4, False)
I32 = IntType("I32", 4, True)
U64 = IntType("U64", 8, False)
I64 = IntType("I64", 8, True)


# TODO: floating point types?


# metadata for dataclass fields that are input edges
_graph_input_key = "graph_input"
_graph_input = {_graph_input_key: True}


@dataclass(eq=False)
class Node:
    """Base class for all nodes in the IR graph."""

    def __post_init__(self, type: IRType=None):
        """This should always and only be called from a subclass or by the
        __init__ method generated by a dataclass decorator."""

        assert type is not None, "please override __post_init__"
        self.type = type
        self.outputs = []

        # doubly link the graph
        for input in self.inputs:
            input.add_output(self)

    def __eq__(self, other):
        """Checks if two nodes are equal.

        Two nodes compare equal if they have the same type and their inputs
        are identical.
        """

        if not isinstance(other, Node):
            return NotImplemented

        if self.type is not other.type:
            return False

        if len(self.inputs) != len(other.inputs):
            return False

        return all(in1 is in2 for in1, in2 in zip(self.inputs, other.inputs))

    def __hash__(self):
        ids = (id(inp) for inp in self.inputs)
        return hash((self.type, tuple(ids)))

    def __iter__(self):
        return reversed(list(reversed(self)))

    def __reversed__(self, visited=None):
        if visited is None:
            visited = set()

        elif id(self) in visited:
            return

        visited.add(id(self))
        for out in self.outputs:
            yield from out.__reversed__(visited=visited)

        yield self

    def add_output(self, output: "Node"):
        self.outputs.append(output)

    def del_output(self, old_out: "Node"):
        self.outputs = [out for out in self.outputs if out is not old_out]

    def replace(self, node: "Node"):
        """Replaces this node"""

        # delete inputs' outputs
        for inp in self.inputs:
            inp.del_output(self)

        # replace outputs' inputs
        for out in self.outputs:
            out.replace_input(self, node)

    @property
    def inputs(self):
        return [getattr(self, in_field.name) for in_field in fields(self)
                if in_field.metadata.get(_graph_input_key, False)]

    def replace_input(self, old: "Node", new: "Node"):
        for in_field in fields(self):
            if not in_field.metadata.get(_graph_input_key, False):
                continue

            if getattr(self, in_field.name) is old:
                setattr(self, in_field.name, new)
                new.add_output(self)

    def __str__(self):
        return self.__class__.__name__


@dataclass(eq=False)
class Module(Node):
    """A module is the start node in a graph."""

    name: str

    def __post_init__(self):
        super().__post_init__(type=Void)

    def __eq__(self, other):
        # there is no good reason to use anything else for this
        return self is other

    def __hash__(self):
        return hash(id(self))

    @property
    def functions(self):
        # TODO: inefficient and does not update
        return {func.name: func for func in self.outputs}


@dataclass(eq=False)
class Function(Node):
    module: Module = field(metadata=_graph_input)
    name: str
    ret_type: IRType

    def __post_init__(self):
        super().__post_init__(Func)

    @property
    def parameters(self):
        return self.inputs[1:]

    # TODO: need to override eq and hash?

    def __str__(self):
        return f"Function {self.name}"


@dataclass(eq=False)
class Parameter(Node):
    function: Function = field(metadata=_graph_input)
    type: InitVar[IRType]
    name: str

    def __str__(self):
        return f"Parameter {self.name}"


@dataclass(eq=False)
class Alloca(Node):
    ctrl: Node = field(metadata=_graph_input)
    size: Node = field(metadata=_graph_input)
    type: InitVar[IRType]


@dataclass(eq=False)
class Load(Node):
    ctrl: Node = field(metadata=_graph_input)
    addr: Node = field(metadata=_graph_input)
    type: InitVar[IRType]


@dataclass(eq=False)
class Store(Node):
    ctrl: Node = field(metadata=_graph_input)
    addr: Node = field(metadata=_graph_input)
    value: Node = field(metadata=_graph_input)

    def __post_init__(self):
        super().__post_init__(Unit)


@dataclass(eq=False)
class CtrlSync(Node):
    ctrl: Node = field(metadata=_graph_input)
    data: Node = field(metadata=_graph_input)

    def __post_init__(self):
        super().__post_init__(self.data.type)


@dataclass(eq=False)
class Phi(Node):
    _inputs: List[CtrlSync]

    def __post_init__(self):
        assert self.inputs

        first_type = self.inputs[0].type
        for inp in self.inputs[1:]:
            assert inp.type is first_type

        super().__post_init__(first_type)

    @property
    def inputs(self):
        return self._inputs

    def replace_input(self, old: "CtrlSync", new: "CtrlSync"):
        for idx, inp in enumerate(self.inputs):
            if inp is old:
                self._inputs[idx] = new


@dataclass(eq=False)
class Constant(Node):
    function: Function = field(metadata=_graph_input)
    type: InitVar[IRType]
    value: int

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False

        if self.value != other.value:
            return False

        return super().__eq__(other)

    def __hash__(self):
        return hash((super(), self.value))

    def __str__(self):
        return f"Constant {self.value}"


@dataclass(eq=False)
class BinOp(Node):
    left: Node = field(metadata=_graph_input)
    right: Node = field(metadata=_graph_input)

    def __post_init__(self):
        assert self.left.type is self.right.type
        super().__post_init__(self.left.type)


class CommutativeMixin:
    def __eq__(self, other):
        normal = self.left is other.left and self.right is other.right
        commutative = self.left is other.right and self.right is other.left
        if (normal or commutative) and self.type is other.type:
            return True
        return False

    def __hash__(self):
        # dumb hash implementation
        return hash((self.type, id(self.left) ^ id(self.right)))


@dataclass(eq=False)
class Add(CommutativeMixin, BinOp):
    pass


@dataclass(eq=False)
class Sub(BinOp):
    pass


@dataclass(eq=False)
class Mul(CommutativeMixin, BinOp):
    pass


# things for making IR analyses and transformations easier
class PassMeta(type):
    def __new__(mcs, name, bases, attrs):
        from functools import singledispatch

        @singledispatch
        def visit_method(self, node):
            pass

        def do_visit(self, node):
            visit_method.dispatch(type(node))(self, node)

        assert "visit" not in attrs
        attrs["visit"] = do_visit

        for attr_name, attr in attrs.items():
            if not (attr_name.startswith("visit_") and callable(attr)):
                # looking for methods whose name start with 'visit'
                continue

            try:
                node_type = attr.__annotations__["node"]
            except KeyError:
                continue

            visit_method.register(node_type)(attr)

        return super().__new__(mcs, name, bases, attrs)


class ForwardsPass(metaclass=PassMeta):
    def run(self, graph: Node):
        for node in graph:
            self.visit(node)


class BackwardsPass(metaclass=PassMeta):
    def run(self, graph: Node):
        for node in reversed(graph):
            self.visit(node)


@dataclass
class BackwardsValidator(BackwardsPass):
    visited: Set[int] = field(default_factory=set)

    def visit_node(self, node: Node):
        assert id(node) not in self.visited
        assert all(id(inp) not in self.visited for inp in node.inputs)
        self.visited.add(id(node))


@dataclass
class ForwardsValidator(ForwardsPass):
    visited: Set[int] = field(default_factory=set)

    def visit_node(self, node: Node):
        assert id(node) not in self.visited
        assert all(id(out) not in self.visited for out in node.outputs)
        self.visited.add(id(node))


class DotGraphPass(BackwardsPass):
    def __init__(self, ignore=None, allow=None):
        from pydot import Dot
        self.dot = Dot(graph_type="digraph")
        self.ignore = ignore
        self.allow = allow

    def allowed(self, node):
        if self.ignore and type(node) in self.ignore:
            return False

        if self.allow and type(node) not in self.allow:
            return False

        return True

    def visit_node(self, node: Node):
        if not self.allowed(node):
            return

        from pydot import Node, Edge
        dot_node = Node(id(node), label=str(node))
        self.dot.add_node(dot_node)
        for out in node.outputs:
            if not self.allowed(out):
                continue

            dot_edge = Edge(id(node), id(out))
            self.dot.add_edge(dot_edge)

    def write_png(self, filename):
        self.dot.write_png(filename)


class RedundancyEliminationPass(ForwardsPass):
    def __init__(self):
        self.seen = {}

    def visit_node(self, node: Node):
        if node not in self.seen:
            self.seen[node] = node
            return

        new = self.seen[node]
        node.replace(new)

