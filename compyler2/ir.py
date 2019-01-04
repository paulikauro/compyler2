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


from dataclasses import dataclass
from typing import Dict


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


# some very basic types
Void = IRType("Void", 0)
Unit = IRType("Unit", 0)


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


@dataclass
class Module:
    functions: Dict[str, "Function"]


class Function:
    pass



