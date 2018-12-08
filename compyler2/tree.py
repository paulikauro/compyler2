# tree.py
#
# Copyright (C) 2018 Pauli Kauro
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


from dataclasses import dataclass
from typing import List

from lexer import Token


pad = "    "


@dataclass
class Enum:
    name: Token
    members: List[str]
    inline: bool = False

    def pretty(self, p=""):
        s = f"{p}enum {self.name.value}\n"
        p += pad
        for member in self.members:
            s += f"{p}{member}\n"
        return s


@dataclass
class Type:
    name: Token
    ptr_level: int

    def pretty(self):
        return self.name.value + self.ptr_level * "*"

    __str__ = pretty


@dataclass
class VarDecl:
    name: Token
    var_type: Type

    def pretty(self, p=""):
        return f"{p}{self.var_type} {self.name.value}\n"

    __str__ = pretty


@dataclass
class Struct:
    name: Token
    members: List  # VarDecl Struct Union
    union: bool
    inline: bool = False
    anon: bool = False

    def pretty(self, p=""):
        s = p
        s += "union " if self.union else "struct "
        if not self.anon:
            s += self.name.value
        s += "\n"
        p += pad
        for member in self.members:
            s += member.pretty(p)
        return s

    __str__ = pretty


@dataclass
class Program:
    enums: List[Enum]
    structs: List  # Enum Struct

    def pretty(self, p=""):
        s = "\nprogram\n"
        p += pad
        for enum in self.enums:
            s += enum.pretty(p)
        for struct in self.structs:
            s += struct.pretty(p)
        return s

    __str__ = pretty

