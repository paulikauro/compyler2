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


from dataclasses import dataclass, fields, is_dataclass
from typing import List, Dict, Any, Union
from collections import namedtuple

from lexer import Token


# used by the type checker and code generator
RecordMember = namedtuple("RecordMember", "name type offset")


def format_tree(tree, indent=0, pad="  ", name=""):
    if name:
        name += ": "
    name += tree.__class__.__name__

    s = indent * pad + name
    if isinstance(tree, list):
        indent += 1
        for item in tree:
            s += "\n" + format_tree(item, indent=indent, pad=pad)
        return s

    elif isinstance(tree, Token):
        return f"{s} {tree.type.name} {tree.value}"

    elif not is_dataclass(tree):
        return f"{s} {tree!r}"

    indent += 1
    for field in fields(tree):
        value = getattr(tree, field.name)
        s += "\n" + format_tree(value, indent=indent, pad=pad, name=field.name)
    return s


# useful type aliases
# TODO
Expression = Any
Statement = Any


@dataclass
class Enum:
    name: Token
    members: List[Token]
    inline: bool = False


@dataclass
class Type:
    name: Token
    ptr_level: int


@dataclass
class VarDecl:
    name: Token
    var_type: Type


@dataclass
class Call:
    func: Any
    args: List[Expression]


@dataclass
class VarAccess:
    name: Token


@dataclass
class StructAccess:
    left: Any  # Record?
    right: Token


@dataclass
class ArrayAccess:
    array: Any
    index: Any


@dataclass
class TypeAccess:
    left: Any
    right: Token


@dataclass
class Record:
    name: Token
    members: List[Union[VarDecl, "Record"]]
    union: bool
    inline: bool = False
    anon: bool = False

    # typechecker fields
    size: int = 0
    expanded: List[RecordMember] = None


@dataclass
class NumLiteral:
    value: Token
    is_char: bool = False
    precedence: int = 0


@dataclass
class StrLiteral:
    value: Token
    precedence: int = 0


@dataclass
class UnOp:
    op: Token
    node: Expression
    precedence: int = 0


@dataclass
class BinOp:
    op: Token
    left: Expression = None
    right: Expression = None
    precedence: int = 0


@dataclass
class Assignment:
    op: Token
    left: Expression
    right: Expression


@dataclass
class TypeExpr:
    type: Type
    is_new: bool
    assignments: Any


@dataclass
class ArrayAlloc:
    type: Type
    is_new: bool
    size_expr: Expression


@dataclass
class TypeConversion:
    type: Type
    is_new: bool
    init_expr: Expression


@dataclass
class Block:
    statements: List[Statement]


@dataclass
class Try:
    statement: Statement
    catch_vardecl: VarDecl
    catch_stmt: Statement


@dataclass
class If:
    condition: Expression
    body: Statement
    else_body: Statement


@dataclass
class While:
    condition: Expression
    body: Statement
    label: str


@dataclass
class Deferred:
    statement: Statement
    on_err: bool


@dataclass
class LoopCtrl:
    out: bool
    label: Token


@dataclass
class FuncCtrl:
    value: Expression
    error: bool


@dataclass
class Delete:
    value: Expression


@dataclass
class Function:
    name: Token
    return_type: Type
    parameters: List[VarDecl]
    body: Statement


@dataclass
class Module:
    enum_types: Dict[Token, Enum]
    record_types: Dict[Token, Record]
    functions: Dict[Token, Function]

