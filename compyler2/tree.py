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
from typing import List, Any

from lexer import Token


def format_tree(tree, indent=0, pad="  ", name=None):
    if not name:
        name = tree.__class__.__name__

    if isinstance(tree, list):
        s = indent * pad + name
        indent += 1
        for item in tree:
            s += "\n" + format_tree(item, indent=indent, pad=pad)
        return s

    elif isinstance(tree, Token):
        return f"{indent * pad}{tree.type.name}: {tree.value}"

    elif not is_dataclass(tree):
        return f"{indent * pad}{name} {tree!r}"

    s = indent * pad + name
    indent += 1
    for field in fields(tree):
        value = getattr(tree, field.name)
        s += "\n" + format_tree(value, indent=indent, pad=pad, name=field.name)
    return s


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
    path: List[str]
    args: List[Any]  # Expression


@dataclass
class VarAccess:
    name: Token


@dataclass
class StructAccess:
    left: Any  # Struct?
    right: Token


@dataclass
class ArrayAccess:
    array: Any
    index: Any


@dataclass
class Struct:
    name: Token
    members: List  # VarDecl Struct
    union: bool
    inline: bool = False
    anon: bool = False

    # typechecker fields
    size: int = 0
    expanded: bool = False


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
    node: Any  # Expression
    precedence: int = 0


@dataclass
class BinOp:
    op: Token
    left: Any = None  # Expression
    right: Any = None  # Expression
    precedence: int = 0


@dataclass
class Assignment:
    op: Token
    left: Any
    right: Any


@dataclass
class TypeExpr:
    typename: Token
    is_new: bool
    assignments: Any


@dataclass
class TypeAccess:
    left: Any
    right: Token


@dataclass
class Block:
    statements: List  # Statement


@dataclass
class Try:
    statement: Any  # Statement
    catch_vardecl: VarDecl
    catch_stmt: Any  # Statement


@dataclass
class If:
    condition: Any  # Expression
    body: Any  # Statement
    else_body: Any  # Statement


@dataclass
class While:
    condition: Any  # Expression
    body: Any  # Statement
    label: str


@dataclass
class Deferred:
    statement: Any  # Statement
    on_err: bool


@dataclass
class LoopCtrl:
    out: bool
    label: Token


@dataclass
class FuncCtrl:
    value: Any  # Expression
    error: bool


@dataclass
class Delete:
    value: Any  # Expression


@dataclass
class VarDeclStmt:
    var_decl: VarDecl
    value: Any  # Expression


@dataclass
class Function:
    name: Token
    return_type: Type
    parameters: List[VarDecl]
    body: Any  # Statement


@dataclass
class Program:
    user_types: List  # Enum Struct
    functions: List  # Function

