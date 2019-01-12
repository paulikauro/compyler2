# tree.py
#
# Copyright (C) 2018-2019 Pauli Kauro
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


from dataclasses import dataclass, field, fields, is_dataclass
from typing import List, Dict, Set, Any, Union

import ir
from lexer import Token, TokenType
from targets import Target


# TODO: this breaks after type checking
def format_tree(tree, indent=0, pad="  ", name=""):
    if name:
        name += ": "
    name += tree.__class__.__name__

    s = indent * pad + name
    if isinstance(tree, (list, dict)):
        if isinstance(tree, dict):
            tree = tree.values()
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

    # for type checking & code gen
    values: Dict[str, int] = None
    type: ir.IntType = ir.U8

    @property
    def size(self):
        return self.type.size


@dataclass(frozen=True)
class Type:
    name: Token
    ptr_level: int
    member: Token = field(compare=False, default=None)

    @staticmethod
    def make(name: str, ptr_level: int):
        return Type(Token(TokenType.TYPENAME, name, -1, -1), ptr_level)

    def __str__(self):
        name = self.name.value
        if self.member:
            name += "." + self.member.value
        return f"{name}{self.ptr_level * '*'}"


@dataclass
class VarDecl:
    name: Token
    var_type: Type


# TODO: factor out the type and throws fields into an Expression base class
# currently this is not possible because dataclasses do not allow fields with
# no default values in a subclass whose superclass has defined default values

@dataclass
class Call:
    func: Token
    args: List[Expression]

    type: Type = None
    throws: Set[Type] = None


@dataclass
class VarAccess:
    name: Token

    type: Type = None
    throws: Set[Type] = None


@dataclass
class StructAccess:
    left: Union[VarAccess, "StructAccess", "ArrayAccess"]  # TODO: all exprs
    right: Token

    type: Type = None
    throws: Set[Type] = None


@dataclass
class ArrayAccess:
    array: Union[VarAccess, "StructAccess", "ArrayAccess"]
    index: Expression

    type: Type = None
    throws: Set[Type] = None


# used by the type checker and code generator
@dataclass
class RecordMember:
    name: str
    type: str
    offset: int
    ptr_level: int = 0


@dataclass
class Record:
    name: Token
    members: List[Union[VarDecl, "Record"]]
    union: bool
    inline: bool = False
    anon: bool = False

    # typechecker fields
    size: int = 0
    expanded: Dict[str, RecordMember] = None


@dataclass
class NumLiteral:
    value: Token
    is_char: bool = False
    precedence: int = 0

    type: Type = None
    throws: Set[Type] = None


@dataclass
class StrLiteral:
    value: Token
    precedence: int = 0

    type: Type = None
    throws: Set[Type] = None


@dataclass
class UnOp:
    op: Token
    node: Expression
    precedence: int = 0

    type: Type = None
    throws: Set[Type] = None


@dataclass
class BinOp:
    op: Token
    left: Expression = None
    right: Expression = None
    precedence: int = 0

    type: Type = None
    throws: Set[Type] = None


@dataclass
class Assignment:
    op: Token
    left: Expression
    right: Expression

    type: Type = None
    throws: Set[Type] = None


@dataclass
class TypeExpr:
    type: Type
    is_new: bool
    assignments: List[Assignment]

    throws: Set[Type] = None


@dataclass
class ArrayAlloc:
    type: Type
    is_new: bool
    size_expr: Expression

    throws: Set[Type] = None


@dataclass
class TypeConversion:
    type: Type
    is_new: bool
    init_expr: Expression

    throws: Set[Type] = None


@dataclass
class Block:
    statements: List[Statement]

    throws: Set[Type] = None


@dataclass
class Try:
    statement: Statement
    catches: List["Catch"]

    throws: Set[Type] = None


@dataclass
class Catch:
    vardecl: VarDecl
    stmt: Statement


@dataclass
class If:
    condition: Expression
    body: Statement
    else_body: Statement

    throws: Set[Type] = None


@dataclass
class While:
    condition: Expression
    body: Statement
    label: Token

    throws: Set[Type] = None


@dataclass
class Deferred:
    statement: Statement
    on_err: bool

    throws: Set[Type] = None


@dataclass
class LoopCtrl:
    out: bool
    label: Token

    throws: Set[Type] = None


@dataclass
class FuncCtrl:
    value: Expression
    error: bool

    throws: Set[Type] = None


@dataclass
class Delete:
    value: Expression

    throws: Set[Type] = None


@dataclass
class Function:
    name: Token
    return_type: Type
    parameters: List[VarDecl]
    throws: Set[Type]
    body: Statement

    checked: bool = False


@dataclass
class Module:
    enum_types: Dict[str, Enum]
    record_types: Dict[str, Record]
    functions: Dict[str, Function]

    # used by typechecker and code gen
    target: Target = None

    def ir_type(self, type: Type) -> ir.IRType:
        if type.ptr_level > 0:
            return self.target.Ptr
        return self.get_type(type.name.value)

    def get_type(self, name: str, **kwargs):
        # checks in order to prevent lots of try-except

        if name in self.enum_types:
            return self.enum_types[name]

        if name in self.record_types:
            return self.record_types[name]

        try:
            return self.target.get_type(name)
        except KeyError:
            pass

        try:
            return ir.get_type(name)
        except KeyError:
            if "default" in kwargs:
                return kwargs["default"]
            raise

