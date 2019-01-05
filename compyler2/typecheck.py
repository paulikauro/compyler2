# typecheck.py
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


import logging
import dataclasses
from functools import singledispatch
from typing import Union
from collections import ChainMap

from lexer import FrontendError, Token, TokenType
from tree import Module, Record, RecordMember, Enum, VarDecl, Type, Function,\
    Block, If, Try, While, Deferred, LoopCtrl, FuncCtrl, Delete,\
    Call, VarAccess, StructAccess, ArrayAccess, NumLiteral, StrLiteral,\
    UnOp, BinOp, Assignment, TypeConversion, TypeExpr, ArrayAlloc
from targets import Target
import ir


class Scopes(ChainMap):
    """A simple class for managing scopes while traversing the AST."""
    # TODO: is this needed?

    def __init__(self, *scopes):
        super().__init__(*scopes)
        self.modified_keys = set()

    def __setitem__(self, key, value):
        # TODO: this is inefficient
        if key in self.parents:
            # keep track of modifications done to parents
            self.modified_keys.add(key)
        super().__setitem__(key, value)

    @property
    def current(self):
        return self.maps[0]


def perform_check(module: Module, target: Target):
    module.target = target
    check_module(module)


@singledispatch
def check(node, *args, **kwargs):
    logging.warning(f"check({type(node)}, {len(args)}, {len(kwargs)})")
    node.throws = {"_undefined"}
    node.type = Type(Token(None, "_undefined", -1, -1), 0)


@check.register
def check_module(module: Module):
    def check_name(names, name):
        if name.value in names:
            raise FrontendError(f"Name {name.value} is already used",
                                name.line, name.col)
        names.add(name.value)

    # check user-defined types
    user_type_names = set()
    # TODO: is list() necessary?
    # (enum_types and record_types are modified during checking)
    for enum in list(module.enum_types.values()):
        check_name(user_type_names, enum.name)
        check(enum, module)

    for record in list(module.record_types.values()):
        check_name(user_type_names, record.name)
        check(record, module)

    # initialize the global scope
    global_scope = Scopes()
    func_names = set()

    # check function names
    for func in module.functions.values():
        check_name(func_names, func.name)
        global_scope[func.name.value] = func

    # check functions
    for func in module.functions.values():
        check(func, module, global_scope)


@check.register
def check_enum(enum: Enum, mod: Module):
    """Checks an enum.

    Enum member uniqueness is checked, during which they are allocated
    a numeric id.
    """

    enum.values = {}
    counter = 0
    for member in enum.members:
        if member.value in enum.values:
            raise FrontendError(f"Enum member {member.value} declared twice",
                                member.line, member.col)
        enum.values[member.value] = counter
        counter += 1

    if counter > 256:
        # maximum allowed is 256 members
        raise FrontendError(f"Enum {enum.name.value} has more than 256 members",
                            enum.name.line, enum.name.col)


@check.register
def check_record(record: Record, mod: Module, seen=None):
    """Checks, inlines and extracts records.

    The size of the record and the offsets of its fields will be calculated.
    Fields of anonymous inline records will be inlined. Non-anonymous inline
    records will be extracted into their own record in the module. During this
    process the record is checked for recursive definitions. Lastly, the record
    will be checked for duplicate field names.
    """

    # for error messages
    record_type_str = "union" if record.union else "struct"

    if seen is None:
        # create new set (mutable default argument bad)
        seen = set()

    if not record.inline:
        # not inline => must be a top-level definition
        # (anonymous records are marked inline as well)
        # check for recursion
        if record.name.value in seen:
            raise FrontendError(f"Recursive {record_type_str} definition",
                                record.name.line, record.name.col)
        # add to seen set
        seen.add(record.name.value)

    if not record.expanded:
        # skip if already expanded
        # note: condition inverted because clean-up code below
        # flatten (expand) recursively
        record.expanded = {}
        record.size = 0
        for member in record.members:
            # both Record and ir.IRType have a size field
            member_type: Union[Record, ir.IRType]
            member_offset = 0
            ptrlevel = 0

            if isinstance(member, Enum):
                # an inline enum: check and extract
                enum_type = member
                check_enum(enum_type, mod)
                enum_tok = enum_type.name
                enum_name = f"{record.name.value}.{enum_tok.value}"
                # note: does not change name inside `member`
                mod.enum_types[enum_name] = member
                # this may be an ugly hack
                enum_type_tok = dataclasses.replace(enum_tok, value=enum_name)
                member = VarDecl(enum_tok, Type(enum_type_tok, 0))

            if isinstance(member, VarDecl):
                ptrlevel = member.var_type.ptr_level
                type_name = member.var_type.name.value
                if type_name in mod.record_types.keys() and ptrlevel == 0:
                    # record type in VarDecl, not a pointer
                    # recursively expand
                    rec_type = mod.record_types[type_name]
                    # will be skipped if already expanded,
                    # but the recursion check is inside flatten_record,
                    # so it is kept here for now
                    check_record(rec_type, mod, seen)
                    member_type = rec_type
                else:
                    # other type in VarDecl
                    try:
                        member_type = mod.get_type(type_name)
                    except KeyError:
                        raise FrontendError(f"Invalid type",
                                            member.var_type.name.line,
                                            member.var_type.name.col)

            else:
                assert isinstance(member, Record), f"{member} is not a Record!"
                # an inline record type
                # recurse
                check_record(member, mod, seen)
                if member.anon:
                    # anonymous: inline members
                    # this is handled as a special case as there are multiple
                    # members to handle
                    for inline_name, inline_member in member.expanded.items():
                        # adjust offset
                        inline_member.offset += member_offset
                        record.expanded[inline_name] = inline_member
                    # done
                    continue
                else:
                    # not anonymous; extract fields into a new record
                    # replace name
                    extracted_name = f"{record.name.value}.{member.name.value}"
                    # create a new token for the name, storing location info
                    extracted_tok = dataclasses.replace(member.name,
                                                        value=extracted_name)
                    mod.record_types[extracted_tok] = member
                    member_type = member

            # use the correct size for pointers
            ptr_size = mod.target.Ptr.size
            member_size = member_type.size if ptrlevel == 0 else ptr_size
            # update offset and size
            if record.union:
                # in a union, all offsets are 0 (default)
                # and size is the size of the largest member
                if member_size > record.size:
                    record.size = member_size
            else:
                # struct, so lay out members normally
                member_offset = record.size
                record.size += member_size

            # add the processed member
            if isinstance(member_type, (Record, Enum)):
                new_name = member_type.name.value
            else:
                assert isinstance(member_type, ir.IRType)
                new_name = member_type.name

            new_member = RecordMember(member.name.value, new_name,
                                      member_offset, ptrlevel)
            record.expanded[new_member.name] = new_member

    if not record.inline:
        # clean up
        seen.remove(record.name.value)

    # check for duplicate names
    names = set()
    for name in record.expanded.keys():
        if name in names:
            # TODO: make line/col info more precise
            raise FrontendError(f"duplicate {record_type_str} field {name!r}",
                                record.name.line, record.name.col)


def valid_type(type, ptr_level: int = 0, tok: Token = None):
    """Checks if the type is valid to use in user code."""
    line, col = -1, -1
    if tok:
        line = tok.line
        col = tok.col

    if type is ir.Void:
        raise FrontendError("Void type not allowed", line, col)
    elif type is ir.Func and ptr_level == 0:
        raise FrontendError("Func type only allowed as a pointer", line, col)


def valid_ast_type(ast_type: Type, mod: Module):
    type = mod.get_type(ast_type.name.value)
    valid_type(type, ast_type.ptr_level, ast_type.name)


@check.register
def check_func(func: Function, mod: Module, global_scope: Scopes):
    """Typechecks a function.

    The return and parameter, uniqueness of parameter names, throws-declaration
    consistency and the statements contained by the function are checked.
    """

    # check return type
    valid_ast_type(func.return_type, mod)
    # initialize a scope
    scopes = global_scope.new_child()

    # check parameters
    for param in func.parameters:
        valid_ast_type(param.var_type, mod)
        param_name = param.name.value
        if param_name in scopes.current:
            raise FrontendError(f"Duplicate function parameter {param_name}",
                                param.name.line, param.name.col)
        scopes[param_name] = param.var_type

    # check function body
    check(func.body, mod, scopes, ret_type=func.return_type)

    # check throws
    if func.body.throws != func.throws:
        raise FrontendError(
            f"incorrect throws-declaration for function {func.name}",
            func.name.line, func.name.col)


@check.register
def check_block(block: Block, mod: Module, scopes: Scopes, **kw):
    """Checks a block of statements."""

    # blocks introduce a new scope
    new_scopes = scopes.new_child()
    block.throws = set()
    for stmt in block.statements:
        check(stmt, mod, new_scopes, **kw)
        block.throws.update(stmt.throws)


def cond_stmt_check(stmt: Union[If, While], mod: Module, scopes: Scopes, **kw):
    """Factored helper function for checking conditional statements."""

    check(stmt.condition, mod, scopes, **kw)
    if stmt.condition.type != ir.Bool:
        # TODO: insert implicit comparison?
        raise FrontendError("conditions must have type Bool")

    check(stmt.body, mod, scopes, **kw)
    stmt.throws = stmt.body.throws | stmt.condition.throws


@check.register
def check_if(stmt: If, mod: Module, scopes: Scopes, **kw):
    cond_stmt_check(stmt, mod, scopes, **kw)

    if stmt.else_body:
        check(stmt.else_body, mod, scopes, **kw)
        stmt.throws.update(stmt.else_body.throws)


@check.register
def check_while(stmt: While, mod: Module, scopes: Scopes, in_loop=None, **kw):
    label = stmt.label.value
    if in_loop:
        if label in in_loop:
            raise FrontendError("loop labels must be unique",
                                stmt.label.line, stmt.label.col)
    else:
        in_loop = set()

    in_loop.add(label)
    # condition will receive in_loop as well, but it doesn't matter
    cond_stmt_check(stmt, mod, scopes, in_loop=in_loop, **kw)
    in_loop.remove(label)


@check.register
def check_try(stmt: Try, mod: Module, scopes: Scopes, **kw):
    """Checks a try statement and propagates exception info."""

    # check statements
    check(stmt.statement, mod, scopes, **kw)
    # which types are caught?
    catch_types = set()
    # which types are thrown in the catch statements?
    catch_throws = set()

    for catch in stmt.catches:
        check(catch.stmt, mod, scopes, **kw)
        catch_throws.update(catch.stmt.throws)

        valid_ast_type(catch.vardecl.var_type, mod)
        catch_types.add(catch.vardecl.var_type)

    # types which are thrown in the try block but not caught
    escaping = stmt.statement.throws - catch_types
    # merge with types thrown in catch blocks to get all thrown types
    stmt.throws = escaping | catch_throws


@check.register
def check_deferred(stmt: Deferred, mod: Module, scopes: Scopes,
                   in_defer=False, **kw):
    """Checks defer and errdefer statements."""

    if in_defer:
        raise FrontendError("nesting deferred statements is not allowed")

    check(stmt.statement, mod, scopes, in_defer=True, **kw)
    if stmt.statement.throws:
        raise FrontendError("deferred statements may not throw")

    stmt.throws = set()


@check.register
def check_loopctrl(stmt: LoopCtrl, mod: Module, scopes: Scopes,
                   in_loop=None, **kw):
    """Checks break and continue statements."""

    if not in_loop:
        raise FrontendError("loop control statements must be inside loops")

    if stmt.label and stmt.label.value not in in_loop:
        raise FrontendError(f"invalid loop label {stmt.label.value}",
                            stmt.label.line, stmt.label.col)
    # never throws
    stmt.throws = set()


@check.register
def check_funcctrl(stmt: FuncCtrl, mod: Module, scopes: Scopes,
                   ret_type=None, **kw):
    """Checks return and throw statements."""

    assert ret_type, "needs return type"

    if stmt.value:
        check(stmt.value, mod, scopes, **kw)
        stmt.throws = set(stmt.value.throws)
    else:
        assert not stmt.error, "parser was supposed to handle this"
        stmt.throws = set()

    if stmt.error:
        if stmt.value.type != ret_type:
            raise FrontendError("wrong or invalid return type")
    else:
        stmt.throws.add(stmt.value.type)


@check.register
def check_delete(stmt: Delete, mod: Module, scopes: Scopes, **kw):
    check(stmt.value, mod, scopes, **kw)
    if stmt.value.type.ptr_level == 0:
        raise FrontendError("cannot delete non-pointer type")
    stmt.throws = set(stmt.value.throws)


# expressions

@check.register
def check_call(expr: Call, mod: Module, scopes: Scopes, **kw):
    try:
        func = mod.functions[expr.func.value]
    except KeyError:
        raise FrontendError(f"function {expr.func.value} does not exist",
                            expr.func.line, expr.func.col)

    # check arguments
    for arg, param in zip(expr.args, func.parameters):
        check(arg, mod, scopes, **kw)
        if arg.type != param.var_type:
            raise FrontendError(f"invalid type for argument {param.name.value}"
                                f" in call to {func.name.value}, expected"
                                f" {param.var_type}, got {arg.type}",
                                expr.func.line, expr.func.col)

    expr.type = func.return_type
    expr.throws = func.throws


@check.register
def check_varaccess(expr: VarAccess, mod: Module, scopes: Scopes,
                    must_exist=True, **kw):

    name = expr.name.value
    expr.throws = set()

    try:
        expr.type = scopes[name]
    except KeyError:
        if not must_exist:
            return
        raise FrontendError(f"variable {name} not in scope",
                            expr.name.line, expr.name.col)


@check.register
def check_structaccess(expr: StructAccess, mod: Module, scopes: Scopes,
                       must_exist=True, **kw):
    check(expr.left, mod, scopes, must_exist=True, **kw)

    # ptr_level can be anything because of automatic dereferencing
    type_name = expr.left.type.name.value
    try:
        record = mod.record_types[type_name]
    except KeyError:
        raise FrontendError(f"{type_name} is not a struct",
                            expr.right.line, expr.right.col) from None

    member_name = expr.right.value
    try:
        member = record.expanded[member_name]
    except KeyError:
        raise FrontendError(f"{type_name} has no field {member_name}",
                            expr.right.line, expr.right.col) from None

    # TODO: fix this in record type checking
    expr.type = Type.make(member.type, member.ptr_level)
    expr.throws = set(expr.left.throws)


@check.register
def check_arrayaccess(expr: ArrayAccess, mod: Module, scopes: Scopes,
                      must_exist=True, **kw):
    check(expr.array, mod, scopes, must_exist=True, **kw)

    ptr_level = expr.array.type.ptr_level
    type_name = expr.index.type.name.value

    if ptr_level == 0:
        raise FrontendError(f"array access on a non-pointer type"
                            f" {expr.array.type}")

    check(expr.index, mod, scopes, **kw)

    if not isinstance(mod.get_type(type_name), ir.IntType):
        raise FrontendError("array index must have an integer type")

    # TODO: insert cast to correct width

    # array access dereferences
    expr.type = Type.make(type_name, ptr_level - 1)
    expr.throws = expr.array.throws | expr.index.throws


@check.register
def check_numliteral(expr: NumLiteral, mod: Module, scopes: Scopes, **kw):
    expr.type = Type.make("Constant", 0)
    expr.throws = set()


@check.register
def check_strliteral(expr: StrLiteral, mod: Module, scopes: Scopes, **kw):
    # type is U8*
    expr.type = Type.make("U8", 1)
    expr.throws = set()


@check.register
def check_unop(expr: UnOp, mod: Module, scopes: Scopes, **kw):
    # possible operations (TokenTypes):
    # NOT (from if/while condition invert hack), MINUS, BNOT, STAR, BAND

    check(expr.node)
    expr.throws = expr.node.throws

    if expr.node.type.name == "Constant":
        # TODO
        raise FrontendError("please cast constants before using unary ops",
                            expr.op.line, expr.op.col)

    if expr.op.type is TokenType.NOT:
        # TODO
        raise NotImplementedError

    type_name = expr.node.type.name
    ptr_level = expr.node.type.ptr_level

    if expr.op.type is TokenType.STAR:
        # dereference
        if ptr_level == 0:
            raise FrontendError("cannot dereference non-pointer type",
                                expr.op.line, expr.op.col)

        expr.type = Type.make(type_name, ptr_level - 1)
        return

    if expr.op.type is TokenType.BAND:
        # addressof
        expr.type = Type.make(type_name, ptr_level + 1)
        return

    # rest of the operations require an integer type
    irtype = mod.get_type(type_name, default=None)

    if not irtype or not isinstance(irtype, ir.IntType):
        raise FrontendError("requires an integer operand",
                            expr.op.line, expr.op.col)

    if expr.op.type is TokenType.MINUS:
        if not irtype.signed:
            raise FrontendError("unary minus can only be applied to signed"
                                " integer types", expr.op.line, expr.op.col)

    else:
        assert expr.op.type is TokenType.BNOT

    expr.type = expr.node.type


@check.register
def check_binop(expr: BinOp, mod: Module, scopes: Scopes, **kw):
    assert expr.left and expr.right

    check(expr.left, mod, scopes, **kw)
    check(expr.right, mod, scopes, **kw)

    # type conversions
    left_const = expr.left.type.name == "Constant"
    right_const = expr.right.type.name == "Constant"
    if left_const and right_const:
        raise FrontendError("please cast at least one operand",
                            expr.op.line, expr.op.col)

    if left_const:
        conv = TypeConversion(expr.right.type, is_new=False,
                              init_expr=expr.left)
        conv.throws = set(expr.left.throws)
        expr.left = conv

    elif right_const:
        conv = TypeConversion(expr.left.type, is_new=False,
                              init_expr=expr.right)
        conv.throws = set(expr.right.throws)
        expr.right = conv

    # check types
    if expr.left.type != expr.right.type:
        raise FrontendError("binary op operands must have the same type",
                            expr.op.line, expr.op.col)

    expr.type = expr.left.type
    expr.throws = expr.left.throws | expr.right.throws


@check.register
def check_assignment(expr: Assignment, mod: Module, scopes: Scopes,
                     lhs_scope=None, **kw):

    struct_assign = False
    if not lhs_scope:
        # for struct assignments
        struct_assign = True
        lhs_scope = scopes

    check(expr.right, mod, scopes, **kw)
    check(expr.left, mod, lhs_scope, must_exist=struct_assign, **kw)

    expr.throws = expr.right.throws | expr.left.throws

    if not expr.left.type:
        assert isinstance(expr.left, VarAccess)
        assert not struct_assign
        expr.left.type = expr.right.type
    else:
        # check types
        if expr.left.type != expr.right.type:
            raise FrontendError(f"tried to assign {expr.left.type} to"
                                f" {expr.right.type}")

    if not isinstance(expr.left, VarAccess):
        return

    try:
        # is this variable in scope already?
        scope_type = lhs_scope[expr.left.name.value]
    except KeyError:
        assert not struct_assign, "cannot add struct members in with-assign"
        # nope, create a mapping
        lhs_scope[expr.left.name.value] = expr.left.type
        return

    # yes, check types
    if scope_type != expr.left.type:
        raise FrontendError(f"tried to assign {expr.left.type} to"
                            f" {expr.left.name.value} which has type"
                            f" {scope_type}",
                            expr.left.name.line, expr.left.name.col)


@check.register
def check_typeexpr(expr: TypeExpr, mod: Module, scopes: Scopes, **kw):
    valid_ast_type(expr.type, mod)
    expr.throws = set()
    old_ptr_level = expr.type.ptr_level

    if expr.assignments:
        if expr.type.name.value not in mod.record_types or old_ptr_level > 0:
            raise FrontendError("can only assign members to direct struct or"
                                " union types",
                                expr.type.name.col, expr.type.name.line)

        record = mod.record_types[expr.type.name.value]

        # create a special scope for assignments
        assign_scope = Scopes()
        for member_name, member in record.expanded.items():
            member_type = Type.make(member.type, member.ptr_level)
            assign_scope[member_name] = member_type

        for assignment in expr.assignments:
            check_assignment(assignment, mod, scopes,
                             lhs_scope=assign_scope, **kw)
            expr.throws.update(assignment.throws)

    if expr.is_new:
        # add exception to throws
        expr.throws.add(Type.make("MemoryFail", 0))
        # increment pointer level
        expr.type = dataclasses.replace(expr.type, ptr_level=old_ptr_level + 1)


@check.register
def check_arrayalloc(expr: ArrayAlloc, mod: Module, scopes: Scopes, **kw):
    valid_ast_type(expr.type, mod)
    check(expr.size_expr, mod, scopes, **kw)
    expr.throws = set(expr.size_expr.throws)

    size_irtype = mod.get_type(expr.size_expr.type.name)
    size_is_ptr = expr.size_expr.type.ptr_level > 0

    if size_irtype is not ir.IntType or size_irtype.signed or size_is_ptr:
        raise FrontendError("array size must be an unsigned integer",
                            expr.type.name.line, expr.type.name.col)

    if expr.is_new:
        expr.throws.add(Type.make("MemoryFail", 0))

    # arrays are pointers regardless of whether or not they were heap alloced
    new_ptr_level = expr.type.ptr_level + 1
    expr.type = dataclasses.replace(expr.type, ptr_level=new_ptr_level)


@check.register
def check_typeconv(expr: TypeConversion, mod: Module, scopes: Scopes, **kw):
    valid_ast_type(expr.type, mod)

    if expr.init_expr:
        check(expr.init_expr, mod, scopes, **kw)
        expr.throws = set(expr.init_expr.throws)

        # check that the conversion is possible
        from_type = expr.type
        to_type = expr.init_expr.type

        # no pointer casts
        if from_type.ptr_level > 0 or to_type.ptr_level > 0:
            raise FrontendError("cannot cast pointer types",
                                expr.type.name.line, expr.type.name.col)

        # no record type casts
        if (from_type.name in mod.record_types
                or to_type.name in mod.record_types):
            raise FrontendError("cannot cast struct or union types",
                                expr.type.name.line, expr.type.name.col)

        # only enum -> integer type
        if to_type.name in mod.enum_types:
            raise FrontendError("cannot cast to enum type",
                                expr.type.name.line, expr.type.name.col)

        # allowed
    else:
        # just a type declaration
        expr.throws = set()

