# parser.py
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

from lexer import TokenType, FrontendError, peek
from tree import Module, Enum, Record, Type, VarDecl, Function, Block,\
    Try, Catch, If, While, Deferred, LoopCtrl, FuncCtrl, Delete, \
    NumLiteral, StrLiteral, Call, BinOp, UnOp, VarAccess, StructAccess,\
    ArrayAccess, Assignment, TypeExpr, TypeConversion, ArrayAlloc


# helper functions
def expect(tokens, *expected, do_peek=False, do_raise=True):
    """Is the next token in `tokens` is of any of the types in `expected`?
    Consumes token if successful, unless do_peek specified."""

    token = peek(tokens)
    if token.type in expected:
        if not do_peek:
            next(tokens)
        return token
    elif do_raise:
        raise FrontendError(f"expected one of {expected}, "
                            f"but got {token}", token.line, token.col)
    else:
        return None


def parse_module(tokens):
    """module ::=
        (NEWLINE | enum_decl | union_decl | struct_decl | func_decl)* EOF
    """

    record_dispatch = {
        TokenType.UNION: parse_union_decl,
        TokenType.STRUCT: parse_struct_decl,
    }

    enum_types = {}
    record_types = {}
    functions = {}

    while True:
        # TODO: fix alignment issues with expect
        token = expect(tokens, TokenType.NEWLINE, TokenType.ENUM,
                       TokenType.UNION, TokenType.STRUCT, TokenType.TYPENAME,
                       TokenType.EOF, do_peek=True)

        if token.type is TokenType.EOF:
            break

        elif token.type is TokenType.NEWLINE:
            next(tokens)
            logging.debug("parse_module: newline")

        elif token.type is TokenType.TYPENAME:
            func = parse_func_decl(tokens)
            if func.name in functions:
                raise FrontendError(
                    f"Function {func.name.value} defined twice",
                    func.name.line, func.name.col)
            functions[func.name.value] = func

        elif token.type is TokenType.ENUM:
            enum = parse_enum_decl(tokens)
            if enum.name in enum_types:
                raise FrontendError(f"Enum {enum.name.value} defined twice",
                                    enum.name.line, enum.name.col)
            enum_types[enum.name.value] = enum

        else:
            record = record_dispatch[token.type](tokens)
            if record.name in record_types:
                record_type = "union" if record.union else "struct"
                raise FrontendError(
                    f"{record_type} {record.name.value!r} defined twice",
                    record.name.line, record.name.col)
            record_types[record.name.value] = record

    return Module(enum_types, record_types, functions)


def parse_begin(tokens):
    """begin ::= NEWLINE INDENT"""
    expect(tokens, TokenType.NEWLINE)
    return expect(tokens, TokenType.INDENT)


def parse_end(tokens):
    """end ::= DEDENT"""
    return expect(tokens, TokenType.DEDENT)


def parse_enum_decl(tokens):
    """enum_decl ::= ENUM TYPENAME enum_members"""
    expect(tokens, TokenType.ENUM)
    name = expect(tokens, TokenType.TYPENAME)
    members = parse_enum_members(tokens)
    return Enum(name, members)


def parse_enum_members(tokens):
    """enum_members ::= begin (TYPENAME NEWLINE)+ end"""
    parse_begin(tokens)
    members = []

    while True:
        member = expect(tokens, TokenType.TYPENAME, do_raise=False)
        if not member:
            break
        members.append(member)
        expect(tokens, TokenType.NEWLINE)

    end = parse_end(tokens)

    if not members:
        # TODO: unreachable?
        raise FrontendError("an enum must have at least one member",
                            end.row, end.col)

    return members


def parse_union_decl(tokens):
    """union_decl ::= UNION TYPENAME struct_members"""
    expect(tokens, TokenType.UNION)
    name = expect(tokens, TokenType.TYPENAME)
    members = parse_struct_members(tokens)
    return Record(name, members, union=True)


def parse_struct_decl(tokens):
    """struct_decl ::= STRUCT TYPENAME struct_members"""
    expect(tokens, TokenType.STRUCT)
    name = expect(tokens, TokenType.TYPENAME)
    members = parse_struct_members(tokens)
    return Record(name, members, union=False)


def parse_struct_members(tokens):
    """struct_members ::=
        begin (var_decl NEWLINE | inline_enum | inline_union | inline_struct)+
        end"""

    dispatch = {
        TokenType.ENUM: parse_inline_enum,
        TokenType.UNION: parse_inline_struct_or_union,
        TokenType.STRUCT: parse_inline_struct_or_union,
    }

    parse_begin(tokens)
    members = []

    # first run of loop: raise exception if expect fails
    first = True
    while True:
        lookahead = expect(tokens, TokenType.TYPENAME, TokenType.ENUM,
                           TokenType.UNION, TokenType.STRUCT, do_raise=first,
                           do_peek=True)
        first = False
        if not lookahead:
            break

        elif lookahead.type is TokenType.TYPENAME:
            members.append(parse_var_decl(tokens))
            expect(tokens, TokenType.NEWLINE)
        else:
            # TODO: refactor union/struct
            members.append(dispatch[lookahead.type](tokens))

    parse_end(tokens)
    return members


def parse_inline_enum(tokens):
    """inline_enum ::= ENUM IDENTIFIER enum_members"""
    expect(tokens, TokenType.ENUM)
    name = expect(tokens, TokenType.IDENTIFIER)
    members = parse_enum_members(tokens)
    # TODO?
    return Enum(name, members, inline=True)


def parse_inline_struct_or_union(tokens):
    """inline_union ::= UNION IDENTIFIER? struct_members
    inline_struct ::= STRUCT IDENTIFIER? struct_members"""

    tok = expect(tokens, TokenType.STRUCT, TokenType.UNION)
    union = tok.type is TokenType.UNION
    name = expect(tokens, TokenType.IDENTIFIER, do_raise=False)
    anon = name is None
    members = parse_struct_members(tokens)
    return Record(name, members, union=union, inline=True, anon=anon)


def parse_type(tokens, name=None, allow_multiple=False):
    """type ::= TYPENAME (DOT TYPENAME)? STAR*"""
    if not name:
        name = expect(tokens, TokenType.TYPENAME)

    member = None

    if allow_multiple and expect(tokens, TokenType.DOT, do_raise=False):
        member = expect(tokens, TokenType.TYPENAME, TokenType.IDENTIFIER)

    ptr_level = 0
    while expect(tokens, TokenType.STAR, do_raise=False):
        ptr_level += 1
    return Type(name, ptr_level, member=member)


def parse_var_decl(tokens):
    """var_decl ::= type IDENTIFIER"""
    var_type = parse_type(tokens)
    name = expect(tokens, TokenType.IDENTIFIER)
    return VarDecl(name, var_type)


def parse_func_decl(tokens):
    """func_decl ::= var_decl '(' var_decl* ')' (THROWS type*)? block_stmt"""
    # TODO: commas
    vardecl = parse_var_decl(tokens)
    expect(tokens, TokenType.LPAREN)

    parameters = []
    while True:
        token = expect(tokens, TokenType.TYPENAME, TokenType.RPAREN,
                       do_peek=True)
        if token.type is TokenType.RPAREN:
            # consume paren
            next(tokens)
            break
        parameters.append(parse_var_decl(tokens))

    # parse throws declaration
    throws = set()
    if expect(tokens, TokenType.THROWS, do_raise=False):
        while not expect(tokens, TokenType.NEWLINE,
                         do_peek=True, do_raise=False):
            throws.add(parse_type(tokens))

    body = parse_block_stmt(tokens)
    return Function(vardecl.name, vardecl.var_type, parameters, throws, body)


def parse_statement(tokens):
    """statement ::= block_stmt | try_stmt | if_stmt | while_stmt
        | defer_stmt | errdefer_stmt | break_stmt | continue_stmt
        | return_stmt | throw_stmt | delete_stmt | expression"""

    # newline hackery
    normal_dispatch = {
        TokenType.NEWLINE: parse_block_stmt,
        TokenType.TRY: parse_try_stmt,
        TokenType.IF: parse_if_stmt,
        TokenType.WHILE: parse_while_stmt,
        TokenType.DEFER: parse_deferred_stmt,
        TokenType.ERRDEFER: parse_deferred_stmt,
    }
    newline_dispatch = {
        TokenType.BREAK: parse_loopctrl_stmt,
        TokenType.CONTINUE: parse_loopctrl_stmt,
        TokenType.RETURN: parse_funcctrl_stmt,
        TokenType.THROW: parse_funcctrl_stmt,
        TokenType.DELETE: parse_delete_stmt,
    }

    # TODO: refactor
    token = expect(tokens, TokenType.NEWLINE, TokenType.TRY, TokenType.IF,
                   TokenType.WHILE, TokenType.DEFER, TokenType.ERRDEFER,
                   TokenType.BREAK, TokenType.CONTINUE, TokenType.RETURN,
                   TokenType.THROW, TokenType.DELETE,
                   do_peek=True, do_raise=False)
    if not token:
        # must be an expression; newline required
        stmt = parse_expression(tokens)
    elif token.type in newline_dispatch:
        # trailing newline required; "simple" statements
        stmt = newline_dispatch[token.type](tokens)
    else:
        # no trailing newline required by these statements
        return normal_dispatch[token.type](tokens)

    # blindly assuming stuff
    if not expect(tokens, TokenType.NEWLINE, do_raise=False):
        if not expect(tokens, TokenType.DEDENT, do_peek=True, do_raise=False):
            logging.debug("did not get NEWLINE or DEDENT")
    return stmt


def parse_block_stmt(tokens):
    """block_stmt ::= begin statement* end"""
    parse_begin(tokens)
    statements = []
    while not expect(tokens, TokenType.DEDENT, do_peek=True, do_raise=False):
        stmt = parse_statement(tokens)
        statements.append(stmt)
        # expect(tokens, TokenType.NEWLINE)

    parse_end(tokens)
    return Block(statements)


def parse_try_stmt(tokens):
    """try_stmt ::= TRY statement (CATCH var_decl statement)+"""
    try_tok = expect(tokens, TokenType.TRY)
    body = parse_statement(tokens)
    catches = []
    while expect(tokens, TokenType.CATCH, do_raise=False):
        catch_vardecl = parse_var_decl(tokens)
        catch_stmt = parse_statement(tokens)
        catches.append(Catch(catch_vardecl, catch_stmt))

    if not catches:
        raise FrontendError("try statement must have at least one catch",
                            try_tok.line, try_tok.col)
    return Try(body, catches)


def parse_if_stmt(tokens):
    """if_stmt ::= IF NOT? expression statement (ELSE statement)?"""
    expect(tokens, TokenType.IF)
    inverted = expect(tokens, TokenType.NOT, do_raise=False)
    condition = parse_expression(tokens)
    # TODO: fix grammar
    if inverted:
        condition = UnOp(inverted, condition)
    body = parse_statement(tokens)
    else_body = None
    if expect(tokens, TokenType.ELSE, do_raise=False):
        else_body = parse_statement(tokens)
    return If(condition, body, else_body)


def parse_while_stmt(tokens):
    """
    while_stmt ::= WHILE (COLON IDENTIFIER)? NOT? expression statement"""
    expect(tokens, TokenType.WHILE)
    label = None
    if expect(tokens, TokenType.COLON, do_raise=False):
        label = expect(tokens, TokenType.IDENTIFIER)
    inverted = expect(tokens, TokenType.NOT, do_raise=False)
    condition = parse_expression(tokens)
    # TODO: fix grammar
    if inverted:
        condition = UnOp(inverted, condition)
    body = parse_statement(tokens)
    return While(condition, body, label)


def parse_deferred_stmt(tokens):
    """defer_stmt ::= DEFER statement
    errdefer_stmt ::= ERRDEFER statement"""
    token = expect(tokens, TokenType.DEFER, TokenType.ERRDEFER)
    on_err = token.type is TokenType.ERRDEFER
    statement = parse_statement(tokens)
    return Deferred(statement, on_err)


def parse_loopctrl_stmt(tokens):
    """break_stmt ::= BREAK IDENTIFIER?
    continue_stmt ::= CONTINUE IDENTIFIER?"""
    token = expect(tokens, TokenType.CONTINUE, TokenType.BREAK)
    out = token.type is TokenType.BREAK
    label = expect(tokens, TokenType.IDENTIFIER, do_raise=False)
    return LoopCtrl(out, label)


def parse_funcctrl_stmt(tokens):
    """return_stmt ::= RETURN expression?
    throw_stmt ::= THROW expression"""
    token = expect(tokens, TokenType.RETURN, TokenType.THROW)
    is_err = token.type is TokenType.THROW
    expr = None
    newline = expect(tokens, TokenType.NEWLINE, do_raise=False, do_peek=True)
    if is_err or not newline:
        expr = parse_expression(tokens)
    return FuncCtrl(expr, is_err)


def parse_delete_stmt(tokens):
    """delete_stmt ::= DELETE expression"""
    expect(tokens, TokenType.DELETE)
    expr = parse_expression(tokens)
    return Delete(expr)


def parse_expression(tokens):
    """expression ::= logical assign_op expression | logical"""
    left = parse_logical(tokens)
    while True:
        assign = expect(tokens, *assign_ops.keys(), do_raise=False)
        if not assign:
            break
        right = parse_logical(tokens)
        augmented_op = assign_ops[assign.type]
        left = Assignment(augmented_op, left, right)
    return left


"""assign_op ::= ASSIGN | BOR_ASSIGN | BAND_ASSIGN | BXOR_ASSIGN
| SHL_ASSIGN | SHR_ASSIGN | PLUS_ASSIGN | MINUS_ASSIGN
| STAR_ASSIGN | SLASH_ASSIGN"""

assign_ops = {
    # normal assignment does not have any operation associated with it
    TokenType.ASSIGN: None,

    # augmented assignment operators do
    TokenType.PLUS_ASSIGN: TokenType.PLUS,
    TokenType.MINUS_ASSIGN: TokenType.MINUS,
    TokenType.STAR_ASSIGN: TokenType.STAR,
    TokenType.SLASH_ASSIGN: TokenType.SLASH,
    # "%=": "PERCENT_ASSIGN",
    TokenType.BOR_ASSIGN: TokenType.BOR,
    TokenType.BAND_ASSIGN: TokenType.BAND,
    TokenType.BXOR_ASSIGN: TokenType.BXOR,
    TokenType.SHL_ASSIGN: TokenType.SHL,
    TokenType.SHR_ASSIGN: TokenType.SHR,
}


# epic hack
def binop(precedence):
    def wrapper(*args, **kwargs):
        return BinOp(*args, precedence=precedence, **kwargs)

    wrapper.precedence = precedence
    return wrapper


op_table = {
    # logical ::= logical (AND | OR | XOR) NOT? relational | NOT? relational
    TokenType.AND: binop(2),
    TokenType.OR: binop(2),
    TokenType.XOR: binop(2),
    # TokenType.NOT: binop(5),

    # relational ::= relational (EQ | NE | LT | GT | LE | GE) bitwise | bitwise
    TokenType.EQ: binop(10),
    TokenType.NE: binop(10),
    TokenType.LE: binop(10),
    TokenType.GE: binop(10),
    TokenType.LT: binop(10),
    TokenType.GT: binop(10),

    # bitwise ::= bitwise (PLUS | MINUS | BAND | BOR | BXOR) shift | shift
    TokenType.PLUS: binop(20),
    TokenType.MINUS: binop(20),
    TokenType.BOR: binop(20),
    TokenType.BAND: binop(20),
    TokenType.BXOR: binop(20),
    TokenType.BNOT: binop(20),

    # shift ::= shift (SHR | SHL | STAR | SLASH) unary | unary
    TokenType.SHL: binop(40),
    TokenType.SHR: binop(40),
    TokenType.STAR: binop(40),
    TokenType.SLASH: binop(40),
    # "%": "PERCENT",
}


def parse_logical(tokens):
    # operator precedence(?)/shunting yard algorithm parser
    # handles only left-associative operators
    stack = []
    left = parse_unary(tokens)
    while True:
        token = expect(tokens, *op_table.keys(), do_raise=False)
        if not token:
            break
        op = token.type
        node_type = op_table[op]
        node = node_type(op)

        while stack and stack[-1].precedence > node_type.precedence:
            # top of stack op binds stronger than current op
            tos = stack.pop()
            tos.right = left
            left = tos

        node.left = left
        stack.append(node)
        left = parse_unary(tokens)

    # pop stack
    while stack:
        tos = stack.pop()
        tos.right = left
        left = tos
    return left


def parse_unary(tokens):
    """unary ::= (MINUS | BNOT | STAR | BAND) unary | primary_expr"""

    stack = []
    while True:
        op = expect(tokens, TokenType.MINUS, TokenType.BNOT, TokenType.STAR,
                    TokenType.BAND, do_raise=False)
        if not op:
            break
        stack.append(op)

    node = parse_primary_expr(tokens)
    while stack:
        node = UnOp(stack.pop(), node)
    return node


def parse_primary_expr(tokens, is_arg=False):
    """primary_expr ::= INT_LITERAL | CHAR_LITERAL | STR_LITERAL
    | LPAREN expression RPAREN | access_or_call | type_expr"""

    token = expect(tokens, TokenType.INT_LITERAL, TokenType.CHAR_LITERAL,
                   TokenType.STR_LITERAL, TokenType.LPAREN,
                   TokenType.IDENTIFIER, TokenType.NEW, TokenType.TYPENAME,
                   do_raise=not is_arg)

    if is_arg and token is None:
        return None

    if token.type is TokenType.INT_LITERAL:
        return NumLiteral(token)

    elif token.type is TokenType.CHAR_LITERAL:
        return NumLiteral(token, is_char=True)

    elif token.type is TokenType.STR_LITERAL:
        return StrLiteral(token)

    elif token.type is TokenType.LPAREN:
        expr = parse_expression(tokens)
        expect(tokens, TokenType.RPAREN)
        return expr

    elif token.type is TokenType.IDENTIFIER:
        node = VarAccess(token)
        # parse_access will lookahead
        node = parse_access(tokens, node)
        if is_arg:
            return node

        # else, parse function arguments if any
        args = []
        while True:
            # NOTE: this will not make runtime exponential
            # there is no backtracking here, only lookahead
            arg = parse_primary_expr(tokens, is_arg=True)
            if not arg:
                break
            args.append(arg)

        if not args:
            return node
        return Call(node, args)
    else:
        # NEW or TYPENAME
        # hack
        return parse_type_expr(tokens, token)


def parse_access(tokens, node):
    """access ::= (LSQB expression RSQB | DOT IDENTIFIER)*
    Returns `node` as-is if nothing applicable.
    """

    while True:
        token = expect(tokens, TokenType.DOT, TokenType.LSQB,
                       do_raise=False)
        if not token:
            break

        elif token.type is TokenType.LSQB:
            # array access
            index = parse_expression(tokens)
            expect(tokens, TokenType.RSQB)
            node = ArrayAccess(node, index)
        else:
            # struct access
            name = expect(tokens, TokenType.IDENTIFIER)
            node = StructAccess(node, name)

    return node


def parse_type_expr(tokens, new_token):
    """
    type_expr ::=
        NEW? type (primary_expr | LSQB expression RSQB | WITH with_assignments)?
    """

    # TODO: using
    is_new = new_token.type is TokenType.NEW
    if is_new:
        new_token = None
    type = parse_type(tokens, new_token, allow_multiple=True)

    lookahead = expect(tokens, TokenType.LSQB, TokenType.WITH,
                       TokenType.NEWLINE, do_peek=True, do_raise=False)
    if not lookahead:
        # must be primary_expr
        expr = parse_primary_expr(tokens)
        return TypeConversion(type, is_new, expr)

    elif lookahead.type is TokenType.LSQB:
        # skip [
        next(tokens)
        expr = parse_expression(tokens)
        expect(tokens, TokenType.RSQB)
        return ArrayAlloc(type, is_new, expr)

    elif lookahead.type is TokenType.WITH:
        # skip WITH
        next(tokens)
        assignments = parse_with_assignments(tokens)
        return TypeExpr(type, is_new, assignments)

    else:
        # NEWLINE
        return TypeExpr(type, is_new, None)


def parse_with_assignments(tokens):
    """with_assignments ::= with_assign | begin (with_assign NEWLINE)+ end
    with_assign ::= IDENTIFIER access assign_op expression
    """

    block = expect(tokens, TokenType.NEWLINE, do_peek=True, do_raise=False)
    if block:
        parse_begin(tokens)

    assignments = []
    while True:
        left = expect(tokens, TokenType.IDENTIFIER)
        # struct and array accesses
        left = parse_access(tokens, left)
        assign = expect(tokens, *assign_ops.keys())
        expr = parse_expression(tokens)

        op = assign_ops[assign.type]
        assignments.append(Assignment(op, left, expr))
        if not block:
            break

        expect(tokens, TokenType.NEWLINE, do_raise=False)
        if expect(tokens, TokenType.DEDENT, do_peek=True, do_raise=False):
            parse_end(tokens)
            break

    return assignments


