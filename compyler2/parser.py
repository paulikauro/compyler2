# parser.py
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


import logging

from lexer import TokenType, FrontendError, peek
from tree import Program, Enum, Struct, Type, VarDecl, Function, Block,\
    Try, If, While, Deferred, LoopCtrl, FuncCtrl, Delete, VarDeclStmt,\
    Constant, Negate, BinOp


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


def parse_program(tokens):
    """program ::=
        (NEWLINE | enum_decl | union_decl | struct_decl | func_decl)* EOF
    """

    dispatch = {
        TokenType.ENUM: parse_enum_decl,
        TokenType.UNION: parse_union_decl,
        TokenType.STRUCT: parse_struct_decl,
    }

    user_types = []
    functions = []

    while True:
        # TODO: fix alignment issues with expect
        token = expect(tokens, TokenType.NEWLINE, TokenType.ENUM,
                       TokenType.UNION, TokenType.STRUCT, TokenType.TYPENAME,
                       TokenType.EOF, do_peek=True)

        if token.type is TokenType.EOF:
            break

        elif token.type is TokenType.NEWLINE:
            logging.debug("parse_program: newline")

        elif token.type is TokenType.TYPENAME:
            functions.append(parse_func_decl(tokens))

        else:
            user_types.append(dispatch[token.type](tokens))

    return Program(user_types, functions)


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
    return Struct(name, members, union=True)


def parse_struct_decl(tokens):
    """struct_decl ::= STRUCT TYPENAME struct_members"""
    expect(tokens, TokenType.STRUCT)
    name = expect(tokens, TokenType.TYPENAME)
    members = parse_struct_members(tokens)
    return Struct(name, members, union=False)


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
    return Struct(name, members, union=union, inline=True, anon=anon)


def parse_type(tokens):
    """type ::= TYPENAME STAR*"""
    name = expect(tokens, TokenType.TYPENAME)
    ptr_level = 0
    while expect(tokens, TokenType.STAR, do_raise=False):
        ptr_level += 1
    return Type(name, ptr_level)


def parse_var_decl(tokens):
    """var_decl ::= type IDENTIFIER"""
    var_type = parse_type(tokens)
    name = expect(tokens, TokenType.IDENTIFIER)
    return VarDecl(name, var_type)


def parse_func_decl(tokens):
    """func_decl ::= var_decl '(' var_decl* ')' statement"""
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

    body = parse_statement(tokens)
    return Function(vardecl.name, vardecl.var_type, parameters, body)


def parse_statement(tokens):
    """statement ::= block_stmt | try_stmt | if_stmt | while_stmt
        | defer_stmt | errdefer_stmt | break_stmt | continue_stmt
        | return_stmt | throw_stmt | delete_stmt | var_decl_stmt | expression"""

    dispatch = {
        TokenType.NEWLINE: parse_block_stmt,
        TokenType.TRY: parse_try_stmt,
        TokenType.IF: parse_if_stmt,
        TokenType.WHILE: parse_while_stmt,
        TokenType.DEFER: parse_deferred_stmt,
        TokenType.ERRDEFER: parse_deferred_stmt,
        TokenType.BREAK: parse_loopctrl_stmt,
        TokenType.CONTINUE: parse_loopctrl_stmt,
        TokenType.RETURN: parse_funcctrl_stmt,
        TokenType.THROW: parse_funcctrl_stmt,
        TokenType.DELETE: parse_delete_stmt,
        TokenType.TYPENAME: parse_var_decl_stmt,
    }

    # TODO: refactor
    token = expect(tokens, TokenType.NEWLINE, TokenType.TRY, TokenType.IF,
                   TokenType.WHILE, TokenType.DEFER, TokenType.ERRDEFER,
                   TokenType.BREAK, TokenType.CONTINUE, TokenType.RETURN,
                   TokenType.THROW, TokenType.DELETE, TokenType.TYPENAME,
                   do_peek=True, do_raise=False)
    if not token:
        # must be an expression
        return parse_expression(tokens)
    return dispatch[token.type](tokens)


def parse_block_stmt(tokens):
    """block_stmt ::= begin statement* end"""
    parse_begin(tokens)
    statements = []
    while not expect(tokens, TokenType.DEDENT, do_peek=True, do_raise=False):
        statements.append(parse_statement(tokens))
        expect(tokens, TokenType.NEWLINE)
    parse_end(tokens)
    return Block(statements)


def parse_try_stmt(tokens):
    """try_stmt ::= TRY statement CATCH var_decl statement"""
    expect(tokens, TokenType.TRY)
    body = parse_statement(tokens)
    expect(tokens, TokenType.CATCH)
    catch_vardecl = parse_var_decl(tokens)
    catch_stmt = parse_statement(tokens)
    return Try(body, catch_vardecl, catch_stmt)


def parse_if_stmt(tokens):
    """if_stmt ::= IF expression statement (ELSE statement)?"""
    expect(tokens, TokenType.IF)
    condition = parse_expression(tokens)
    body = parse_statement(tokens)
    else_body = None
    if expect(tokens, TokenType.ELSE, do_raise=False):
        else_body = parse_statement(tokens)
    return If(condition, body, else_body)


def parse_while_stmt(tokens):
    """
    while_stmt ::= WHILE (COLON IDENTIFIER)? expression statement"""
    expect(tokens, TokenType.WHILE)
    label = None
    if expect(tokens, TokenType.COLON, do_raise=False):
        label = expect(tokens, TokenType.IDENTIFIER)
    condition = parse_expression(tokens)
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
    """return_stmt ::= RETURN expression
    throw_stmt ::= THROW expression"""
    token = expect(tokens, TokenType.RETURN, TokenType.THROW)
    is_err = token.type is TokenType.THROW
    expr = parse_expression(tokens)
    return FuncCtrl(expr, is_err)


def parse_delete_stmt(tokens):
    """delete_stmt ::= DELETE expression"""
    expect(tokens, TokenType.DELETE)
    expr = parse_expression(tokens)
    return Delete(expr)


def parse_var_decl_stmt(tokens):
    """var_decl_stmt ::= var_decl (ASSIGN expression)?"""
    var_decl = parse_var_decl(tokens)
    expr = None
    if expect(tokens, TokenType.ASSIGN, do_raise=False):
        expr = parse_expression(tokens)
    return VarDeclStmt(var_decl, expr)


def parse_expression(tokens):
    pass


# epic hack
def binop(precedence, left_assoc=False):
    def wrapper(*args, **kwargs):
        return BinOp(*args, precedence=precedence, **kwargs)

    if left_assoc:
        # note: don't do this inside the wrapper
        precedence -= 1
    wrapper.precedence = precedence
    return wrapper


op_table = {
    # logical
    TokenType.AND: binop(2),
    TokenType.OR: binop(2),
    TokenType.NOT: binop(5),

    # relational
    TokenType.EQ: binop(10),
    TokenType.NE: binop(10),
    TokenType.LE: binop(10),
    TokenType.GE: binop(10),
    TokenType.LT: binop(10),
    TokenType.GT: binop(10),

    # bitwise & shifts
    TokenType.BOR: binop(20),
    TokenType.BAND: binop(20),
    TokenType.BXOR: binop(20),
    TokenType.BNOT: binop(20),
    TokenType.SHL: binop(20),
    TokenType.SHR: binop(20),

    # arithmetic
    TokenType.PLUS: binop(30),
    TokenType.MINUS: binop(30),

    TokenType.STAR: binop(40),
    TokenType.SLASH: binop(40),
    # "%": "PERCENT",

    # other (all are not really operators)
    # TokenType.DOT: binop(10),
    # TokenType.LSQB: binop(10),
    # TokenType.RSQB: binop(10),
    # TokenType.LPAREN: binop(10),
    # TokenType.RPAREN: binop(10),

    # assignment & augmented assignment
    TokenType.ASSIGN: binop(100),
    TokenType.PLUS_ASSIGN: binop(100),
    TokenType.MINUS_ASSIGN: binop(100),
    TokenType.STAR_ASSIGN: binop(100),
    TokenType.SLASH_ASSIGN: binop(100),
    # "%=": "PERCENT_ASSIGN",
    TokenType.BOR_ASSIGN: binop(100),
    TokenType.BAND_ASSIGN: binop(100),
    TokenType.BXOR_ASSIGN: binop(100),
    TokenType.BNOT_ASSIGN: binop(100),
    TokenType.SHL_ASSIGN: binop(100),
    TokenType.SHR_ASSIGN: binop(100),
}


def parse_expr(tokens):
    stack = []
    left = parse_primary_expr(tokens)
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
        left = parse_primary_expr(tokens)

    # pop stack
    while stack:
        tos = stack.pop()
        tos.right = left
        left = tos
    return left


def parse_primary_expr(tokens):
    """primary_expr ::= INT_LITERAL | CHAR_LITERAL | STR_LITERAL | IDENTIFIER
        | LPAREN expression RPAREN"""
    token = expect(tokens, TokenType.INT_LITERAL, TokenType.CHAR_LITERAL,
                   TokenType.STR_LITERAL, TokenType.IDENTIFIER,
                   TokenType.LPAREN)

    if token.type is TokenType.LPAREN:
        expr = parse_expr(tokens)
        expect(tokens, TokenType.RPAREN)
        return expr
    elif token.type is TokenType.INT_LITERAL:
        return Constant(token)
    raise NotImplemented

