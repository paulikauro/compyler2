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
from tree import Program, Enum, Struct, Type, VarDecl


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

    # keep in sync
    lookahead = peek(tokens)
    token = lookahead.type

    enums = []
    structs = []

    while token is not TokenType.EOF:
        # TODO: expect + dispatch
        if token is TokenType.NEWLINE:
            logging.debug("parse_program: newline")

        elif token is TokenType.ENUM:
            enums.append(parse_enum_decl(tokens))

        elif token is TokenType.UNION:
            structs.append(parse_union_decl(tokens))

        elif token is TokenType.STRUCT:
            structs.append(parse_struct_decl(tokens))

        elif token is TokenType.TYPENAME:
            pass
        else:
            raise FrontendError(f"unexpected: {token}",
                                lookahead.line, lookahead.col)

        # keep in sync
        lookahead = peek(tokens)
        token = lookahead.type

    # consume EOF
    next(tokens)
    return Program(enums, structs)


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
        members.append(member.value)
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


