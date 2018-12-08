# lexer.py
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
from dataclasses import dataclass
from typing import Any
from enum import Enum
import re


class FrontendError(Exception):
    """An error in the fronted phases of the compiler"""

    def __init__(self, msg, line, col):
        super().__init__(msg)
        self.line = line
        self.col = col

    # TODO: repr
    def __str__(self):
        return f"line {self.line}, column {self.col}: {super().__str__()}"


def peekable_gen(it):
    """A generator wrapper around iterable `it` that supports peeking.
    Must be initialized with next() or by sending None."""

    peeking = yield
    for x in it:
        while peeking:
            peeking = yield x
        peeking = yield x


def peek(gen):
    """Returns the next item in peekable generator `gen` without consuming the
    item in the underlying iterator."""
    return gen.send(True)


operators = {
    # comparison
    "==": "EQ",
    "!=": "NE",
    "<=": "LE",
    ">=": "GE",
    "<": "LT",
    ">": "GT",

    # assignment & augmented assignment
    "=": "ASSIGN",
    "+=": "PLUS_ASSIGN",
    "-=": "MINUS_ASSIGN",
    "*=": "STAR_ASSIGN",
    "/=": "SLASH_ASSIGN",
    "%=": "PERCENT_ASSIGN",

    # arithmetic (keep these after augmented assignments for regex)
    "+": "PLUS",
    "-": "MINUS",
    "*": "STAR",
    "/": "SLASH",
    "%": "PERCENT",

    # other (all are not really operators)
    ".": "DOT",
    ":": "COLON",
    "[": "LSQB",
    "]": "RSQB",
    "(": "LPAREN",
    ")": "RPAREN",
}
operator_string = "|".join(map(re.escape, operators))


keywords = (
    # data structures
    "enum",
    "struct",
    "union",

    # data structure manipulation
    "with",
    "new",
    "delete",

    # control flow structures
    "if",
    "else",
    "while",
    "break",
    "continue",
    "return",

    # error handling and resource management
    "try",
    "throw",
    "catch",
    "defer",
    "errdefer",
)
keyword_string = "|".join(keywords)


lexer_regex = fr"""
    # comments
      (?P<comment>            \#[^\n]* (\r?\n)*)

    # newlines + indentation
    | (?P<newline_indent>     (?P<newline> (\r?\n)+) (?P<indent> (\ |\t)*))

    # other whitespace
    | (?P<whitespace>         \s+)

    # keywords
    | (?P<keyword>            ({keyword_string}) (?=\W))

    # operators etc
    | (?P<operator>           {operator_string})

    # type names
    | (?P<typename>           [A-Z]\w* (?=\W))

    # identifiers
    | (?P<identifier>         [a-z_]\w* (?=\W))

    # string literals
    # TODO: escapes
    | (?P<string>             \"[^\"]*\" (?=\W))

    # character literals
    | (?P<char>               \'[^\']\' (?=\W))

    # base 16 integer literals
    | (?P<b16int>             0x[0-9a-fA-F]+)

    # base 10 integer literals
    | (?P<b10int>             [1-9][0-9]*)
    # TODO: other integer literals
"""


pattern = re.compile(lexer_regex, re.VERBOSE | re.MULTILINE)


others = ("INDENT", "DEDENT", "NEWLINE", "TYPENAME", "IDENTIFIER",
          "STRING_LITERAL", "CHAR_LITERAL", "INT_LITERAL", "EOF")


TokenType = Enum("TokenType",
                 [*operators.values(), *map(str.upper, keywords), *others])


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

    def __eq__(self, other):
        if not isinstance(other, Token):
            raise NotImplemented
        return self.type is other.type and self.value == other.value

    __hash__ = None


def token_gen(source):
    line = 1
    col = 1
    # stack for indent levels
    indents = [0]
    indentchar = None

    for m in pattern.finditer(source):
        # usually only one match, lastgroup is the outermost one
        group = m.lastgroup
        value = m.groupdict()[group]

        t_line, t_col = line, col

        match_len = m.end() - m.start()
        col += match_len

        # skip whitespace and comments
        if group in {"whitespace", "comment"}:
            continue

        # deal with indentation and newlines
        elif group == "newline_indent":
            yield Token(TokenType.NEWLINE, value, line, col)

            # newline cannot be empty (regex matches at least once)
            newlines = m.groupdict()["newline"].count("\n")

            # keep track of position
            line += newlines
            col = 1

            indent = m.groupdict()["indent"]

            if indent:
                if not indentchar:
                    indentchar = indent[0]
                elif indentchar != indent[0]:
                    raise FrontendError("use tabs or spaces",
                                        line, col)
            else:
                indent = ""

            # indentation
            current_level = len(indent)
            diff = current_level - indents[-1]
            if diff > 0:
                # push level to stack
                indents.append(current_level)
                yield Token(TokenType.INDENT, value, t_line, t_col)
            elif diff < 0:
                # yield dedent tokens until indent levels match
                while current_level != indents[-1]:
                    yield Token(TokenType.DEDENT, value, t_line, t_col)
                    indents.pop()
                    if not indents:
                        raise FrontendError("inconsistent indentation",
                                            line, col)
            # else no change in indentation level
            continue

        # process other tokens
        elif group == "keyword":
            token = TokenType[value.upper()]
        elif group == "operator":
            token = TokenType[operators[value]]
        elif group == "typename":
            token = TokenType.TYPENAME
        elif group == "identifier":
            token = TokenType.IDENTIFIER
        elif group == "string":
            token = TokenType.STRING_LITERAL
            value = value[1:-1]
        elif group == "char":
            token = TokenType.CHAR_LITERAL
            value = value[1:-1]
        elif group == "b16int":
            token = TokenType.INT_LITERAL
            value = int(value, 16)
        elif group == "b10int":
            token = TokenType.INT_LITERAL
            value = int(value, 10)
        else:
            logging.warning(f"got a {group}: {value}, but not supported yet")
            raise LexerError(f"bug: {group}")

        yield Token(token, value, t_line, t_col)

    # TODO: check for end
    yield Token(TokenType.EOF, None, line, col)


def tokenize(source):
    """Generates a stream of tokens from `source`."""
    gen = peekable_gen(token_gen(source))
    next(gen)
    return gen

