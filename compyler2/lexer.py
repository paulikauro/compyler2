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


from collections import namedtuple
from enum import Enum, auto
import re


class LexerError(Exception):
    """Tokenization failed"""

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


Token = namedtuple("Token", "type value line col")


class TokenType(Enum):
    Indent = auto()
    Dedent = auto()
    Newline = auto()
    Keyword = auto()
    Operator = auto()
    Typename = auto()
    Identifier = auto()
    StringLiteral = auto()
    CharLiteral = auto()
    IntLiteral = auto()
    EOF = auto()


operators = "|".join(map(re.escape, (
    # arithmetic
    "+", "-", "*", "/", "%",

    # comparison
    "==", "!=", "<=", ">=", "<", ">",

    # assignment & augmented assignment
    "=", "+=", "-=", "*=", "/=", "%=",

    # other (all are not really operators)
    ".", ":", "[", "]", "(", ")",
)))


keywords = "|".join((
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
))


lexer_regex = fr"""
    # comments
      (?P<comment>            \#[^\n]* (\r?\n)*)

    # newlines + indentation
    | (?P<newline_indent>     (?P<newline> (\r?\n)+) (?P<indent> (\ |\t)*))

    # other whitespace
    | (?P<whitespace>         \s+)

    # keywords
    | (?P<Keyword>            {keywords})

    # operators etc
    | (?P<Operator>           {operators})

    # type names
    | (?P<Typename>           [A-Z][a-zA-Z0-9_]*)

    # identifiers
    | (?P<Identifier>         [a-z_][a-zA-Z0-9_]*)

    # string literals
    # TODO: escapes
    | (?P<string>             \"[^\"]*\")

    # character literals
    | (?P<char>               \'[^\']\')

    # base 16 integer literals
    | (?P<b16int>             0x[0-9a-fA-F]+)

    # base 10 integer literals
    | (?P<b10int>             [1-9][0-9]*)
    # TODO: other integer literals
"""


pattern = re.compile(lexer_regex, re.VERBOSE | re.MULTILINE)


def token_gen(source):
    line = 1
    col = 1
    # stack for indent levels
    indents = [0]
    indentchar = None

    for m in pattern.finditer(source):
        # usually only one match, lastgroup is the outermost one
        t = m.lastgroup
        val = m.groupdict()[t]

        t_line, t_col = line, col

        match_len = m.end() - m.start()
        col += match_len

        # skip whitespace and comments
        if t in {"whitespace", "comment"}:
            continue

        # deal with indentation and newlines
        elif t == "newline_indent":
            yield Token(TokenType.Newline, val, line, col)

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
                    raise LexerError("mixing tabs and spaces for indentation",
                                     line, col)
            else:
                indent = ""

            # indentation
            current_level = len(indent)
            diff = current_level - indents[-1]
            if diff > 0:
                # push level to stack
                indents.append(current_level)
                yield Token(TokenType.Indent, val, t_line, t_col)
            elif diff < 0:
                # yield dedent tokens until indent levels match
                while current_level != indents[-1]:
                    yield Token(TokenType.Dedent, val, t_line, t_col)
                    indents.pop()
                    if not indents:
                        raise LexerError("inconsistent indentation", line, col)
            # else no change in indentation level
            continue

        # process other tokens
        elif t[0].isupper():
            # ugly hack
            tt = TokenType[t]
        elif t == "string":
            tt = TokenType.StringLiteral
            val = val[1:-1]
        elif t == "char":
            tt = TokenType.CharLiteral
            val = val[1:-1]
        elif t == "b16int":
            tt = TokenType.IntLiteral
            val = int(val, 16)
        elif t == "b10int":
            tt = TokenType.IntLiteral
            val = int(val, 10)
        else:
            print(f"got a {t}: {val}, but not supported yet")
            continue

        yield Token(tt, val, t_line, t_col)

    # TODO: check for end
    yield Token(TokenType.EOF, None, line, col)


def tokenize(source):
    """Generates a stream of tokens from `source`."""
    gen = peekable_gen(token_gen(source))
    next(gen)
    return gen

