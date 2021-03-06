# main.py
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


import sys
import logging

from lexer import tokenize, FrontendError
from parser import parse_module
from tree import format_tree
from typecheck import perform_check
from targets import x86_64
from irgen import do_irgen
from ir import DotGraphPass


def main(argv):
    # TODO: argparse
    logging.basicConfig(level=logging.DEBUG)
    if len(argv) != 2:
        print("usage:", argv[0], "sourcefile")
        return 2

    file = argv[1]
    try:
        # TODO: mmap?
        with open(file, "r") as f:
            source = f.read()
    except IOError as e:
        print("failed to open source file:", e)
        return 1

    try:
        # TODO: make this optional and more efficient
        # tokens = tokenize(source)
        # logging.debug("Lexer output:")
        # for token in tokens:
        #     logging.debug(token)

        tokens = tokenize(source)
        tree = parse_module(tokens)
        logging.debug("\n" + format_tree(tree))
        perform_check(tree, target=x86_64)
        ir = do_irgen(tree)
        dot_pass = DotGraphPass()
        dot_pass.run(ir)
        dot_pass.write_png(f"{file}.png")

    except FrontendError as front_err:
        print(f"{file}: {front_err}")
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv))

