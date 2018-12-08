# main.py
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


import sys

from lexer import tokenize


def main(argv):
    if len(argv) != 2:
        print("usage:", argv[0], "sourcefile")
        return 2

    try:
        with open(argv[1], "r") as f:
            source = f.read()
    except IOError as e:
        print("failed to open source file:", e)
        return 1

    tokens = tokenize(source)
    for token in tokens:
        print(token)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

