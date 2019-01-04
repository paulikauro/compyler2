# targets.py
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


"""This module contains various information about different targets."""


from dataclasses import dataclass

import ir


@dataclass(frozen=True)
class Target:
    Ptr: ir.PointerType

    def get_type(self, name):
        types = {
            "Ptr": self.Ptr,
            "Int": ir.I64,
        }
        # TODO: try?
        return types[name]


x86_64_ptr = ir.PointerType(target_name="x86_64", size=8)
x86_64 = Target(Ptr=x86_64_ptr)

