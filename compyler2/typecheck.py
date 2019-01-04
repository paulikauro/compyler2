# typecheck.py
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
import dataclasses
from functools import singledispatch
from typing import Union

from lexer import FrontendError
from tree import Module, Record, RecordMember, Enum, VarDecl, Type, Function
from targets import Target
import ir


def perform_check(module: Module, target: Target):
    module.target = target
    check_module(module)


@singledispatch
def check(node, *args, **kwargs):
    logging.warning(f"check({type(node)}, {len(args)}, {len(kwargs)})")


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

    func_names = set()
    # check functions
    for func in module.functions.values():
        check_name(func_names, func.name)
        check(func, module)


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
            new_member = RecordMember(member.name.value, member_type,
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


