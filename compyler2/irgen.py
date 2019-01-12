# irgen.py
#
# Copyright (C) 2019 Pauli Kauro
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


from dataclasses import dataclass, replace
from functools import singledispatch

import ir
from lexer import TokenType
from tree import Module, Function, Block, Assignment, VarAccess, BinOp
from typecheck import Scopes


@dataclass
class ScopeEntry:
    node: ir.Node
    type: ir.IRType


@dataclass
class IRGenState:
    mod: Module
    ir_mod: ir.Module
    ir_func: ir.Function
    scopes: Scopes
    ctrl: ir.Node


def do_irgen(mod: Module):
    state = IRGenState(None, None, None, None, None)
    return gen_module(mod, state)


@singledispatch
def gen(node, state: IRGenState, **kw):
    raise NotImplementedError


@gen.register
def gen_module(mod: Module, state: IRGenState):
    state.mod = mod
    state.ir_mod = ir.Module("ir_module")
    state.scopes = Scopes()

    # create function stubs into the global scope
    for func_name, func in mod.functions.items():
        ir_ret_type = mod.ir_type(func.return_type)
        ir_func = ir.Function(state.ir_mod, func_name, ir_ret_type)
        state.scopes[func_name] = ScopeEntry(ir_func, ir_func.type)

    # fill the stubs
    for func in mod.functions.values():
        gen(func, state)

    return state.ir_mod


@gen.register
def gen_function(func: Function, state: IRGenState):
    state.ir_func = state.scopes[func.name.value].node
    state.scopes = state.scopes.new_child()
    state.ctrl = state.ir_func

    # generate parameters
    for param in func.parameters:
        param_ir_type = state.mod.ir_type(param.var_type)
        param_name = param.name.value
        ir_param = ir.Parameter(state.ir_func, param_ir_type, param_name)
        param_size = ir.Constant(state.ir_func, ir.U64, param_ir_type.size)
        addr = ir.Alloca(state.ctrl, param_size, state.mod.target.Ptr)
        state.ctrl = ir.Store(addr, addr, ir_param)
        state.scopes[param_name] = ScopeEntry(addr, param_ir_type)

    # generate body
    gen(func.body, state)

    # restore scope
    state.scopes = state.scopes.parents


@gen.register
def gen_block(block: Block, state: IRGenState, **kw):
    state.scopes = state.scopes.new_child()

    for stmt in block.statements:
        gen(stmt, state, **kw)

    block_scope = state.scopes
    state.scopes = state.scopes.parents
    return block_scope


@gen.register
def gen_assign(assign: Assignment, state: IRGenState, **kw):
    node = gen(assign.right, state, **kw)
    gen(assign.left, state, lvalue=node, **kw)


@gen.register
def gen_varaccess(var: VarAccess, state: IRGenState, lvalue=None, **kw):
    name = var.name.value
    try:
        scope_entry = state.scopes[name]
    except KeyError:
        assert lvalue is not None
        addr = ir.Alloca(state.ctrl, lvalue.type.size, state.mod.target.Ptr)
        scope_entry = ScopeEntry(addr, lvalue.type)

    node = scope_entry.node
    if lvalue is not None:
        # store to this location
        state.ctrl = ir.Store(state.ctrl, node, lvalue)
    else:
        load = ir.Load(state.ctrl, node, scope_entry.type)
        state.ctrl = load
        return load


binops = {
    TokenType.PLUS: ir.Add,
    TokenType.MINUS: ir.Sub,
    TokenType.STAR: ir.Mul,
}


@gen.register
def gen_binop(binop: BinOp, state: IRGenState, **kw):
    left = gen(binop.left, state, **kw)
    right = gen(binop.right, state, **kw)
    node_type = binops[binop.op]
    return node_type(left, right)

