"""
Code generator for TyC.
"""

from __future__ import annotations

from typing import Any, Optional

from ..utils.nodes import (
    ASTNode,
    Program,
    StructDecl,
    MemberDecl,
    FuncDecl,
    Param,
    IntType,
    FloatType,
    StringType,
    VoidType,
    StructType,
    Type,
    BlockStmt,
    VarDecl,
    IfStmt,
    WhileStmt,
    ForStmt,
    SwitchStmt,
    CaseStmt,
    DefaultStmt,
    BreakStmt,
    ContinueStmt,
    ReturnStmt,
    ExprStmt,
    BinaryOp,
    PrefixOp,
    PostfixOp,
    AssignExpr,
    MemberAccess,
    FuncCall,
    Identifier,
    StructLiteral,
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    Expr,
)
from ..utils.visitor import BaseVisitor
from .emitter import Emitter, is_float_type, is_int_type, is_string_type, is_struct_type, is_void_type
from .frame import Frame
from .io import IO_SYMBOL_LIST
from .utils import Access, CName, FunctionType, Index, SubBody, Symbol


class StringArrayType:
    """Marker type for JVM main(String[] args)."""

    pass


class ExprAccess(Access):
    """Expression access context with optional expected type."""

    def __init__(
        self,
        frame,
        sym: list[Symbol],
        is_left: bool = False,
        is_first: bool = False,
        expected_type: Optional[Type] = None,
    ):
        super().__init__(frame, sym, is_left=is_left, is_first=is_first)
        self.expected_type = expected_type


class CodeGenerator(BaseVisitor):
    """AST -> Jasmin code generator for TyC."""

    def __init__(self):
        self.emit: Optional[Emitter] = None
        self.functions: dict[str, Symbol] = {}
        self.structs: dict[str, StructDecl] = {}
        self.current_return_type: Type = VoidType()
        self.class_name = "TyC"

    def _make_access(
        self,
        frame: Frame,
        sym: list[Symbol],
        *,
        is_left: bool = False,
        expected_type: Optional[Type] = None,
    ) -> ExprAccess:
        return ExprAccess(frame, sym, is_left=is_left, expected_type=expected_type)

    def _lookup_symbol(self, name: str, sym_list: list[Symbol]) -> Symbol:
        for sym in reversed(sym_list):
            if sym.name == name:
                return sym
        raise RuntimeError(f"Undeclared symbol: {name}")

    def _lookup_member(self, struct_name: str, member_name: str) -> MemberDecl:
        struct_decl = self.structs.get(struct_name)
        if struct_decl is None:
            raise RuntimeError(f"Undeclared struct: {struct_name}")
        for member in struct_decl.members:
            if member.name == member_name:
                return member
        raise RuntimeError(f"Unknown member '{member_name}' in struct '{struct_name}'")

    def _infer_return_type(self, func_decl: FuncDecl) -> Type:
        """Best-effort inference for omitted function return type."""
        found = self._collect_returns(func_decl.body)
        for ret_stmt in found:
            if ret_stmt.expr is not None:
                param_syms = [Symbol(p.name, p.param_type, Index(-1)) for p in func_decl.params]
                try:
                    inferred = self._infer_type(ret_stmt.expr, param_syms, None)
                    if inferred is not None:
                        return inferred
                except Exception:
                    continue
        return VoidType()

    def _collect_returns(self, node: ASTNode) -> list[ReturnStmt]:
        if isinstance(node, ReturnStmt):
            return [node]

        if isinstance(node, BlockStmt):
            out: list[ReturnStmt] = []
            for stmt in node.statements:
                out.extend(self._collect_returns(stmt))
            return out

        if isinstance(node, IfStmt):
            out = self._collect_returns(node.then_stmt)
            if node.else_stmt is not None:
                out.extend(self._collect_returns(node.else_stmt))
            return out

        if isinstance(node, WhileStmt):
            return self._collect_returns(node.body)

        if isinstance(node, ForStmt):
            return self._collect_returns(node.body)

        if isinstance(node, SwitchStmt):
            out: list[ReturnStmt] = []
            for c in node.cases:
                out.extend(self._collect_returns(c))
            if node.default_case is not None:
                out.extend(self._collect_returns(node.default_case))
            return out

        if isinstance(node, CaseStmt):
            out: list[ReturnStmt] = []
            for stmt in node.statements:
                out.extend(self._collect_returns(stmt))
            return out

        if isinstance(node, DefaultStmt):
            out: list[ReturnStmt] = []
            for stmt in node.statements:
                out.extend(self._collect_returns(stmt))
            return out

        return []

    def _infer_type(self, node: Expr, sym_list: list[Symbol], expected_type: Optional[Type]) -> Optional[Type]:
        if isinstance(node, IntLiteral):
            return IntType()
        if isinstance(node, FloatLiteral):
            return FloatType()
        if isinstance(node, StringLiteral):
            return StringType()
        if isinstance(node, Identifier):
            return self._lookup_symbol(node.name, sym_list).type
        if isinstance(node, FuncCall):
            fn_sym = self.functions.get(node.name)
            return fn_sym.type.return_type if fn_sym else None
        if isinstance(node, AssignExpr):
            return self._infer_type(node.rhs, sym_list, expected_type)
        if isinstance(node, PrefixOp):
            if node.operator in ("++", "--", "!"):
                return IntType()
            return self._infer_type(node.operand, sym_list, expected_type)
        if isinstance(node, PostfixOp):
            return IntType()
        if isinstance(node, MemberAccess):
            obj_type = self._infer_type(node.obj, sym_list, None)
            if obj_type is None or not is_struct_type(obj_type):
                return None
            return self._lookup_member(obj_type.struct_name, node.member).member_type
        if isinstance(node, StructLiteral):
            return expected_type if expected_type is not None and is_struct_type(expected_type) else None
        if isinstance(node, BinaryOp):
            if node.operator in ("&&", "||", "<", "<=", ">", ">=", "==", "!=", "%"):
                return IntType()
            if node.operator in ("+", "-", "*", "/"):
                left_type = self._infer_type(node.left, sym_list, None)
                right_type = self._infer_type(node.right, sym_list, None)
                if left_type is not None and right_type is not None:
                    if is_float_type(left_type) or is_float_type(right_type):
                        return FloatType()
                return IntType()
        return None

    def _emit_struct_classes(self) -> None:
        for struct_decl in self.structs.values():
            struct_emitter = Emitter(f"{struct_decl.name}.j")
            struct_emitter.print_out(struct_emitter.emit_prolog(struct_decl.name))

            for member in struct_decl.members:
                struct_emitter.print_out(
                    f".field public {member.name} {struct_emitter.get_jvm_type(member.member_type)}\n"
                )

            struct_emitter.print_out("\n.method public <init>()V\n")
            struct_emitter.print_out(
                f".var 0 is this L{struct_decl.name}; from Label0 to Label1\n"
            )
            struct_emitter.print_out("Label0:\n")
            struct_emitter.print_out("\taload_0\n")
            struct_emitter.print_out("\tinvokespecial java/lang/Object/<init>()V\n")
            for member in struct_decl.members:
                if is_struct_type(member.member_type):
                    nested = member.member_type.struct_name
                    member_desc = struct_emitter.get_jvm_type(member.member_type)
                    struct_emitter.print_out("\taload_0\n")
                    struct_emitter.print_out(f"\tnew {nested}\n")
                    struct_emitter.print_out("\tdup\n")
                    struct_emitter.print_out(f"\tinvokespecial {nested}/<init>()V\n")
                    struct_emitter.print_out(
                        f"\tputfield {struct_decl.name}/{member.name} {member_desc}\n"
                    )
            struct_emitter.print_out("\treturn\n")
            struct_emitter.print_out("Label1:\n")
            struct_emitter.print_out(".limit stack 4\n")
            struct_emitter.print_out(".limit locals 1\n")
            struct_emitter.print_out(".end method\n")
            struct_emitter.emit_epilog()

    def _build_struct_copy_method(self, struct_decl: StructDecl) -> str:
        struct_name = struct_decl.name
        descriptor = f"(L{struct_name};)L{struct_name};"
        lines: list[str] = []
        lines.append(f"\n.method private static __copy_{struct_name}{descriptor}\n")
        lines.append(f".var 0 is src L{struct_name}; from Label0 to Label2\n")
        lines.append(f".var 1 is dst L{struct_name}; from Label0 to Label2\n")
        lines.append("Label0:\n")
        lines.append("\taload_0\n")
        lines.append("\tifnull Label1\n")
        lines.append(f"\tnew {struct_name}\n")
        lines.append("\tdup\n")
        lines.append(f"\tinvokespecial {struct_name}/<init>()V\n")
        lines.append("\tastore_1\n")

        for member in struct_decl.members:
            member_desc = self.emit.get_jvm_type(member.member_type)
            lines.append("\taload_1\n")
            lines.append("\taload_0\n")
            lines.append(f"\tgetfield {struct_name}/{member.name} {member_desc}\n")
            if is_struct_type(member.member_type):
                nested = member.member_type.struct_name
                lines.append(
                    f"\tinvokestatic {self.class_name}/__copy_{nested}(L{nested};)L{nested};\n"
                )
            lines.append(f"\tputfield {struct_name}/{member.name} {member_desc}\n")

        lines.append("\taload_1\n")
        lines.append("\tareturn\n")
        lines.append("Label1:\n")
        lines.append("\taconst_null\n")
        lines.append("\tareturn\n")
        lines.append("Label2:\n")
        lines.append(".limit stack 8\n")
        lines.append(".limit locals 2\n")
        lines.append(".end method\n")
        return "".join(lines)

    def _emit_struct_copy_helpers(self) -> None:
        for struct_decl in self.structs.values():
            self.emit.print_out(self._build_struct_copy_method(struct_decl))

    def _emit_clone_struct(self, struct_type: StructType, frame: Frame) -> str:
        clone_type = FunctionType([StructType(struct_type.struct_name)], StructType(struct_type.struct_name))
        return self.emit.emit_invoke_static(
            f"{self.class_name}/__copy_{struct_type.struct_name}", clone_type, frame
        )

    def _emit_type_adjust(self, src_type: Type, dst_type: Type, frame: Frame, *, clone_struct: bool) -> str:
        if is_int_type(src_type) and is_float_type(dst_type):
            return self.emit.emit_i2f(frame)

        if is_struct_type(src_type) and is_struct_type(dst_type):
            if src_type.struct_name != dst_type.struct_name:
                raise RuntimeError(
                    f"Type mismatch in struct conversion: {src_type.struct_name} -> {dst_type.struct_name}"
                )
            if clone_struct:
                return self._emit_clone_struct(dst_type, frame)

        if (
            type(src_type) is not type(dst_type)
            and not (is_int_type(src_type) and is_float_type(dst_type))
        ):
            raise RuntimeError(f"Type mismatch: cannot convert {src_type} to {dst_type}")
        return ""

    def _emit_default_init(self, var_type: Type, frame: Frame) -> str:
        if is_int_type(var_type):
            return self.emit.emit_push_iconst(0, frame)
        if is_float_type(var_type):
            return self.emit.emit_push_fconst("0.0", frame)
        if is_string_type(var_type):
            frame.push()
            return self.emit.jvm.emitPUSHNULL()
        if is_struct_type(var_type):
            return self.emit.emit_new_instance(var_type.struct_name, frame)
        raise RuntimeError(f"Unsupported default initialization type: {var_type}")

    def _emit_member_obj_and_field(self, node: MemberAccess, o: ExprAccess) -> tuple[str, str, Type, str]:
        obj_code, obj_type = self.visit(node.obj, self._make_access(o.frame, o.sym))
        if not is_struct_type(obj_type):
            raise RuntimeError("Member access on non-struct type")
        member_decl = self._lookup_member(obj_type.struct_name, node.member)
        return obj_code, obj_type.struct_name, member_decl.member_type, node.member

    def _emit_cond_jump_false(self, cond_type: Type, label: int, frame: Frame) -> str:
        if is_float_type(cond_type):
            return (
                self.emit.emit_push_fconst("0.0", frame)
                + self.emit.emit_re_op("==", FloatType(), frame)
                + self.emit.emit_if_true(label, frame)
            )
        return self.emit.emit_if_false(label, frame)

    def _emit_cond_jump_true(self, cond_type: Type, label: int, frame: Frame) -> str:
        if is_float_type(cond_type):
            return (
                self.emit.emit_push_fconst("0.0", frame)
                + self.emit.emit_re_op("!=", FloatType(), frame)
                + self.emit.emit_if_true(label, frame)
            )
        return self.emit.emit_if_true(label, frame)

    def visit_program(self, node: Program, o: Any = None):
        self.structs = {
            decl.name: decl for decl in node.decls if isinstance(decl, StructDecl)
        }
        self._emit_struct_classes()

        self.emit = Emitter(f"{self.class_name}.j")
        self.emit.print_out(self.emit.emit_prolog(self.class_name))

        self.functions = {}
        for io_sym in IO_SYMBOL_LIST:
            self.functions[io_sym.name] = io_sym

        for decl in node.decls:
            if isinstance(decl, FuncDecl):
                return_type = decl.return_type if decl.return_type is not None else self._infer_return_type(decl)
                param_types = [param.param_type for param in decl.params]
                self.functions[decl.name] = Symbol(
                    decl.name,
                    FunctionType(param_types, return_type),
                    CName(self.class_name),
                )

        self._emit_struct_copy_helpers()

        for decl in node.decls:
            if isinstance(decl, FuncDecl):
                self.visit(decl, None)

        self.emit.emit_epilog()

    def visit_func_decl(self, node: FuncDecl, o: Any = None):
        if node.name not in self.functions:
            raise RuntimeError(f"Unknown function: {node.name}")

        fn_type = self.functions[node.name].type
        self.current_return_type = fn_type.return_type

        frame = Frame(node.name, self.current_return_type)
        frame.enter_scope(True)

        if node.name == "main":
            method_type = FunctionType([StringArrayType()], VoidType())
            self.current_return_type = VoidType()
        else:
            method_type = FunctionType([param.param_type for param in node.params], self.current_return_type)

        self.emit.print_out(self.emit.emit_method(node.name, method_type, True))

        start_label = frame.get_start_label()
        end_label = frame.get_end_label()
        self.emit.print_out(self.emit.emit_label(start_label, frame))

        local_syms: list[Symbol] = []
        if node.name == "main":
            args_idx = frame.get_new_index()
            self.emit.print_out(
                self.emit.emit_var(args_idx, "args", StringArrayType(), start_label, end_label)
            )

        for param in node.params:
            idx = frame.get_new_index()
            self.emit.print_out(
                self.emit.emit_var(idx, param.name, param.param_type, start_label, end_label)
            )
            local_syms.append(Symbol(param.name, param.param_type, Index(idx)))

        self.visit(node.body, SubBody(frame, local_syms))

        if is_void_type(self.current_return_type):
            self.emit.print_out(self.emit.emit_return(VoidType(), frame))

        self.emit.print_out(self.emit.emit_label(end_label, frame))
        frame.exit_scope()
        self.emit.print_out(self.emit.emit_end_method(frame))

    def visit_block_stmt(self, node: BlockStmt, o: SubBody = None):
        frame = o.frame
        old_len = len(o.sym)
        frame.enter_scope(False)

        block_start = frame.get_start_label()
        block_end = frame.get_end_label()
        self.emit.print_out(self.emit.emit_label(block_start, frame))

        block_ctx = SubBody(frame, o.sym)
        for stmt in node.statements:
            block_ctx = self.visit(stmt, block_ctx)

        self.emit.print_out(self.emit.emit_label(block_end, frame))
        frame.exit_scope()
        del o.sym[old_len:]
        return o

    def visit_var_decl(self, node: VarDecl, o: SubBody = None):
        frame = o.frame
        idx = frame.get_new_index()

        init_code = ""
        init_type: Optional[Type] = None
        var_type = node.var_type

        if node.init_value is not None:
            init_code, init_type = self.visit(
                node.init_value, self._make_access(frame, o.sym, expected_type=var_type)
            )
            if var_type is None:
                var_type = init_type
                if var_type is None:
                    raise RuntimeError(f"Cannot infer type for variable: {node.name}")
            init_code += self._emit_type_adjust(
                init_type, var_type, frame, clone_struct=is_struct_type(var_type)
            )
        else:
            if var_type is None:
                raise RuntimeError(
                    f"Variable '{node.name}' with auto type requires initializer for code generation"
                )
            init_code = self._emit_default_init(var_type, frame)

        self.emit.print_out(
            self.emit.emit_var(
                idx, node.name, var_type, frame.get_start_label(), frame.get_end_label()
            )
        )
        self.emit.print_out(init_code)
        self.emit.print_out(self.emit.emit_write_var(node.name, var_type, idx, frame))

        o.sym.append(Symbol(node.name, var_type, Index(idx)))
        return o

    def visit_expr_stmt(self, node: ExprStmt, o: SubBody = None):
        code, expr_type = self.visit(node.expr, self._make_access(o.frame, o.sym))
        self.emit.print_out(code)
        if not is_void_type(expr_type):
            self.emit.print_out(self.emit.emit_pop(o.frame))
        return o

    def visit_if_stmt(self, node: IfStmt, o: SubBody = None):
        frame = o.frame
        else_label = frame.get_new_label()
        end_label = frame.get_new_label()

        cond_code, cond_type = self.visit(node.condition, self._make_access(frame, o.sym))
        self.emit.print_out(cond_code)
        self.emit.print_out(self._emit_cond_jump_false(cond_type, else_label, frame))
        self.visit(node.then_stmt, o)
        self.emit.print_out(self.emit.emit_goto(end_label, frame))
        self.emit.print_out(self.emit.emit_label(else_label, frame))
        if node.else_stmt is not None:
            self.visit(node.else_stmt, o)
        self.emit.print_out(self.emit.emit_label(end_label, frame))
        return o

    def visit_while_stmt(self, node: WhileStmt, o: SubBody = None):
        frame = o.frame
        frame.enter_loop()
        cond_label = frame.get_continue_label()
        break_label = frame.get_break_label()

        self.emit.print_out(self.emit.emit_label(cond_label, frame))
        cond_code, cond_type = self.visit(node.condition, self._make_access(frame, o.sym))
        self.emit.print_out(cond_code)
        self.emit.print_out(self._emit_cond_jump_false(cond_type, break_label, frame))

        self.visit(node.body, o)
        self.emit.print_out(self.emit.emit_goto(cond_label, frame))
        self.emit.print_out(self.emit.emit_label(break_label, frame))
        frame.exit_loop()
        return o

    def visit_for_stmt(self, node: ForStmt, o: SubBody = None):
        frame = o.frame
        old_len = len(o.sym)
        frame.enter_scope(False)

        scope_start = frame.get_start_label()
        scope_end = frame.get_end_label()
        self.emit.print_out(self.emit.emit_label(scope_start, frame))

        for_ctx = SubBody(frame, o.sym)
        if node.init is not None:
            for_ctx = self.visit(node.init, for_ctx)

        frame.enter_loop()
        continue_label = frame.get_continue_label()
        break_label = frame.get_break_label()
        cond_label = frame.get_new_label()

        self.emit.print_out(self.emit.emit_label(cond_label, frame))
        if node.condition is not None:
            cond_code, cond_type = self.visit(node.condition, self._make_access(frame, o.sym))
            self.emit.print_out(cond_code)
            self.emit.print_out(self._emit_cond_jump_false(cond_type, break_label, frame))

        self.visit(node.body, for_ctx)

        self.emit.print_out(self.emit.emit_label(continue_label, frame))
        if node.update is not None:
            update_code, update_type = self.visit(node.update, self._make_access(frame, o.sym))
            self.emit.print_out(update_code)
            if not is_void_type(update_type):
                self.emit.print_out(self.emit.emit_pop(frame))

        self.emit.print_out(self.emit.emit_goto(cond_label, frame))
        self.emit.print_out(self.emit.emit_label(break_label, frame))
        frame.exit_loop()

        self.emit.print_out(self.emit.emit_label(scope_end, frame))
        frame.exit_scope()
        del o.sym[old_len:]
        return o

    def visit_switch_stmt(self, node: SwitchStmt, o: SubBody = None):
        frame = o.frame
        old_len = len(o.sym)
        frame.enter_scope(False)

        scope_start = frame.get_start_label()
        scope_end = frame.get_end_label()
        self.emit.print_out(self.emit.emit_label(scope_start, frame))

        switch_idx = frame.get_new_index()
        switch_name = f"$switch{switch_idx}"
        self.emit.print_out(
            self.emit.emit_var(
                switch_idx, switch_name, IntType(), scope_start, scope_end
            )
        )

        expr_code, expr_type = self.visit(node.expr, self._make_access(frame, o.sym))
        self.emit.print_out(expr_code)
        if is_float_type(expr_type):
            raise RuntimeError("Switch expression must be int")
        self.emit.print_out(self.emit.emit_write_var(switch_name, IntType(), switch_idx, frame))

        end_label = frame.get_new_label()
        case_labels = [frame.get_new_label() for _ in node.cases]
        default_label = frame.get_new_label() if node.default_case is not None else end_label

        frame.brk_label.append(end_label)
        for case_stmt, case_label in zip(node.cases, case_labels):
            self.emit.print_out(self.emit.emit_read_var(switch_name, IntType(), switch_idx, frame))
            case_code, case_type = self.visit(case_stmt.expr, self._make_access(frame, o.sym))
            if is_float_type(case_type):
                raise RuntimeError("Case expression must be int")
            self.emit.print_out(case_code)
            self.emit.print_out(self.emit.emit_re_op("==", IntType(), frame))
            self.emit.print_out(self.emit.emit_if_true(case_label, frame))

        self.emit.print_out(self.emit.emit_goto(default_label, frame))

        for case_stmt, case_label in zip(node.cases, case_labels):
            self.emit.print_out(self.emit.emit_label(case_label, frame))
            self.visit(case_stmt, o)

        if node.default_case is not None:
            self.emit.print_out(self.emit.emit_label(default_label, frame))
            self.visit(node.default_case, o)

        self.emit.print_out(self.emit.emit_label(end_label, frame))
        frame.brk_label.pop()

        self.emit.print_out(self.emit.emit_label(scope_end, frame))
        frame.exit_scope()
        del o.sym[old_len:]
        return o

    def visit_case_stmt(self, node: CaseStmt, o: SubBody = None):
        for stmt in node.statements:
            self.visit(stmt, o)
        return o

    def visit_default_stmt(self, node: DefaultStmt, o: SubBody = None):
        for stmt in node.statements:
            self.visit(stmt, o)
        return o

    def visit_break_stmt(self, node: BreakStmt, o: SubBody = None):
        self.emit.print_out(self.emit.emit_goto(o.frame.get_break_label(), o.frame))
        return o

    def visit_continue_stmt(self, node: ContinueStmt, o: SubBody = None):
        self.emit.print_out(self.emit.emit_goto(o.frame.get_continue_label(), o.frame))
        return o

    def visit_return_stmt(self, node: ReturnStmt, o: SubBody = None):
        frame = o.frame
        if node.expr is None:
            self.emit.print_out(self.emit.emit_return(VoidType(), frame))
            return o

        ret_code, ret_type = self.visit(
            node.expr, self._make_access(frame, o.sym, expected_type=self.current_return_type)
        )
        self.emit.print_out(ret_code)
        self.emit.print_out(
            self._emit_type_adjust(
                ret_type,
                self.current_return_type,
                frame,
                clone_struct=is_struct_type(self.current_return_type),
            )
        )
        self.emit.print_out(self.emit.emit_return(self.current_return_type, frame))
        return o

    def visit_binary_op(self, node: BinaryOp, o: ExprAccess = None):
        frame = o.frame

        if node.operator == "&&":
            false_label = frame.get_new_label()
            end_label = frame.get_new_label()
            left_code, left_type = self.visit(node.left, self._make_access(frame, o.sym))
            right_code, right_type = self.visit(node.right, self._make_access(frame, o.sym))
            code = left_code
            code += self._emit_cond_jump_false(left_type, false_label, frame)
            code += right_code
            code += self._emit_cond_jump_false(right_type, false_label, frame)
            code += self.emit.emit_push_iconst(1, frame)
            code += self.emit.emit_goto(end_label, frame)
            code += self.emit.emit_label(false_label, frame)
            code += self.emit.emit_push_iconst(0, frame)
            code += self.emit.emit_label(end_label, frame)
            return code, IntType()

        if node.operator == "||":
            true_label = frame.get_new_label()
            end_label = frame.get_new_label()
            left_code, left_type = self.visit(node.left, self._make_access(frame, o.sym))
            right_code, right_type = self.visit(node.right, self._make_access(frame, o.sym))
            code = left_code
            code += self._emit_cond_jump_true(left_type, true_label, frame)
            code += right_code
            code += self._emit_cond_jump_true(right_type, true_label, frame)
            code += self.emit.emit_push_iconst(0, frame)
            code += self.emit.emit_goto(end_label, frame)
            code += self.emit.emit_label(true_label, frame)
            code += self.emit.emit_push_iconst(1, frame)
            code += self.emit.emit_label(end_label, frame)
            return code, IntType()

        left_code, left_type = self.visit(node.left, self._make_access(frame, o.sym))
        right_code, right_type = self.visit(node.right, self._make_access(frame, o.sym))

        if node.operator in ("+", "-", "*", "/"):
            result_type: Type = FloatType() if is_float_type(left_type) or is_float_type(right_type) else IntType()
            code = left_code
            if is_float_type(result_type):
                code += self._emit_type_adjust(left_type, result_type, frame, clone_struct=False)
            code += right_code
            if is_float_type(result_type):
                code += self._emit_type_adjust(right_type, result_type, frame, clone_struct=False)

            if node.operator in ("+", "-"):
                code += self.emit.emit_add_op(node.operator, result_type, frame)
            else:
                code += self.emit.emit_mul_op(node.operator, result_type, frame)
            return code, result_type

        if node.operator == "%":
            return left_code + right_code + self.emit.emit_mod(frame), IntType()

        if node.operator in ("<", "<=", ">", ">=", "==", "!="):
            cmp_type: Type = FloatType() if is_float_type(left_type) or is_float_type(right_type) else IntType()
            code = left_code
            if is_float_type(cmp_type):
                code += self._emit_type_adjust(left_type, cmp_type, frame, clone_struct=False)
            code += right_code
            if is_float_type(cmp_type):
                code += self._emit_type_adjust(right_type, cmp_type, frame, clone_struct=False)
            code += self.emit.emit_re_op(node.operator, cmp_type, frame)
            return code, IntType()

        raise RuntimeError(f"Unsupported operator: {node.operator}")

    def visit_prefix_op(self, node: PrefixOp, o: ExprAccess = None):
        frame = o.frame
        if node.operator == "+":
            code, typ = self.visit(node.operand, self._make_access(frame, o.sym))
            return code, typ

        if node.operator == "-":
            code, typ = self.visit(node.operand, self._make_access(frame, o.sym))
            return code + self.emit.emit_neg_op(typ, frame), typ

        if node.operator == "!":
            code, typ = self.visit(node.operand, self._make_access(frame, o.sym))
            code += self.emit.emit_push_iconst(0, frame)
            code += self.emit.emit_re_op("==", IntType(), frame)
            return code, IntType()

        if node.operator in ("++", "--"):
            delta_op = "+" if node.operator == "++" else "-"
            if isinstance(node.operand, Identifier):
                sym = self._lookup_symbol(node.operand.name, o.sym)
                idx = sym.value.value
                code = self.emit.emit_read_var(sym.name, sym.type, idx, frame)
                code += self.emit.emit_push_iconst(1, frame)
                code += self.emit.emit_add_op(delta_op, IntType(), frame)
                code += self.emit.emit_dup(frame)
                code += self.emit.emit_write_var(sym.name, sym.type, idx, frame)
                return code, IntType()

            if isinstance(node.operand, MemberAccess):
                obj_code, owner_name, member_type, member_name = self._emit_member_obj_and_field(
                    node.operand, self._make_access(frame, o.sym)
                )
                code = obj_code
                code += self.emit.emit_dup(frame)
                code += self.emit.emit_get_field(f"{owner_name}/{member_name}", member_type, frame)
                code += self.emit.emit_push_iconst(1, frame)
                code += self.emit.emit_add_op(delta_op, IntType(), frame)
                code += self.emit.emit_dup_x1(frame)
                code += self.emit.emit_put_field(f"{owner_name}/{member_name}", member_type, frame)
                return code, IntType()

        raise RuntimeError(f"Unsupported prefix operator: {node.operator}")

    def visit_postfix_op(self, node: PostfixOp, o: ExprAccess = None):
        frame = o.frame
        if node.operator not in ("++", "--"):
            raise RuntimeError(f"Unsupported postfix operator: {node.operator}")

        delta_op = "+" if node.operator == "++" else "-"

        if isinstance(node.operand, Identifier):
            sym = self._lookup_symbol(node.operand.name, o.sym)
            idx = sym.value.value
            code = self.emit.emit_read_var(sym.name, sym.type, idx, frame)
            code += self.emit.emit_dup(frame)
            code += self.emit.emit_push_iconst(1, frame)
            code += self.emit.emit_add_op(delta_op, IntType(), frame)
            code += self.emit.emit_write_var(sym.name, sym.type, idx, frame)
            return code, IntType()

        if isinstance(node.operand, MemberAccess):
            obj_code, owner_name, member_type, member_name = self._emit_member_obj_and_field(
                node.operand, self._make_access(frame, o.sym)
            )
            code = obj_code
            code += self.emit.emit_dup(frame)
            code += self.emit.emit_get_field(f"{owner_name}/{member_name}", member_type, frame)
            code += self.emit.emit_dup_x1(frame)
            code += self.emit.emit_push_iconst(1, frame)
            code += self.emit.emit_add_op(delta_op, IntType(), frame)
            code += self.emit.emit_put_field(f"{owner_name}/{member_name}", member_type, frame)
            return code, IntType()

        raise RuntimeError("Postfix operator requires identifier or member access")

    def visit_assign_expr(self, node: AssignExpr, o: ExprAccess = None):
        frame = o.frame

        if isinstance(node.lhs, Identifier):
            lhs_sym = self._lookup_symbol(node.lhs.name, o.sym)
            if lhs_sym.type is None:
                raise RuntimeError(f"Cannot assign to unresolved auto variable: {lhs_sym.name}")
            rhs_code, rhs_type = self.visit(
                node.rhs, self._make_access(frame, o.sym, expected_type=lhs_sym.type)
            )
            rhs_code += self._emit_type_adjust(
                rhs_type, lhs_sym.type, frame, clone_struct=is_struct_type(lhs_sym.type)
            )
            code = rhs_code
            code += self.emit.emit_dup(frame)
            code += self.emit.emit_write_var(lhs_sym.name, lhs_sym.type, lhs_sym.value.value, frame)
            return code, lhs_sym.type

        if isinstance(node.lhs, MemberAccess):
            obj_code, owner_name, field_type, member_name = self._emit_member_obj_and_field(
                node.lhs, self._make_access(frame, o.sym)
            )
            rhs_code, rhs_type = self.visit(
                node.rhs, self._make_access(frame, o.sym, expected_type=field_type)
            )
            rhs_code += self._emit_type_adjust(
                rhs_type, field_type, frame, clone_struct=is_struct_type(field_type)
            )
            code = obj_code + rhs_code
            code += self.emit.emit_dup_x1(frame)
            code += self.emit.emit_put_field(f"{owner_name}/{member_name}", field_type, frame)
            return code, field_type

        raise RuntimeError("Assignment target must be identifier or member access")

    def visit_member_access(self, node: MemberAccess, o: ExprAccess = None):
        obj_code, obj_type = self.visit(node.obj, self._make_access(o.frame, o.sym))
        if not is_struct_type(obj_type):
            raise RuntimeError("Member access on non-struct type")
        member_decl = self._lookup_member(obj_type.struct_name, node.member)
        code = obj_code + self.emit.emit_get_field(
            f"{obj_type.struct_name}/{node.member}", member_decl.member_type, o.frame
        )
        return code, member_decl.member_type

    def visit_func_call(self, node: FuncCall, o: ExprAccess = None):
        frame = o.frame
        fn_sym = self.functions.get(node.name)
        if fn_sym is None:
            raise RuntimeError(f"Undeclared function: {node.name}")

        fn_type: FunctionType = fn_sym.type
        if len(node.args) != len(fn_type.param_types):
            raise RuntimeError(
                f"Function '{node.name}' expects {len(fn_type.param_types)} args, got {len(node.args)}"
            )

        code = ""
        for arg, param_type in zip(node.args, fn_type.param_types):
            arg_code, arg_type = self.visit(
                arg, self._make_access(frame, o.sym, expected_type=param_type)
            )
            arg_code += self._emit_type_adjust(
                arg_type, param_type, frame, clone_struct=is_struct_type(param_type)
            )
            code += arg_code

        code += self.emit.emit_invoke_static(
            f"{fn_sym.value.value}/{node.name}", fn_type, frame
        )
        return code, fn_type.return_type

    def visit_identifier(self, node: Identifier, o: ExprAccess = None):
        sym = self._lookup_symbol(node.name, o.sym)
        if sym.type is None:
            raise RuntimeError(f"Cannot read unresolved auto variable: {sym.name}")
        return self.emit.emit_read_var(sym.name, sym.type, sym.value.value, o.frame), sym.type

    def visit_struct_literal(self, node: StructLiteral, o: ExprAccess = None):
        expected_type = getattr(o, "expected_type", None)
        if expected_type is None or not is_struct_type(expected_type):
            raise RuntimeError("Cannot infer struct literal type in code generation")

        struct_decl = self.structs.get(expected_type.struct_name)
        if struct_decl is None:
            raise RuntimeError(f"Undeclared struct: {expected_type.struct_name}")
        if len(node.values) != len(struct_decl.members):
            raise RuntimeError(
                f"Struct literal for '{expected_type.struct_name}' expects {len(struct_decl.members)} values, got {len(node.values)}"
            )

        frame = o.frame
        code = self.emit.emit_new_instance(expected_type.struct_name, frame)

        for value_expr, member_decl in zip(node.values, struct_decl.members):
            value_code, value_type = self.visit(
                value_expr,
                self._make_access(frame, o.sym, expected_type=member_decl.member_type),
            )
            value_code += self._emit_type_adjust(
                value_type,
                member_decl.member_type,
                frame,
                clone_struct=is_struct_type(member_decl.member_type),
            )

            code += self.emit.emit_dup(frame)
            code += value_code
            code += self.emit.emit_put_field(
                f"{expected_type.struct_name}/{member_decl.name}", member_decl.member_type, frame
            )

        return code, StructType(expected_type.struct_name)

    def visit_int_literal(self, node: IntLiteral, o: ExprAccess = None):
        return self.emit.emit_push_iconst(node.value, o.frame), IntType()

    def visit_float_literal(self, node: FloatLiteral, o: ExprAccess = None):
        return self.emit.emit_push_fconst(str(node.value), o.frame), FloatType()

    def visit_string_literal(self, node: StringLiteral, o: ExprAccess = None):
        return self.emit.emit_push_const(node.value, StringType(), o.frame), StringType()

    def visit_struct_decl(self, node: StructDecl, o: Any = None):
        return None

    def visit_member_decl(self, node: MemberDecl, o: Any = None):
        return None

    def visit_param(self, node: Param, o: Any = None):
        return None

    def visit_int_type(self, node: IntType, o: Any = None):
        return node

    def visit_float_type(self, node: FloatType, o: Any = None):
        return node

    def visit_string_type(self, node: StringType, o: Any = None):
        return node

    def visit_void_type(self, node: VoidType, o: Any = None):
        return node

    def visit_struct_type(self, node: StructType, o: Any = None):
        return node
