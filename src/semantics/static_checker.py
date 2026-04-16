"""
Static Semantic Checker for TyC Programming Language

This module implements a comprehensive static semantic checker using visitor pattern
for the TyC procedural programming language. It performs type checking,
scope management, type inference, and detects all semantic errors as
specified in the TyC language specification.
"""

from functools import reduce
from typing import (
    Dict,
    List,
    Set,
    Optional,
    Any,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from ..utils.visitor import ASTVisitor
from ..utils.nodes import (
    ASTNode,
    Program,
    StructDecl,
    MemberDecl,
    FuncDecl,
    Param,
    VarDecl,
    IfStmt,
    WhileStmt,
    ForStmt,
    BreakStmt,
    ContinueStmt,
    ReturnStmt,
    BlockStmt,
    SwitchStmt,
    CaseStmt,
    DefaultStmt,
    Type,
    IntType,
    FloatType,
    StringType,
    VoidType,
    StructType,
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
    ExprStmt,
    Expr,
    Stmt,
    Decl,
)
from .static_error import (
    StaticError,
    Redeclared,
    UndeclaredIdentifier,
    UndeclaredFunction,
    UndeclaredStruct,
    TypeCannotBeInferred,
    TypeMismatchInStatement,
    TypeMismatchInExpression,
    MustInLoop,
)

class UnknownType(Type):
    def __str__(self):
        return "UnknownType()"
    def accept(self, visitor, o=None):
        pass

TyCType = Union[IntType, FloatType, StringType, VoidType, StructType, UnknownType]

class StaticChecker(ASTVisitor):
    def __init__(self):
        self.global_structs: Dict[str, StructDecl] = {}
        # func env maps name to (return_type, List[param_types])
        self.global_funcs: Dict[str, Tuple[Type, List[Type]]] = {
            "readInt": (IntType(), []),
            "readFloat": (FloatType(), []),
            "readString": (StringType(), []),
            "printInt": (VoidType(), [IntType()]),
            "printFloat": (VoidType(), [FloatType()]),
            "printString": (VoidType(), [StringType()]),
        }
        # index 0 is global/outermost local, the last is innermost
        self.local_envs: List[Dict[str, Type]] = []
        
        self.in_loops = 0
        self.in_switch = 0
        self.current_func_name: Optional[str] = None
        self.current_func_ret_type: Optional[Type] = None
        self.current_func_params: Set[str] = set()

    def check_program(self, ast: ASTNode):
        return self.visit(ast)

    def enter_scope(self):
        self.local_envs.append({})

    def exit_scope(self):
        self.local_envs.pop()

    def declare_var(self, name: str, var_type: Type):
        if name in self.local_envs[-1]:
            raise Redeclared("Variable", name)
        if self.current_func_params and name in self.current_func_params:
            raise Redeclared("Variable", name)
        self.local_envs[-1][name] = var_type

    def declare_param(self, name: str, var_type: Type):
        if name in self.local_envs[-1]:
            raise Redeclared("Parameter", name)
        self.local_envs[-1][name] = var_type
        self.current_func_params.add(name)

    def lookup_var(self, name: str) -> Optional[Type]:
        for env in reversed(self.local_envs):
            if name in env:
                return env[name]
        return None

    def update_var_type(self, name: str, new_type: Type):
        for env in reversed(self.local_envs):
            if name in env:
                env[name] = new_type
                return

    def is_same_type(self, t1: Type, t2: Type) -> bool:
        if isinstance(t1, UnknownType) or isinstance(t2, UnknownType):
            return False
        if type(t1) != type(t2):
            return False
        if isinstance(t1, StructType) and isinstance(t2, StructType):
            return t1.struct_name == t2.struct_name
        return True

    def infer_var_type(self, expr: Expr, new_type: Type):
        if isinstance(expr, Identifier):
            self.update_var_type(expr.name, new_type)

    def visit_program(self, node: "Program", o: Any = None):
        for decl in node.decls:
            self.visit(decl, o)

    def visit_struct_decl(self, node: "StructDecl", o: Any = None):
        if node.name in self.global_structs:
            raise Redeclared("Struct", node.name)
        
        member_names = set()
        for m in node.members:
            if m.name in member_names:
                raise Redeclared("Member", m.name)
            member_names.add(m.name)
            
            if m.member_type is None:
                continue # Let it fail if somehow AST allowed None
                
            if isinstance(m.member_type, StructType):
                if m.member_type.struct_name not in self.global_structs:
                    raise UndeclaredStruct(m.member_type.struct_name)
                    
        self.global_structs[node.name] = node

    def visit_func_decl(self, node: "FuncDecl", o: Any = None):
        if node.name in self.global_funcs:
            raise Redeclared("Function", node.name)
            
        ret_type = node.return_type
        
        param_types = []
        param_names = set()
        for p in node.params:
            if p.name in param_names:
                raise Redeclared("Parameter", p.name)
            param_names.add(p.name)
            
            if isinstance(p.param_type, StructType):
                if p.param_type.struct_name not in self.global_structs:
                    raise UndeclaredStruct(p.param_type.struct_name)
            param_types.append(p.param_type)
            
        self.global_funcs[node.name] = (ret_type if ret_type else UnknownType(), param_types)
        
        self.current_func_name = node.name
        self.current_func_ret_type = ret_type
        self.current_func_params = param_names
        
        self.enter_scope()
        for p in node.params:
            self.declare_param(p.name, p.param_type)
            
        if isinstance(node.body, BlockStmt):
            self.visit(node.body, o)
        elif isinstance(node.body, list):
            for stmt in node.body:
                self.visit(stmt, o)
            
        self.exit_scope()
        
        if self.current_func_ret_type is None:
            self.current_func_ret_type = VoidType()
            self.global_funcs[node.name] = (VoidType(), param_types)
            
        self.current_func_name = None
        self.current_func_ret_type = None
        self.current_func_params = set()

    def visit_block_stmt(self, node: "BlockStmt", o: Any = None):
        self.enter_scope()
        for stmt in node.statements:
            self.visit(stmt, o)
            
        for k, v in self.local_envs[-1].items():
            if isinstance(v, UnknownType):
                raise TypeCannotBeInferred(node)
                
        self.exit_scope()

    def visit_var_decl(self, node: "VarDecl", o: Any = None):
        if node.var_type is not None:
            if isinstance(node.var_type, StructType):
                if node.var_type.struct_name not in self.global_structs:
                    raise UndeclaredStruct(node.var_type.struct_name)

        if node.init_value:
            init_t = self.visit(node.init_value, node.var_type)
            
            if node.var_type is None:
                if isinstance(init_t, UnknownType):
                    raise TypeCannotBeInferred(node)
                node.var_type = init_t
                self.declare_var(node.name, init_t)
            else:
                if isinstance(init_t, UnknownType):
                    self.infer_var_type(node.init_value, node.var_type)
                    init_t = node.var_type
                    
                if not self.is_same_type(node.var_type, init_t):
                    raise TypeMismatchInStatement(node)
                self.declare_var(node.name, node.var_type)
        else:
            if node.var_type is None:
                self.declare_var(node.name, UnknownType())
            else:
                self.declare_var(node.name, node.var_type)

    def visit_if_stmt(self, node: "IfStmt", o: Any = None):
        cond_t = self.visit(node.condition, o)
        if isinstance(cond_t, UnknownType):
            self.infer_var_type(node.condition, IntType())
            cond_t = IntType()
        if type(cond_t) != IntType:
            raise TypeMismatchInStatement(node)
            
        self.visit(node.then_stmt, o)
        if node.else_stmt:
            self.visit(node.else_stmt, o)

    def visit_while_stmt(self, node: "WhileStmt", o: Any = None):
        cond_t = self.visit(node.condition, o)
        if isinstance(cond_t, UnknownType):
            self.infer_var_type(node.condition, IntType())
            cond_t = IntType()
        if type(cond_t) != IntType:
            raise TypeMismatchInStatement(node)
            
        self.in_loops += 1
        self.visit(node.body, o)
        self.in_loops -= 1

    def visit_for_stmt(self, node: "ForStmt", o: Any = None):
        self.enter_scope()
        
        if node.init:
            self.visit(node.init, o)
            
        if node.condition:
            cond_t = self.visit(node.condition, o)
            if isinstance(cond_t, UnknownType):
                self.infer_var_type(node.condition, IntType())
                cond_t = IntType()
            if type(cond_t) != IntType:
                raise TypeMismatchInStatement(node)
                
        if node.update:
            self.visit(node.update, o)
            
        self.in_loops += 1
        self.visit(node.body, o)
        self.in_loops -= 1
        
        for k, v in self.local_envs[-1].items():
            if isinstance(v, UnknownType):
                raise TypeCannotBeInferred(node)
                
        self.exit_scope()

    def visit_switch_stmt(self, node: "SwitchStmt", o: Any = None):
        expr_t = self.visit(node.expr, o)
        if isinstance(expr_t, UnknownType):
            self.infer_var_type(node.expr, IntType())
            expr_t = IntType()
        if type(expr_t) != IntType:
            raise TypeMismatchInStatement(node)
            
        self.in_switch += 1
        for case_stmt in node.cases:
            self.visit(case_stmt, o)
        if node.default_case:
            self.visit(node.default_case, o)
        self.in_switch -= 1

    def visit_case_stmt(self, node: "CaseStmt", o: Any = None):
        case_t = self.visit(node.expr, o)
        if isinstance(case_t, UnknownType):
            self.infer_var_type(node.expr, IntType())
            case_t = IntType()
        if type(case_t) != IntType:
            raise TypeMismatchInStatement(node)
            
        for stmt in node.statements:
            self.visit(stmt, o)

    def visit_default_stmt(self, node: "DefaultStmt", o: Any = None):
        for stmt in node.statements:
            self.visit(stmt, o)

    def visit_break_stmt(self, node: "BreakStmt", o: Any = None):
        if self.in_loops == 0 and self.in_switch == 0:
            raise MustInLoop(node)

    def visit_continue_stmt(self, node: "ContinueStmt", o: Any = None):
        if self.in_loops == 0:
            raise MustInLoop(node)

    def visit_return_stmt(self, node: "ReturnStmt", o: Any = None):
        if self.current_func_ret_type is None:
            if node.expr is None:
                self.current_func_ret_type = VoidType()
            else:
                expr_t = self.visit(node.expr, o)
                if isinstance(expr_t, UnknownType):
                    raise TypeCannotBeInferred(node)
                self.current_func_ret_type = expr_t
            self.global_funcs[self.current_func_name] = (self.current_func_ret_type, self.global_funcs[self.current_func_name][1])
        else:
            if node.expr is None:
                if not isinstance(self.current_func_ret_type, VoidType):
                    raise TypeMismatchInStatement(node)
            else:
                if isinstance(self.current_func_ret_type, VoidType):
                    raise TypeMismatchInStatement(node)
                    
                expr_t = self.visit(node.expr, self.current_func_ret_type)
                if isinstance(expr_t, UnknownType):
                    self.infer_var_type(node.expr, self.current_func_ret_type)
                    expr_t = self.current_func_ret_type
                if not self.is_same_type(expr_t, self.current_func_ret_type):
                    raise TypeMismatchInStatement(node)

    def visit_expr_stmt(self, node: "ExprStmt", o: Any = None):
        if isinstance(node.expr, AssignExpr):
            try:
                self.visit(node.expr, o)
            except TypeMismatchInExpression as e:
                if getattr(e, 'expr', None) == node.expr:
                    raise TypeMismatchInStatement(node)
                raise e
        else:
            self.visit(node.expr, o)

    def visit_binary_op(self, node: "BinaryOp", o: Any = None):
        left_t = self.visit(node.left, o)
        right_t = self.visit(node.right, o)

        if isinstance(left_t, UnknownType) and not isinstance(right_t, UnknownType):
            if type(right_t) in (IntType, FloatType) and node.operator in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '!='):
                self.infer_var_type(node.left, right_t)
                left_t = right_t
        elif not isinstance(left_t, UnknownType) and isinstance(right_t, UnknownType):
            if type(left_t) in (IntType, FloatType) and node.operator in ('+', '-', '*', '/', '<', '<=', '>', '>=', '==', '!='):
                self.infer_var_type(node.right, left_t)
                right_t = left_t
        elif isinstance(left_t, UnknownType) and isinstance(right_t, UnknownType):
            raise TypeCannotBeInferred(node)

        if node.operator in ('+', '-', '*', '/'):
            if type(left_t) not in (IntType, FloatType) or type(right_t) not in (IntType, FloatType):
                raise TypeMismatchInExpression(node)
            if type(left_t) == FloatType or type(right_t) == FloatType:
                return FloatType()
            return IntType()
        elif node.operator == '%':
            if type(left_t) != IntType or type(right_t) != IntType:
                raise TypeMismatchInExpression(node)
            return IntType()
        elif node.operator in ('<', '<=', '>', '>=', '==', '!='):
            if type(left_t) not in (IntType, FloatType) or type(right_t) not in (IntType, FloatType):
                raise TypeMismatchInExpression(node)
            return IntType()
        elif node.operator in ('&&', '||'):
            if type(left_t) != IntType or type(right_t) != IntType:
                raise TypeMismatchInExpression(node)
            return IntType()

        raise TypeMismatchInExpression(node)

    def visit_prefix_op(self, node: "PrefixOp", o: Any = None):
        operand_t = self.visit(node.operand, o)
        if isinstance(operand_t, UnknownType):
            if node.operator in ('++', '--', '!'):
                self.infer_var_type(node.operand, IntType())
                operand_t = IntType()
            elif node.operator in ('+', '-'):
                raise TypeCannotBeInferred(node)

        if node.operator in ('++', '--'):
            if not isinstance(node.operand, (Identifier, MemberAccess)):
                raise TypeMismatchInExpression(node)
            if type(operand_t) != IntType:
                raise TypeMismatchInExpression(node)
            return IntType()
        elif node.operator == '!':
            if type(operand_t) != IntType:
                raise TypeMismatchInExpression(node)
            return IntType()
        elif node.operator in ('+', '-'):
            if type(operand_t) not in (IntType, FloatType):
                raise TypeMismatchInExpression(node)
            return operand_t
            
        raise TypeMismatchInExpression(node)

    def visit_postfix_op(self, node: "PostfixOp", o: Any = None):
        operand_t = self.visit(node.operand, o)
        if isinstance(operand_t, UnknownType):
            if node.operator in ('++', '--'):
                self.infer_var_type(node.operand, IntType())
                operand_t = IntType()

        if node.operator in ('++', '--'):
            if not isinstance(node.operand, (Identifier, MemberAccess)):
                raise TypeMismatchInExpression(node)
            if type(operand_t) != IntType:
                raise TypeMismatchInExpression(node)
            return IntType()

        raise TypeMismatchInExpression(node)

    def visit_assign_expr(self, node: "AssignExpr", o: Any = None):
        if not isinstance(node.lhs, (Identifier, MemberAccess)):
            raise TypeMismatchInExpression(node)
            
        left_t = self.visit(node.lhs, o)
        right_t = self.visit(node.rhs, left_t)
        
        if isinstance(left_t, UnknownType) and not isinstance(right_t, UnknownType):
            self.infer_var_type(node.lhs, right_t)
            left_t = right_t
        elif not isinstance(left_t, UnknownType) and isinstance(right_t, UnknownType):
            self.infer_var_type(node.rhs, left_t)
            right_t = left_t
        elif isinstance(left_t, UnknownType) and isinstance(right_t, UnknownType):
            raise TypeCannotBeInferred(node)
            
        if not self.is_same_type(left_t, right_t):
            raise TypeMismatchInExpression(node)
            
        return left_t

    def visit_member_access(self, node: "MemberAccess", o: Any = None):
        obj_t = self.visit(node.obj, o)
        if isinstance(obj_t, UnknownType):
            raise TypeCannotBeInferred(node)
            
        if not isinstance(obj_t, StructType):
            raise TypeMismatchInExpression(node)
            
        struct_decl = self.global_structs.get(obj_t.struct_name)
        if not struct_decl:
            raise UndeclaredStruct(obj_t.struct_name)
            
        for m in struct_decl.members:
            if m.name == node.member:
                return m.member_type
                
        raise TypeMismatchInExpression(node)

    def visit_func_call(self, node: "FuncCall", o: Any = None):
        func_name = node.name
        if func_name not in self.global_funcs:
            raise UndeclaredFunction(func_name)
            
        ret_type, param_types = self.global_funcs[func_name]
        
        if len(node.args) != len(param_types):
            raise TypeMismatchInExpression(node)
        
        for arg_expr, p_type in zip(node.args, param_types):
            arg_t = self.visit(arg_expr, p_type)
            if isinstance(arg_t, UnknownType):
                self.infer_var_type(arg_expr, p_type)
                arg_t = p_type
            if not self.is_same_type(arg_t, p_type):
                raise TypeMismatchInExpression(node)
                
        return ret_type

    def visit_identifier(self, node: "Identifier", o: Any = None):
        t = self.lookup_var(node.name)
        if t is None:
            raise UndeclaredIdentifier(node.name)
        return t

    def visit_struct_literal(self, node: "StructLiteral", o: Any = None):
        if not isinstance(o, StructType):
            if o is None:
                raise TypeCannotBeInferred(node)
        
        struct_name = o.struct_name
        if struct_name not in self.global_structs:
            raise UndeclaredStruct(struct_name)
            
        struct_decl = self.global_structs[struct_name]
        if len(node.values) != len(struct_decl.members):
            raise TypeMismatchInExpression(node)
            
        for val_expr, member_decl in zip(node.values, struct_decl.members):
            val_t = self.visit(val_expr, member_decl.member_type)
            if isinstance(val_t, UnknownType):
                self.infer_var_type(val_expr, member_decl.member_type)
                val_t = member_decl.member_type
            if not self.is_same_type(val_t, member_decl.member_type):
                raise TypeMismatchInExpression(node)
                
        return o

    def visit_int_literal(self, node: "IntLiteral", o: Any = None):
        return IntType()

    def visit_float_literal(self, node: "FloatLiteral", o: Any = None):
        return FloatType()

    def visit_string_literal(self, node: "StringLiteral", o: Any = None):
        return StringType()

    def visit_member_decl(self, node: "MemberDecl", o: Any = None):
        pass

    def visit_param(self, node: "Param", o: Any = None):
        pass

    def visit_int_type(self, node: "IntType", o: Any = None):
        return IntType()

    def visit_float_type(self, node: "FloatType", o: Any = None):
        return FloatType()

    def visit_string_type(self, node: "StringType", o: Any = None):
        return StringType()

    def visit_void_type(self, node: "VoidType", o: Any = None):
        return VoidType()

    def visit_struct_type(self, node: "StructType", o: Any = None):
        return StructType(node.struct_name)
