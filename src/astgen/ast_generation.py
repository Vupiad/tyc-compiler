"""
AST Generation module for TyC programming language.
This module contains the ASTGeneration class that converts parse trees
into Abstract Syntax Trees using the visitor pattern.
"""

from functools import reduce
from build.TyCVisitor import TyCVisitor
from build.TyCParser import TyCParser
from src.utils.nodes import *



class ASTGeneration(TyCVisitor):
    """AST Generation visitor for TyC language."""

    def _get_id_text(self, ids, index=-1):
        """Safely extract text from an ANTLR ID context which might be a list or single token."""
        if isinstance(ids, list):
            return ids[index].getText()
        return ids.getText()
        
    def _get_id_list(self, ids):
        """Ensure IDs are always treated as a list for consistent indexing."""
        if isinstance(ids, list):
            return ids
        return [ids]

    def visitProgram(self, ctx: TyCParser.ProgramContext):
        decls = [self.visit(decl) for decl in ctx.getChildren() if isinstance(decl, (TyCParser.StructDeclContext, TyCParser.FuncDeclContext))]
        return Program(decls)

    def visitStructDecl(self, ctx: TyCParser.StructDeclContext):
        name = self._get_id_text(ctx.ID())
        members = [self.visit(member) for member in ctx.memberDecl()]
        return StructDecl(name, members)

    def visitMemberDecl(self, ctx: TyCParser.MemberDeclContext):
        ids = self._get_id_list(ctx.ID())
        if ctx.primitiveType():
            member_type = self.visit(ctx.primitiveType())
            name = ids[0].getText()
        else:
            member_type = StructType(ids[0].getText())
            name = ids[1].getText()
        return MemberDecl(member_type, name)

    def visitFuncDecl(self, ctx: TyCParser.FuncDeclContext):
        return_type = self.visit(ctx.type_()) if ctx.type_() else (VoidType() if ctx.VOID() else None)
        name = self._get_id_text(ctx.ID(), -1)  # Function name is always the last ID
        params = self.visit(ctx.paramList()) if ctx.paramList() else []
        body = self.visit(ctx.block())
        return FuncDecl(return_type, name, params, body)

    def visitParamList(self, ctx: TyCParser.ParamListContext):
        return [self.visit(param) for param in ctx.param()]

    def visitParam(self, ctx: TyCParser.ParamContext):
        ids = self._get_id_list(ctx.ID())
        if ctx.primitiveType():
            param_type = self.visit(ctx.primitiveType())
            name = ids[0].getText()
        else:
            param_type = StructType(ids[0].getText())
            name = ids[1].getText()
        return Param(param_type, name)

    def visitStatement(self, ctx: TyCParser.StatementContext):
        return self.visit(ctx.getChild(0))

    def visitVarDecl(self, ctx: TyCParser.VarDeclContext):
        var_type = self.visit(ctx.type_()) if ctx.type_() else None
        name = self._get_id_text(ctx.ID(), -1)
        init_value = self.visit(ctx.expr()) if ctx.expr() else None
        return VarDecl(var_type, name, init_value)

    def visitBlock(self, ctx: TyCParser.BlockContext):
        statements = [self.visit(stmt) for stmt in ctx.getChildren() if isinstance(stmt, (TyCParser.VarDeclContext, TyCParser.StatementContext))]
        return BlockStmt(statements)

    def visitIfStmt(self, ctx: TyCParser.IfStmtContext):
        condition = self.visit(ctx.expr())
        then_stmt = self.visit(ctx.statement(0))
        else_stmt = self.visit(ctx.statement(1)) if ctx.statement(1) else None
        return IfStmt(condition, then_stmt, else_stmt)

    def visitWhileStmt(self, ctx: TyCParser.WhileStmtContext):
        condition = self.visit(ctx.expr())
        body = self.visit(ctx.statement())
        return WhileStmt(condition, body)

    def visitForStmt(self, ctx: TyCParser.ForStmtContext):
        # 1. Initialization part
        init_ctx = ctx.getChild(2)
        if isinstance(init_ctx, (TyCParser.VarDeclContext, TyCParser.ExprStmtContext)):
            init = self.visit(init_ctx)
        else:
            init = None
            
        condition = None
        update = None
        
        # 2. Iterate through children to safely separate condition and update by looking for the SEMI
        passed_semi = False
        for i in range(3, ctx.getChildCount()):
            child = ctx.getChild(i)
            if child.getText() == ';':
                passed_semi = True
            elif isinstance(child, TyCParser.ExprContext):
                if not passed_semi:
                    condition = self.visit(child)
                else:
                    update = self.visit(child)
                    
        body = self.visit(ctx.statement())
        return ForStmt(init, condition, update, body)

    def visitSwitchStmt(self, ctx: TyCParser.SwitchStmtContext):
        expr = self.visit(ctx.expr())
        cases = [self.visit(case) for case in ctx.caseStmt()]
        default_case = self.visit(ctx.defaultStmt(0)) if ctx.defaultStmt() else None
        return SwitchStmt(expr, cases, default_case)

    def visitCaseStmt(self, ctx: TyCParser.CaseStmtContext):
        expr = self.visit(ctx.expr())
        statements = [self.visit(stmt) for stmt in ctx.getChildren() if isinstance(stmt, (TyCParser.VarDeclContext, TyCParser.StatementContext))]
        return CaseStmt(expr, statements)

    def visitDefaultStmt(self, ctx: TyCParser.DefaultStmtContext):
        statements = [self.visit(stmt) for stmt in ctx.getChildren() if isinstance(stmt, (TyCParser.VarDeclContext, TyCParser.StatementContext))]
        return DefaultStmt(statements)

    def visitBreakStmt(self, ctx: TyCParser.BreakStmtContext):
        return BreakStmt()

    def visitContinueStmt(self, ctx: TyCParser.ContinueStmtContext):
        return ContinueStmt()

    def visitReturnStmt(self, ctx: TyCParser.ReturnStmtContext):
        expr = self.visit(ctx.expr()) if ctx.expr() else None
        return ReturnStmt(expr)

    def visitExprStmt(self, ctx: TyCParser.ExprStmtContext):
        return ExprStmt(self.visit(ctx.expr()))
        
    def visitType(self, ctx: TyCParser.TypeContext):
        if ctx.primitiveType():
            return self.visit(ctx.primitiveType())
        return StructType(self._get_id_text(ctx.ID()))

    def visitPrimitiveType(self, ctx: TyCParser.PrimitiveTypeContext):
        if ctx.INT(): return IntType()
        if ctx.FLOAT(): return FloatType()
        if ctx.STRING(): return StringType()

    def visitAssignExpr(self, ctx: TyCParser.AssignExprContext):
        lhs = self.visit(ctx.expr(0))
        rhs = self.visit(ctx.expr(1))
        return AssignExpr(lhs, rhs)

    def visitBinaryExpr(self, ctx: TyCParser.BinaryExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        return BinaryOp(left, op, right)

    def visitUnaryExpr(self, ctx: TyCParser.UnaryExprContext):
        op = ctx.getChild(0).getText()
        operand = self.visit(ctx.expr())
        return PrefixOp(op, operand)

    def visitPostfixExpr(self, ctx: TyCParser.PostfixExprContext):
        if ctx.LPAREN():
            base_expr = self.visit(ctx.expr())
            name = base_expr.name if isinstance(base_expr, Identifier) else str(base_expr)
            args = self.visit(ctx.argList()) if ctx.argList() else []
            return FuncCall(name, args)
        
        op = ctx.getChild(1).getText()
        operand = self.visit(ctx.expr())
        return PostfixOp(op, operand)

    def visitMemberAccess(self, ctx: TyCParser.MemberAccessContext):
        obj = self.visit(ctx.expr())
        member = self._get_id_text(ctx.ID())
        return MemberAccess(obj, member)

    def visitPrimaryExpr(self, ctx: TyCParser.PrimaryExprContext):
        return self.visit(ctx.primary())

    def visitArgList(self, ctx: TyCParser.ArgListContext):
        return [self.visit(expr) for expr in ctx.expr()]
        
    def visitPrimary(self, ctx: TyCParser.PrimaryContext):
        if ctx.ID():
            return Identifier(self._get_id_text(ctx.ID()))
        if ctx.INT_LIT():
            return IntLiteral(int(ctx.INT_LIT().getText()))
        if ctx.FLOAT_LIT():
            return FloatLiteral(float(ctx.FLOAT_LIT().getText()))
        if ctx.STRING_LIT():
            return StringLiteral(ctx.STRING_LIT().getText())
        if ctx.structLiteral():
            return self.visit(ctx.structLiteral())
        if ctx.expr():
            return self.visit(ctx.expr())

    def visitStructLiteral(self, ctx: TyCParser.StructLiteralContext):
        values = self.visit(ctx.argList()) if ctx.argList() else []
        return StructLiteral(values)
