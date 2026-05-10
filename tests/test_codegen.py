"""
Assignment 4 code generation tests.
100 executable output-verification test cases.
"""

import pytest

from src.utils.nodes import *
from tests.utils import CodeGenerator


def _program_with_main(statements, extra_decls=None):
    decls = list(extra_decls or [])
    decls.append(FuncDecl(VoidType(), "main", [], BlockStmt(statements)))
    return Program(decls)


def _add_case(cases, ast, expected, input_data=""):
    case_id = f"{len(cases) + 1:03d}"
    cases.append((case_id, ast, expected, input_data))


def _build_cases():
    cases = []

    # ------------------------------------------------------------------
    # 1) Basic I/O and literals (6)
    # ------------------------------------------------------------------
    _add_case(
        cases,
        _program_with_main([ExprStmt(FuncCall("printString", [StringLiteral("Hello World")]))]),
        "Hello World",
    )
    _add_case(
        cases,
        _program_with_main([ExprStmt(FuncCall("printInt", [IntLiteral(42)]))]),
        "42",
    )
    _add_case(
        cases,
        _program_with_main([ExprStmt(FuncCall("printFloat", [FloatLiteral(3.5)]))]),
        "3.5",
    )
    _add_case(
        cases,
        _program_with_main(
            [ExprStmt(FuncCall("printInt", [FuncCall("readInt", [])]))]
        ),
        "7",
        "7\n",
    )
    _add_case(
        cases,
        _program_with_main(
            [ExprStmt(FuncCall("printFloat", [FuncCall("readFloat", [])]))]
        ),
        "2.5",
        "2.5\n",
    )
    _add_case(
        cases,
        _program_with_main(
            [ExprStmt(FuncCall("printString", [FuncCall("readString", [])]))]
        ),
        "token",
        "token\n",
    )

    # ------------------------------------------------------------------
    # 2) Arithmetic / comparison (24 => cumulative 30)
    # ------------------------------------------------------------------
    for i in range(1, 13):
        ast = _program_with_main(
            [
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [BinaryOp(IntLiteral(i), "+", IntLiteral(i + 1))],
                    )
                )
            ]
        )
        _add_case(cases, ast, str(i + i + 1))

    for i in range(1, 7):
        ast = _program_with_main(
            [
                ExprStmt(
                    FuncCall(
                        "printFloat",
                        [BinaryOp(FloatLiteral(i + 0.5), "+", IntLiteral(i))],
                    )
                )
            ]
        )
        _add_case(cases, ast, str((i + 0.5) + i))

    for i in range(1, 7):
        left = i * 3 + 1
        right = i + 1
        ast = _program_with_main(
            [
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [BinaryOp(BinaryOp(IntLiteral(left), "%", IntLiteral(right)), ">", IntLiteral(0))],
                    )
                )
            ]
        )
        expected = 1 if (left % right) > 0 else 0
        _add_case(cases, ast, str(expected))

    # ------------------------------------------------------------------
    # 3) Control flow / increments (20 => cumulative 50)
    # ------------------------------------------------------------------
    for i in range(5):
        cond = IntLiteral(1 if i % 2 == 0 else 0)
        ast = _program_with_main(
            [
                IfStmt(
                    cond,
                    ExprStmt(FuncCall("printInt", [IntLiteral(11 + i)])),
                    ExprStmt(FuncCall("printInt", [IntLiteral(21 + i)])),
                )
            ]
        )
        expected = str(11 + i) if i % 2 == 0 else str(21 + i)
        _add_case(cases, ast, expected)

    for n in range(1, 6):
        # sum 1..n
        ast = _program_with_main(
            [
                VarDecl(IntType(), "i", IntLiteral(1)),
                VarDecl(IntType(), "s", IntLiteral(0)),
                WhileStmt(
                    BinaryOp(Identifier("i"), "<=", IntLiteral(n)),
                    BlockStmt(
                        [
                            ExprStmt(
                                AssignExpr(
                                    Identifier("s"),
                                    BinaryOp(Identifier("s"), "+", Identifier("i")),
                                )
                            ),
                            ExprStmt(
                                AssignExpr(
                                    Identifier("i"),
                                    BinaryOp(Identifier("i"), "+", IntLiteral(1)),
                                )
                            ),
                        ]
                    ),
                ),
                ExprStmt(FuncCall("printInt", [Identifier("s")])),
            ]
        )
        _add_case(cases, ast, str(n * (n + 1) // 2))

    for n in range(3, 8):
        # sum odd numbers < n using continue
        total = sum(x for x in range(n) if x % 2 == 1)
        ast = _program_with_main(
            [
                VarDecl(IntType(), "s", IntLiteral(0)),
                ForStmt(
                    VarDecl(IntType(), "i", IntLiteral(0)),
                    BinaryOp(Identifier("i"), "<", IntLiteral(n)),
                    PrefixOp("++", Identifier("i")),
                    BlockStmt(
                        [
                            IfStmt(
                                BinaryOp(
                                    BinaryOp(Identifier("i"), "%", IntLiteral(2)),
                                    "==",
                                    IntLiteral(0),
                                ),
                                ContinueStmt(),
                            ),
                            ExprStmt(
                                AssignExpr(
                                    Identifier("s"),
                                    BinaryOp(Identifier("s"), "+", Identifier("i")),
                                )
                            ),
                        ]
                    ),
                ),
                ExprStmt(FuncCall("printInt", [Identifier("s")])),
            ]
        )
        _add_case(cases, ast, str(total))

    # Prefix/postfix on identifiers
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(1)),
                ExprStmt(FuncCall("printInt", [PrefixOp("++", Identifier("x"))])),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "22",
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(1)),
                ExprStmt(FuncCall("printInt", [PostfixOp("++", Identifier("x"))])),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "12",
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(3)),
                ExprStmt(FuncCall("printInt", [PrefixOp("--", Identifier("x"))])),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "22",
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(3)),
                ExprStmt(FuncCall("printInt", [PostfixOp("--", Identifier("x"))])),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "32",
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(1)),
                ExprStmt(FuncCall("printInt", [BinaryOp(PrefixOp("++", Identifier("x")), "+", IntLiteral(5))])),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "72",
    )

    # ------------------------------------------------------------------
    # 4) Functions / assignment expr / short-circuit (20 => cumulative 70)
    # ------------------------------------------------------------------
    for i in range(1, 6):
        add_fn = FuncDecl(
            IntType(),
            "add",
            [Param(IntType(), "a"), Param(IntType(), "b")],
            BlockStmt([ReturnStmt(BinaryOp(Identifier("a"), "+", Identifier("b")))]),
        )
        ast = _program_with_main(
            [ExprStmt(FuncCall("printInt", [FuncCall("add", [IntLiteral(i), IntLiteral(i + 2)])]))],
            [add_fn],
        )
        _add_case(cases, ast, str(i + i + 2))

    for n in [3, 4, 5]:
        fact_fn = FuncDecl(
            IntType(),
            "fact",
            [Param(IntType(), "n")],
            BlockStmt(
                [
                    IfStmt(
                        BinaryOp(Identifier("n"), "<=", IntLiteral(1)),
                        ReturnStmt(IntLiteral(1)),
                    ),
                    ReturnStmt(
                        BinaryOp(
                            Identifier("n"),
                            "*",
                            FuncCall("fact", [BinaryOp(Identifier("n"), "-", IntLiteral(1))]),
                        )
                    ),
                ]
            ),
        )
        ast = _program_with_main(
            [ExprStmt(FuncCall("printInt", [FuncCall("fact", [IntLiteral(n)])]))],
            [fact_fn],
        )
        expected = "6" if n == 3 else ("24" if n == 4 else "120")
        _add_case(cases, ast, expected)

    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x"),
                VarDecl(IntType(), "y"),
                VarDecl(IntType(), "z"),
                ExprStmt(
                    AssignExpr(
                        Identifier("x"),
                        AssignExpr(Identifier("y"), AssignExpr(Identifier("z"), IntLiteral(7))),
                    )
                ),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
                ExprStmt(FuncCall("printInt", [Identifier("y")])),
                ExprStmt(FuncCall("printInt", [Identifier("z")])),
            ]
        ),
        "777",
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(1)),
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [BinaryOp(AssignExpr(Identifier("x"), IntLiteral(5)), "+", IntLiteral(2))],
                    )
                ),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        ),
        "75",
    )

    # 5 cases for &&, 5 cases for ||
    for i in range(5):
        left = 0 if i % 2 == 0 else 1
        assigned = i + 1
        expected = 0 if left == 0 else assigned
        ast = _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(0)),
                ExprStmt(
                    BinaryOp(
                        IntLiteral(left),
                        "&&",
                        AssignExpr(Identifier("x"), IntLiteral(assigned)),
                    )
                ),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        )
        _add_case(cases, ast, str(expected))

    for i in range(5):
        left = 1 if i % 2 == 0 else 0
        assigned = i + 2
        expected = 0 if left == 1 else assigned
        ast = _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(0)),
                ExprStmt(
                    BinaryOp(
                        IntLiteral(left),
                        "||",
                        AssignExpr(Identifier("x"), IntLiteral(assigned)),
                    )
                ),
                ExprStmt(FuncCall("printInt", [Identifier("x")])),
            ]
        )
        _add_case(cases, ast, str(expected))

    # ------------------------------------------------------------------
    # 5) Switch / case / fall-through (10 => cumulative 80)
    # ------------------------------------------------------------------
    def _switch_program(x):
        return _program_with_main(
            [
                VarDecl(IntType(), "x", IntLiteral(x)),
                SwitchStmt(
                    Identifier("x"),
                    [
                        CaseStmt(
                            IntLiteral(1),
                            [
                                ExprStmt(FuncCall("printInt", [IntLiteral(1)])),
                                BreakStmt(),
                            ],
                        ),
                        CaseStmt(
                            IntLiteral(2),
                            [ExprStmt(FuncCall("printInt", [IntLiteral(2)]))],
                        ),
                        CaseStmt(
                            IntLiteral(3),
                            [
                                ExprStmt(FuncCall("printInt", [IntLiteral(3)])),
                                BreakStmt(),
                            ],
                        ),
                    ],
                    DefaultStmt([ExprStmt(FuncCall("printInt", [IntLiteral(9)]))]),
                ),
            ]
        )

    for x in [1, 2, 3, 4, 2, 1, 5, 3, 2, 0]:
        if x == 1:
            expected = "1"
        elif x == 2:
            expected = "23"
        elif x == 3:
            expected = "3"
        else:
            expected = "9"
        _add_case(cases, _switch_program(x), expected)

    # ------------------------------------------------------------------
    # 6) Structs (20 => cumulative 100)
    # ------------------------------------------------------------------
    point_decl = StructDecl("Point", [MemberDecl(IntType(), "x"), MemberDecl(IntType(), "y")])

    # 5 cases: struct literal + read
    for i in range(1, 6):
        ast = _program_with_main(
            [
                VarDecl(StructType("Point"), "p", StructLiteral([IntLiteral(i), IntLiteral(i + 1)])),
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [BinaryOp(MemberAccess(Identifier("p"), "x"), "+", MemberAccess(Identifier("p"), "y"))],
                    )
                ),
            ],
            [point_decl],
        )
        _add_case(cases, ast, str(i + i + 1))

    # 5 cases: assignment to members
    for i in range(1, 6):
        ast = _program_with_main(
            [
                VarDecl(StructType("Point"), "p"),
                ExprStmt(AssignExpr(MemberAccess(Identifier("p"), "x"), IntLiteral(i))),
                ExprStmt(AssignExpr(MemberAccess(Identifier("p"), "y"), IntLiteral(i + 1))),
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [BinaryOp(MemberAccess(Identifier("p"), "y"), "-", MemberAccess(Identifier("p"), "x"))],
                    )
                ),
            ],
            [point_decl],
        )
        _add_case(cases, ast, "1")

    # 5 cases: struct copy semantics (value copy, not alias)
    for i in range(1, 6):
        ast = _program_with_main(
            [
                VarDecl(StructType("Point"), "a", StructLiteral([IntLiteral(i), IntLiteral(i + 1)])),
                VarDecl(StructType("Point"), "b"),
                ExprStmt(AssignExpr(Identifier("b"), Identifier("a"))),
                ExprStmt(
                    AssignExpr(
                        MemberAccess(Identifier("b"), "x"),
                        BinaryOp(MemberAccess(Identifier("b"), "x"), "+", IntLiteral(10)),
                    )
                ),
                ExprStmt(FuncCall("printInt", [MemberAccess(Identifier("a"), "x")])),
                ExprStmt(FuncCall("printInt", [MemberAccess(Identifier("b"), "x")])),
            ],
            [point_decl],
        )
        _add_case(cases, ast, f"{i}{i + 10}")

    # 3 cases: nested struct literal/member access
    inner_decl = StructDecl("Inner", [MemberDecl(IntType(), "v")])
    outer_decl = StructDecl("Outer", [MemberDecl(StructType("Inner"), "i"), MemberDecl(IntType(), "k")])
    for i in range(1, 4):
        ast = _program_with_main(
            [
                VarDecl(
                    StructType("Outer"),
                    "o",
                    StructLiteral([StructLiteral([IntLiteral(i)]), IntLiteral(i + 2)]),
                ),
                ExprStmt(
                    FuncCall(
                        "printInt",
                        [
                            BinaryOp(
                                MemberAccess(MemberAccess(Identifier("o"), "i"), "v"),
                                "+",
                                MemberAccess(Identifier("o"), "k"),
                            )
                        ],
                    )
                ),
            ],
            [inner_decl, outer_decl],
        )
        _add_case(cases, ast, str(2 * i + 2))

    # 2 cases: struct in function argument / return
    sum_point = FuncDecl(
        IntType(),
        "sumPoint",
        [Param(StructType("Point"), "p")],
        BlockStmt(
            [
                ReturnStmt(
                    BinaryOp(
                        MemberAccess(Identifier("p"), "x"),
                        "+",
                        MemberAccess(Identifier("p"), "y"),
                    )
                )
            ]
        ),
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(StructType("Point"), "p", StructLiteral([IntLiteral(2), IntLiteral(3)])),
                ExprStmt(FuncCall("printInt", [FuncCall("sumPoint", [Identifier("p")])])),
            ],
            [point_decl, sum_point],
        ),
        "5",
    )

    make_point = FuncDecl(
        StructType("Point"),
        "makePoint",
        [Param(IntType(), "v")],
        BlockStmt(
            [
                VarDecl(
                    StructType("Point"),
                    "p",
                    StructLiteral([Identifier("v"), BinaryOp(Identifier("v"), "+", IntLiteral(1))]),
                ),
                ReturnStmt(Identifier("p")),
            ]
        ),
    )
    _add_case(
        cases,
        _program_with_main(
            [
                VarDecl(StructType("Point"), "q", FuncCall("makePoint", [IntLiteral(4)])),
                ExprStmt(FuncCall("printInt", [MemberAccess(Identifier("q"), "y")])),
            ],
            [point_decl, make_point],
        ),
        "5",
    )

    assert len(cases) == 100
    return cases


CASES = _build_cases()


@pytest.mark.parametrize(
    "case_id,ast,expected,input_data",
    CASES,
    ids=[f"case_{case_id}" for case_id, _, _, _ in CASES],
)
def test_codegen(case_id, ast, expected, input_data):
    result = CodeGenerator().generate_and_run(ast, input_data)
    assert result == expected, f"[{case_id}] Expected '{expected}', got '{result}'"
