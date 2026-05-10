import os

HEADER = '''"""
Test cases for TyC Static Semantic Checker

This module contains test cases for the static semantic checker.
100 test cases covering all error types and comprehensive scenarios.
"""

from tests.utils import Checker
from src.utils.nodes import (
    Program, FuncDecl, BlockStmt, VarDecl, AssignExpr, ExprStmt,
    IntType, FloatType, StringType, VoidType, StructType,
    IntLiteral, FloatLiteral, StringLiteral, Identifier,
    BinaryOp, MemberAccess, FuncCall, StructDecl, MemberDecl,
    Param, ReturnStmt,
)

'''

def gen_test(num, name, source, expected):
    return f'''
def test_{num:03d}():
    """{name}"""
    source = """{source}"""
    expected = "{expected}"
    assert expected in Checker(source).check_from_source(), f"Expected '{{expected}}' but got '{{Checker(source).check_from_source()}}'"
'''

tests = []

# ==========================================
# Valid Programs (001 - 015)
# ==========================================
tests.append((1, "Test a valid program that should pass all checks", """
void main() {
    int x = 5;
    int y = x + 1;
}
""", "Static checking passed"))

tests.append((2, "Test valid program with auto type inference", """
void main() {
    auto x = 10;
    auto y = 3.14;
    auto z = x + y;
}
""", "Static checking passed"))

tests.append((3, "Test valid program with functions", """
int add(int x, int y) {
    return x + y;
}
void main() {
    int sum = add(5, 3);
}
""", "Static checking passed"))

tests.append((4, "Test valid program with struct", """
struct Point {
    int x;
    int y;
};
void main() {
    Point p;
    p.x = 10;
    p.y = 20;
}
""", "Static checking passed"))

tests.append((5, "Test valid program with nested blocks", """
void main() {
    int x = 10;
    {
        int y = 20;
        int z = x + y;
    }
}
""", "Static checking passed"))

tests.append((6, "Test valid program with shadowed variables", """
void main() {
    int x = 10;
    {
        int x = 20;
        {
            int x = 30;
        }
    }
}
""", "Static checking passed"))

tests.append((7, "Test valid program with struct containing another struct", """
struct Point { int x; int y; };
struct Rect { Point p1; Point p2; };
void main() {
    Rect r;
    r.p1.x = 0;
}
""", "Static checking passed"))

tests.append((8, "Test valid function calls including built-ins", """
void main() {
    int x = readInt();
    printInt(x);
}
""", "Static checking passed"))

tests.append((9, "Test valid for loop with break and continue", """
void main() {
    for (int i = 0; i < 10; ++i) {
        if (i == 5) break;
        if (i == 2) continue;
    }
}
""", "Static checking passed"))

tests.append((10, "Test valid while loop with break and continue", """
void main() {
    int i = 0;
    while (i < 10) {
        if (i == 5) break;
        if (i == 2) continue;
        ++i;
    }
}
""", "Static checking passed"))

tests.append((11, "Test valid switch statement", """
void main() {
    int x = 2;
    switch(x) {
        case 1: x = 10; break;
        case 2: x = 20; break;
    }
}
""", "Static checking passed"))

tests.append((12, "Test valid auto inference with literals and functions", """
int getVal() { return 42; }
void main() {
    auto a = getVal();
    auto b;
    b = a + 5;
}
""", "Static checking passed"))

tests.append((13, "Test valid relational and logical operators", """
void main() {
    int a = 5;
    float b = 6.0;
    int c = a < b;
    int d = c && (a == 5);
}
""", "Static checking passed"))

tests.append((14, "Test valid assignment expressions", """
void main() {
    int x;
    int y = (x = 5) + 3;
    int a; int b; int c;
    a = b = c = 10;
}
""", "Static checking passed"))

tests.append((15, "Test valid struct literal initialization", """
struct Point { int x; int y; };
void main() {
    Point p1 = {10, 20};
    Point p2;
    p2 = p1;
}
""", "Static checking passed"))

# ==========================================
# Redeclared (016 - 025)
# ==========================================
tests.append((16, "Redeclared Struct", """
struct Point { int x; };
struct Point { int y; };
void main() {}
""", "Redeclared(Struct, Point)"))

tests.append((17, "Redeclared Function", """
int add() { return 1; }
int add(int x) { return x; }
void main() {}
""", "Redeclared(Function, add)"))

tests.append((18, "Redeclared Variable in same block", """
void main() {
    int x = 5;
    int x = 10;
}
""", "Redeclared(Variable, x)"))

tests.append((19, "Redeclared Parameter", """
int calculate(int x, int x) {
    return x;
}
void main() {}
""", "Redeclared(Parameter, x)"))

tests.append((20, "Local variable reuses parameter name", """
void func(int x) {
    int x = 10;
}
void main() {}
""", "Redeclared(Variable, x)"))

tests.append((21, "Duplicate struct member", """
struct Point {
    int x;
    int x;
};
void main() {}
""", "Redeclared(Member, x)"))

tests.append((22, "Multiple variable redeclarations", """
void main() {
    int a = 1;
    float b = 2.0;
    string a = "test";
}
""", "Redeclared(Variable, a)"))

tests.append((23, "Local variable reuses parameter name in nested block", """
void func(int p) {
    {
        int p = 5;
    }
}
void main() {}
""", "Redeclared(Variable, p)"))

tests.append((24, "Struct member duplicate with different type", """
struct Test {
    int val;
    float val;
};
void main() {}
""", "Redeclared(Member, val)"))

tests.append((25, "Function parameter duplicate with different type", """
void test(int a, float a) {}
void main() {}
""", "Redeclared(Parameter, a)"))

# ==========================================
# UndeclaredIdentifier (026 - 035)
# ==========================================
tests.append((26, "Undeclared Variable", """
void main() {
    int x = y + 1;
}
""", "UndeclaredIdentifier(y)"))

tests.append((27, "Variable used before declaration", """
void main() {
    x = 5;
    int x;
}
""", "UndeclaredIdentifier(x)"))

tests.append((28, "Out of scope access", """
void main() {
    { int x = 5; }
    int y = x;
}
""", "UndeclaredIdentifier(x)"))

tests.append((29, "Variable used in its own initializer", """
void main() {
    int x = x + 1;
}
""", "UndeclaredIdentifier(x)"))

tests.append((30, "Undeclared in expression", """
void main() {
    int x = 5 + z * 2;
}
""", "UndeclaredIdentifier(z)"))

tests.append((31, "Undeclared in function call", """
void foo(int a) {}
void main() {
    foo(undeclared);
}
""", "UndeclaredIdentifier(undeclared)"))

tests.append((32, "Undeclared as condition", """
void main() {
    if (cond) {}
}
""", "UndeclaredIdentifier(cond)"))

tests.append((33, "Undeclared in return", """
int foo() {
    return retVal;
}
void main() {}
""", "UndeclaredIdentifier(retVal)"))

tests.append((34, "Out of scope access in another function", """
void func1() { int a = 1; }
void func2() { int b = a; }
void main() {}
""", "UndeclaredIdentifier(a)"))

tests.append((35, "Parameter used outside function", """
void func1(int p) {}
void func2() { int x = p; }
void main() {}
""", "UndeclaredIdentifier(p)"))

# ==========================================
# UndeclaredFunction (036 - 045)
# ==========================================
tests.append((36, "Call undeclared function", """
void main() {
    foo();
}
""", "UndeclaredFunction(foo)"))

tests.append((37, "Call function before declaration", """
void main() {
    bar();
}
void bar() {}
""", "UndeclaredFunction(bar)"))

tests.append((38, "Undeclared function in assignment", """
void main() {
    int x = getVal();
}
""", "UndeclaredFunction(getVal)"))

tests.append((39, "Undeclared function in expression", """
void main() {
    int x = 5 + calc(2);
}
""", "UndeclaredFunction(calc)"))

tests.append((40, "Undeclared function with arguments", """
void main() {
    process(1, 2.0, "3");
}
""", "UndeclaredFunction(process)"))

tests.append((41, "Undeclared function in return", """
int foo() {
    return getMissing();
}
void main() {}
""", "UndeclaredFunction(getMissing)"))

tests.append((42, "Undeclared function in condition", """
void main() {
    if (check()) {}
}
""", "UndeclaredFunction(check)"))

tests.append((43, "Undeclared function nested call", """
void main() {
    printInt(missing());
}
""", "UndeclaredFunction(missing)"))

tests.append((44, "Call variable as function", """
void main() {
    int x = 5;
    x();
}
""", "UndeclaredFunction(x)"))

tests.append((45, "Call parameter as function", """
void test(int p) {
    p();
}
void main() {}
""", "UndeclaredFunction(p)"))

# ==========================================
# UndeclaredStruct (046 - 055)
# ==========================================
tests.append((46, "Variable of undeclared struct", """
void main() {
    Point p;
}
""", "UndeclaredStruct(Point)"))

tests.append((47, "Struct type used before declaration", """
void test() {
    Data d;
}
struct Data { int x; };
void main() {}
""", "UndeclaredStruct(Data)"))

tests.append((48, "Struct member using undeclared struct", """
struct Node {
    Missing m;
};
void main() {}
""", "UndeclaredStruct(Missing)"))

tests.append((49, "Undeclared struct in parameter", """
void process(Unknown u) {}
void main() {}
""", "UndeclaredStruct(Unknown)"))

tests.append((50, "Undeclared struct as return type", """
Result getRes() {}
void main() {}
""", "UndeclaredStruct(Result)"))

tests.append((51, "Undeclared struct in multiple declarations", """
void main() {
    int x;
    BadStruct b;
}
""", "UndeclaredStruct(BadStruct)"))

tests.append((52, "Struct inheriting undeclared struct (member)", """
struct A { B b; };
void main() {}
""", "UndeclaredStruct(B)"))

tests.append((53, "Undeclared struct in function prototype", """
void doWork(Data d) {}
void main() {}
""", "UndeclaredStruct(Data)"))

tests.append((54, "Struct literal with undeclared struct", """
void main() {
    Missing m = {1, 2};
}
""", "UndeclaredStruct(Missing)"))

tests.append((55, "Undeclared struct in nested scope", """
void main() {
    {
        InnerStruct s;
    }
}
""", "UndeclaredStruct(InnerStruct)"))

# ==========================================
# TypeCannotBeInferred (056 - 065)
# ==========================================
tests.append((56, "Neither auto has known type", """
void main() {
    auto x;
    auto y;
    auto z = x + y;
}
""", "TypeCannotBeInferred"))

tests.append((57, "Auto without init never used", """
void main() {
    auto x;
}
""", "TypeCannotBeInferred"))

tests.append((58, "Auto mutually dependent assignment", """
void main() {
    auto a;
    auto b;
    a = b;
}
""", "TypeCannotBeInferred"))

tests.append((59, "Auto relational operator", """
void main() {
    auto x;
    auto y;
    int z = x < y;
}
""", "TypeCannotBeInferred"))

tests.append((60, "Auto string mix", """
void main() {
    auto x;
    auto y = x + "hello";
}
""", "TypeCannotBeInferred"))

tests.append((61, "Auto returned but unknown", """
func() {
    auto x;
    return x;
}
void main() {}
""", "TypeCannotBeInferred"))

tests.append((62, "Auto passed to printInt but no type", """
void main() {
    auto a;
    auto b;
    int c = a * b;
}
""", "TypeCannotBeInferred"))

tests.append((63, "Auto assignment loop", """
void main() {
    auto a;
    a = a;
}
""", "TypeCannotBeInferred"))

tests.append((64, "Auto unused in nested block", """
void main() {
    {
        auto val;
    }
}
""", "TypeCannotBeInferred"))

tests.append((65, "Auto mixed with float", """
void main() {
    auto x;
    auto y;
    float f = x + y;
}
""", "TypeCannotBeInferred"))

# ==========================================
# TypeMismatchInStatement (066 - 078)
# ==========================================
tests.append((66, "If condition float", """
void main() {
    if (3.14) {}
}
""", "TypeMismatchInStatement"))

tests.append((67, "If condition string", """
void main() {
    if ("true") {}
}
""", "TypeMismatchInStatement"))

tests.append((68, "If condition struct", """
struct S { int x; };
void main() {
    S s;
    if (s) {}
}
""", "TypeMismatchInStatement"))

tests.append((69, "While condition string", """
void main() {
    while ("loop") {}
}
""", "TypeMismatchInStatement"))

tests.append((70, "For condition float", """
void main() {
    for (int i = 0; 1.5; ++i) {}
}
""", "TypeMismatchInStatement"))

tests.append((71, "Assignment different types", """
void main() {
    int x;
    x = 3.14;
}
""", "TypeMismatchInStatement"))

tests.append((72, "Assignment different struct types", """
struct A { int x; };
struct B { int x; };
void main() {
    A a; B b;
    a = b;
}
""", "TypeMismatchInStatement"))

tests.append((73, "Return string in int func", """
int foo() {
    return "text";
}
void main() {}
""", "TypeMismatchInStatement"))

tests.append((74, "Return void in int func", """
int foo() {
    return;
}
void main() {}
""", "TypeMismatchInStatement"))

tests.append((75, "Return int in void func", """
void main() {
    return 1;
}
""", "TypeMismatchInStatement"))

tests.append((76, "Switch condition float", """
void main() {
    switch (3.14) { case 1: break; }
}
""", "TypeMismatchInStatement"))

tests.append((77, "Switch condition string", """
void main() {
    switch ("str") { case 1: break; }
}
""", "TypeMismatchInStatement"))

tests.append((78, "Switch condition struct", """
struct S { int x; };
void main() {
    S s;
    switch (s) { case 1: break; }
}
""", "TypeMismatchInStatement"))

# ==========================================
# TypeMismatchInExpression (079 - 092)
# ==========================================
tests.append((79, "Int + string", """
void main() {
    int x = 5 + "text";
}
""", "TypeMismatchInExpression"))

tests.append((80, "Int + struct", """
struct S { int x; };
void main() {
    S s;
    int x = 5 + s;
}
""", "TypeMismatchInExpression"))

tests.append((81, "Float % int", """
void main() {
    int x = 5.5 % 2;
}
""", "TypeMismatchInExpression"))

tests.append((82, "Float < string", """
void main() {
    int b = 1.0 < "1";
}
""", "TypeMismatchInExpression"))

tests.append((83, "Struct == struct", """
struct S { int x; };
void main() {
    S s1; S s2;
    int b = s1 == s2;
}
""", "TypeMismatchInExpression"))

tests.append((84, "Float && int", """
void main() {
    int b = 1.5 && 1;
}
""", "TypeMismatchInExpression"))

tests.append((85, "Logical NOT on float", """
void main() {
    int b = !1.5;
}
""", "TypeMismatchInExpression"))

tests.append((86, "Increment float", """
void main() {
    float f = 1.0;
    ++f;
}
""", "TypeMismatchInExpression"))

tests.append((87, "Increment literal", """
void main() {
    ++5;
}
""", "TypeMismatchInExpression"))

tests.append((88, "Increment expression", """
void main() {
    int x = 1;
    (x + 1)++;
}
""", "TypeMismatchInExpression"))

tests.append((89, "Member access on non-struct", """
void main() {
    int x = 1;
    int y = x.mem;
}
""", "TypeMismatchInExpression"))

tests.append((90, "Non-existent member access", """
struct S { int x; };
void main() {
    S s;
    int y = s.y;
}
""", "TypeMismatchInExpression"))

tests.append((91, "Call wrong arg type", """
void foo(int a) {}
void main() {
    foo("123");
}
""", "TypeMismatchInExpression"))

tests.append((92, "Call wrong arg count", """
void foo(int a) {}
void main() {
    foo(1, 2);
}
""", "TypeMismatchInExpression"))

# ==========================================
# MustInLoop (093 - 100)
# ==========================================
tests.append((93, "Assignment expression invalid LHS", """
void main() {
    int x = (5 = 2) + 1;
}
""", "TypeMismatchInExpression"))

tests.append((94, "Break outside loop", """
void main() {
    break;
}
""", "MustInLoop"))

tests.append((95, "Continue outside loop", """
void main() {
    continue;
}
""", "MustInLoop"))

tests.append((96, "Break in if", """
void main() {
    if (1) { break; }
}
""", "MustInLoop"))

tests.append((97, "Continue in if", """
void main() {
    if (1) { continue; }
}
""", "MustInLoop"))

tests.append((98, "Continue in switch", """
void main() {
    int x = 1;
    switch(x) {
        case 1: continue;
    }
}
""", "MustInLoop"))

tests.append((99, "Break in function called from loop", """
void helper() { break; }
void main() {
    while(1) { helper(); }
}
""", "MustInLoop"))

tests.append((100, "Continue in nested function call", """
void helper() { continue; }
void main() {
    for(int i=0; i<1; ++i) { helper(); }
}
""", "MustInLoop"))

with open('d:/Projects/tyc-compiler/tests/test_checker.py', 'w') as f:
    f.write(HEADER)
    for t in tests:
        f.write(gen_test(*t))

print("Done generating 100 test cases.")
