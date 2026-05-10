"""
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


def test_001():
    """Test a valid program that should pass all checks"""
    source = """
void main() {
    int x = 5;
    int y = x + 1;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_002():
    """Test valid program with auto type inference"""
    source = """
void main() {
    auto x = 10;
    auto y = 3.14;
    auto z = x + y;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_003():
    """Test valid program with functions"""
    source = """
int add(int x, int y) {
    return x + y;
}
void main() {
    int sum = add(5, 3);
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_004():
    """Test valid program with struct"""
    source = """
struct Point {
    int x;
    int y;
};
void main() {
    Point p;
    p.x = 10;
    p.y = 20;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_005():
    """Test valid program with nested blocks"""
    source = """
void main() {
    int x = 10;
    {
        int y = 20;
        int z = x + y;
    }
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_006():
    """Test valid program with shadowed variables"""
    source = """
void main() {
    int x = 10;
    {
        int x = 20;
        {
            int x = 30;
        }
    }
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_007():
    """Test valid program with struct containing another struct"""
    source = """
struct Point { int x; int y; };
struct Rect { Point p1; Point p2; };
void main() {
    Rect r;
    r.p1.x = 0;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_008():
    """Test valid function calls including built-ins"""
    source = """
void main() {
    int x = readInt();
    printInt(x);
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_009():
    """Test valid for loop with break and continue"""
    source = """
void main() {
    for (int i = 0; i < 10; ++i) {
        if (i == 5) break;
        if (i == 2) continue;
    }
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_010():
    """Test valid while loop with break and continue"""
    source = """
void main() {
    int i = 0;
    while (i < 10) {
        if (i == 5) break;
        if (i == 2) continue;
        ++i;
    }
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_011():
    """Test valid switch statement"""
    source = """
void main() {
    int x = 2;
    switch(x) {
        case 1: x = 10; break;
        case 2: x = 20; break;
    }
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_012():
    """Test valid auto inference with literals and functions"""
    source = """
int getVal() { return 42; }
void main() {
    auto a = getVal();
    auto b;
    b = a + 5;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_013():
    """Test valid relational and logical operators"""
    source = """
void main() {
    int a = 5;
    float b = 6.0;
    int c = a < b;
    int d = c && (a == 5);
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_014():
    """Test valid assignment expressions"""
    source = """
void main() {
    int x;
    int y = (x = 5) + 3;
    int a; int b; int c;
    a = b = c = 10;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_015():
    """Test valid struct literal initialization"""
    source = """
struct Point { int x; int y; };
void main() {
    Point p1 = {10, 20};
    Point p2;
    p2 = p1;
}
"""
    expected = "Static checking passed"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_016():
    """Redeclared Struct"""
    source = """
struct Point { int x; };
struct Point { int y; };
void main() {}
"""
    expected = "Redeclared(Struct, Point)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_017():
    """Redeclared Function"""
    source = """
int add() { return 1; }
int add(int x) { return x; }
void main() {}
"""
    expected = "Redeclared(Function, add)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_018():
    """Redeclared Variable in same block"""
    source = """
void main() {
    int x = 5;
    int x = 10;
}
"""
    expected = "Redeclared(Variable, x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_019():
    """Redeclared Parameter"""
    source = """
int calculate(int x, int x) {
    return x;
}
void main() {}
"""
    expected = "Redeclared(Parameter, x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_020():
    """Local variable reuses parameter name"""
    source = """
void func(int x) {
    int x = 10;
}
void main() {}
"""
    expected = "Redeclared(Variable, x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_021():
    """Duplicate struct member"""
    source = """
struct Point {
    int x;
    int x;
};
void main() {}
"""
    expected = "Redeclared(Member, x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_022():
    """Multiple variable redeclarations"""
    source = """
void main() {
    int a = 1;
    float b = 2.0;
    string a = "test";
}
"""
    expected = "Redeclared(Variable, a)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_023():
    """Local variable reuses parameter name in nested block"""
    source = """
void func(int p) {
    {
        int p = 5;
    }
}
void main() {}
"""
    expected = "Redeclared(Variable, p)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_024():
    """Struct member duplicate with different type"""
    source = """
struct Test {
    int val;
    float val;
};
void main() {}
"""
    expected = "Redeclared(Member, val)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_025():
    """Function parameter duplicate with different type"""
    source = """
void test(int a, float a) {}
void main() {}
"""
    expected = "Redeclared(Parameter, a)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_026():
    """Undeclared Variable"""
    source = """
void main() {
    int x = y + 1;
}
"""
    expected = "UndeclaredIdentifier(y)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_027():
    """Variable used before declaration"""
    source = """
void main() {
    x = 5;
    int x;
}
"""
    expected = "UndeclaredIdentifier(x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_028():
    """Out of scope access"""
    source = """
void main() {
    { int x = 5; }
    int y = x;
}
"""
    expected = "UndeclaredIdentifier(x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_029():
    """Variable used in its own initializer"""
    source = """
void main() {
    int x = x + 1;
}
"""
    expected = "UndeclaredIdentifier(x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_030():
    """Undeclared in expression"""
    source = """
void main() {
    int x = 5 + z * 2;
}
"""
    expected = "UndeclaredIdentifier(z)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_031():
    """Undeclared in function call"""
    source = """
void foo(int a) {}
void main() {
    foo(undeclared);
}
"""
    expected = "UndeclaredIdentifier(undeclared)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_032():
    """Undeclared as condition"""
    source = """
void main() {
    if (cond) {}
}
"""
    expected = "UndeclaredIdentifier(cond)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_033():
    """Undeclared in return"""
    source = """
int foo() {
    return retVal;
}
void main() {}
"""
    expected = "UndeclaredIdentifier(retVal)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_034():
    """Out of scope access in another function"""
    source = """
void func1() { int a = 1; }
void func2() { int b = a; }
void main() {}
"""
    expected = "UndeclaredIdentifier(a)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_035():
    """Parameter used outside function"""
    source = """
void func1(int p) {}
void func2() { int x = p; }
void main() {}
"""
    expected = "UndeclaredIdentifier(p)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_036():
    """Call undeclared function"""
    source = """
void main() {
    foo();
}
"""
    expected = "UndeclaredFunction(foo)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_037():
    """Call function before declaration"""
    source = """
void main() {
    bar();
}
void bar() {}
"""
    expected = "UndeclaredFunction(bar)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_038():
    """Undeclared function in assignment"""
    source = """
void main() {
    int x = getVal();
}
"""
    expected = "UndeclaredFunction(getVal)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_039():
    """Undeclared function in expression"""
    source = """
void main() {
    int x = 5 + calc(2);
}
"""
    expected = "UndeclaredFunction(calc)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_040():
    """Undeclared function with arguments"""
    source = """
void main() {
    process(1, 2.0, "3");
}
"""
    expected = "UndeclaredFunction(process)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_041():
    """Undeclared function in return"""
    source = """
int foo() {
    return getMissing();
}
void main() {}
"""
    expected = "UndeclaredFunction(getMissing)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_042():
    """Undeclared function in condition"""
    source = """
void main() {
    if (check()) {}
}
"""
    expected = "UndeclaredFunction(check)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_043():
    """Undeclared function nested call"""
    source = """
void main() {
    printInt(missing());
}
"""
    expected = "UndeclaredFunction(missing)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_044():
    """Call variable as function"""
    source = """
void main() {
    int x = 5;
    x();
}
"""
    expected = "UndeclaredFunction(x)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_045():
    """Call parameter as function"""
    source = """
void test(int p) {
    p();
}
void main() {}
"""
    expected = "UndeclaredFunction(p)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_046():
    """Variable of undeclared struct"""
    source = """
void main() {
    Point p;
}
"""
    expected = "UndeclaredStruct(Point)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_047():
    """Struct type used before declaration"""
    source = """
void test() {
    Data d;
}
struct Data { int x; };
void main() {}
"""
    expected = "UndeclaredStruct(Data)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_048():
    """Struct member using undeclared struct"""
    source = """
struct Node {
    Missing m;
};
void main() {}
"""
    expected = "UndeclaredStruct(Missing)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_049():
    """Undeclared struct in parameter"""
    source = """
void process(Unknown u) {}
void main() {}
"""
    expected = "UndeclaredStruct(Unknown)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_050():
    """Undeclared struct as return type"""
    source = """
Result getRes() {}
void main() {}
"""
    expected = "UndeclaredStruct(Result)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_051():
    """Undeclared struct in multiple declarations"""
    source = """
void main() {
    int x;
    BadStruct b;
}
"""
    expected = "UndeclaredStruct(BadStruct)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_052():
    """Struct inheriting undeclared struct (member)"""
    source = """
struct A { B b; };
void main() {}
"""
    expected = "UndeclaredStruct(B)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_053():
    """Undeclared struct in function prototype"""
    source = """
void doWork(Data d) {}
void main() {}
"""
    expected = "UndeclaredStruct(Data)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_054():
    """Struct literal with undeclared struct"""
    source = """
void main() {
    Missing m = {1, 2};
}
"""
    expected = "UndeclaredStruct(Missing)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_055():
    """Undeclared struct in nested scope"""
    source = """
void main() {
    {
        InnerStruct s;
    }
}
"""
    expected = "UndeclaredStruct(InnerStruct)"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_056():
    """Neither auto has known type"""
    source = """
void main() {
    auto x;
    auto y;
    auto z = x + y;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_057():
    """Auto without init never used"""
    source = """
void main() {
    auto x;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_058():
    """Auto mutually dependent assignment"""
    source = """
void main() {
    auto a;
    auto b;
    a = b;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_059():
    """Auto relational operator"""
    source = """
void main() {
    auto x;
    auto y;
    int z = x < y;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_060():
    """Auto string mix"""
    source = """
void main() {
    auto x;
    auto y = x + "hello";
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_061():
    """Auto returned but unknown"""
    source = """
func() {
    auto x;
    return x;
}
void main() {}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_062():
    """Auto passed to printInt but no type"""
    source = """
void main() {
    auto a;
    auto b;
    int c = a * b;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_063():
    """Auto assignment loop"""
    source = """
void main() {
    auto a;
    a = a;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_064():
    """Auto unused in nested block"""
    source = """
void main() {
    {
        auto val;
    }
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_065():
    """Auto mixed with float"""
    source = """
void main() {
    auto x;
    auto y;
    float f = x + y;
}
"""
    expected = "TypeCannotBeInferred"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_066():
    """If condition float"""
    source = """
void main() {
    if (3.14) {}
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_067():
    """If condition string"""
    source = """
void main() {
    if ("true") {}
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_068():
    """If condition struct"""
    source = """
struct S { int x; };
void main() {
    S s;
    if (s) {}
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_069():
    """While condition string"""
    source = """
void main() {
    while ("loop") {}
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_070():
    """For condition float"""
    source = """
void main() {
    for (int i = 0; 1.5; ++i) {}
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_071():
    """Assignment different types"""
    source = """
void main() {
    int x;
    x = 3.14;
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_072():
    """Assignment different struct types"""
    source = """
struct A { int x; };
struct B { int x; };
void main() {
    A a; B b;
    a = b;
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_073():
    """Return string in int func"""
    source = """
int foo() {
    return "text";
}
void main() {}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_074():
    """Return void in int func"""
    source = """
int foo() {
    return;
}
void main() {}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_075():
    """Return int in void func"""
    source = """
void main() {
    return 1;
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_076():
    """Switch condition float"""
    source = """
void main() {
    switch (3.14) { case 1: break; }
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_077():
    """Switch condition string"""
    source = """
void main() {
    switch ("str") { case 1: break; }
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_078():
    """Switch condition struct"""
    source = """
struct S { int x; };
void main() {
    S s;
    switch (s) { case 1: break; }
}
"""
    expected = "TypeMismatchInStatement"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_079():
    """Int + string"""
    source = """
void main() {
    int x = 5 + "text";
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_080():
    """Int + struct"""
    source = """
struct S { int x; };
void main() {
    S s;
    int x = 5 + s;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_081():
    """Float % int"""
    source = """
void main() {
    int x = 5.5 % 2;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_082():
    """Float < string"""
    source = """
void main() {
    int b = 1.0 < "1";
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_083():
    """Struct == struct"""
    source = """
struct S { int x; };
void main() {
    S s1; S s2;
    int b = s1 == s2;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_084():
    """Float && int"""
    source = """
void main() {
    int b = 1.5 && 1;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_085():
    """Logical NOT on float"""
    source = """
void main() {
    int b = !1.5;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_086():
    """Increment float"""
    source = """
void main() {
    float f = 1.0;
    ++f;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_087():
    """Increment literal"""
    source = """
void main() {
    ++5;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_088():
    """Increment expression"""
    source = """
void main() {
    int x = 1;
    (x + 1)++;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_089():
    """Member access on non-struct"""
    source = """
void main() {
    int x = 1;
    int y = x.mem;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_090():
    """Non-existent member access"""
    source = """
struct S { int x; };
void main() {
    S s;
    int y = s.y;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_091():
    """Call wrong arg type"""
    source = """
void foo(int a) {}
void main() {
    foo("123");
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_092():
    """Call wrong arg count"""
    source = """
void foo(int a) {}
void main() {
    foo(1, 2);
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_093():
    """Assignment expression invalid LHS"""
    source = """
void main() {
    int x = (5 = 2) + 1;
}
"""
    expected = "TypeMismatchInExpression"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_094():
    """Break outside loop"""
    source = """
void main() {
    break;
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_095():
    """Continue outside loop"""
    source = """
void main() {
    continue;
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_096():
    """Break in if"""
    source = """
void main() {
    if (1) { break; }
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_097():
    """Continue in if"""
    source = """
void main() {
    if (1) { continue; }
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_098():
    """Continue in switch"""
    source = """
void main() {
    int x = 1;
    switch(x) {
        case 1: continue;
    }
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_099():
    """Break in function called from loop"""
    source = """
void helper() { break; }
void main() {
    while(1) { helper(); }
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"

def test_100():
    """Continue in nested function call"""
    source = """
void helper() { continue; }
void main() {
    for(int i=0; i<1; ++i) { helper(); }
}
"""
    expected = "MustInLoop"
    assert expected in Checker(source).check_from_source(), f"Expected '{expected}' but got '{Checker(source).check_from_source()}'"
