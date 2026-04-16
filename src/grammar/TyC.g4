grammar TyC;

@lexer::header {
from lexererr import *
}

@lexer::members {
def emit(self):
    tk = self.type
    if tk == self.STRING_LIT:
        # Get the original text (e.g., '"hello"')
        original_text = self.text
        # Strip the first and last characters (the quotes)
        self.text = original_text[1:-1]
        return super().emit()
        
    elif tk == self.UNCLOSE_STRING:       
        result = super().emit()
        # Lexeme does not include opening quote 
        raise UncloseString(result.text[1:])
        
    elif tk == self.ILLEGAL_ESCAPE:
        result = super().emit()
        # Wrong string is from beginning without opening quote 
        raise IllegalEscape(result.text[1:])
        
    elif tk == self.ERROR_CHAR:
        result = super().emit()
        raise ErrorToken(result.text)
        
    else:
        return super().emit()
}

options {
    language = Python3;
}

// --- PARSER RULES ---

program: (structDecl | funcDecl)* EOF; 

// Declarations
structDecl: STRUCT ID LBRACE memberDecl* RBRACE SEMI; 
memberDecl: (primitiveType | ID) ID SEMI; 

funcDecl: (type | VOID)? ID LPAREN paramList? RPAREN block; 

paramList: param (COMMA param)*; 
param: (primitiveType | ID) ID; 

// Statements
statement: varDecl 
         | block 
         | ifStmt 
         | whileStmt 
         | forStmt 
         | switchStmt 
         | breakStmt 
         | continueStmt 
         | returnStmt 
         | exprStmt; 

varDecl: (AUTO | type) ID (ASSIGN expr)? SEMI; 

block: LBRACE (varDecl | statement)* RBRACE; 

ifStmt: IF LPAREN expr RPAREN statement (ELSE statement)?; 

whileStmt: WHILE LPAREN expr RPAREN statement; 

forStmt: FOR LPAREN (varDecl | exprStmt | SEMI) (expr? SEMI) expr? RPAREN statement; 

switchStmt: SWITCH LPAREN expr RPAREN LBRACE (caseStmt | defaultStmt)* RBRACE; 
caseStmt: CASE expr COLON (varDecl | statement)*; 
defaultStmt: DEFAULT COLON (varDecl | statement)*; 

breakStmt: BREAK SEMI; 
continueStmt: CONTINUE SEMI; 
returnStmt: RETURN expr? SEMI; 
exprStmt: expr SEMI; 

// Types
type: primitiveType | ID; 
primitiveType: INT | FLOAT | STRING; 

// Expressions (Precedence: Highest to Lowest) 
expr: primary                                   #primaryExpr
    | expr DOT ID                               #memberAccess
    | expr (INC | DEC | LPAREN argList? RPAREN) #postfixExpr
    | (NOT | SUB | ADD | INC | DEC) expr        #unaryExpr
    | expr (MUL | DIV | MOD) expr               #binaryExpr
    | expr (ADD | SUB) expr                     #binaryExpr
    | expr (LT | LTE | GT | GTE) expr           #binaryExpr
    | expr (EQ | NEQ) expr                      #binaryExpr
    | expr AND expr                             #binaryExpr
    | expr OR expr                              #binaryExpr
    | <assoc=right> expr ASSIGN expr            #assignExpr
    ;

argList: expr (COMMA expr)*; 

primary: ID 
       | INT_LIT 
       | FLOAT_LIT 
       | STRING_LIT 
       | structLiteral 
       | LPAREN expr RPAREN; 

structLiteral: LBRACE argList? RBRACE; 

// --- LEXER RULES ---

// Keywords 
AUTO: 'auto';
BREAK: 'break';
CASE: 'case';
CONTINUE: 'continue';
DEFAULT: 'default';
ELSE: 'else';
FLOAT: 'float';
FOR: 'for';
IF: 'if';
INT: 'int';
RETURN: 'return';
STRING: 'string';
STRUCT: 'struct';
SWITCH: 'switch';
VOID: 'void';
WHILE: 'while';

// Operators & Separators 
ADD: '+';
SUB: '-';
MUL: '*';
DIV: '/';
MOD: '%';
EQ: '==';
NEQ: '!=';
LT: '<';
GT: '>';
LTE: '<=';
GTE: '>=';
OR: '||';
AND: '&&';
NOT: '!';
INC: '++';
DEC: '--';
ASSIGN: '=';
DOT: '.';

LBRACE: '{';
RBRACE: '}';
LPAREN: '(';
RPAREN: ')';
SEMI: ';';
COMMA: ',';
COLON: ':';

// Literals 
INT_LIT: [0-9]+; 

FLOAT_LIT: ([0-9]+ '.' [0-9]* EXP? | '.' [0-9]+ EXP? | [0-9]+ EXP); 
fragment EXP: [eE] [+-]? [0-9]+;

// Valid String
STRING_LIT: '"' ( ESC | ~["\\\r\n] )* '"'; 
fragment ESC: '\\' [bfrtn"\\/]; // Removed the backslash before /

ID: [a-zA-Z_] [a-zA-Z_0-9]*; 

// Comments & WS 
BLOCK_COMMENT: '/*' .*? '*/' -> skip;
LINE_COMMENT: '//' ~[\r\n]* -> skip;
WS : [ \t\r\n\f]+ -> skip;

// Error Handling Requirements [cite: 1, 2]
ILLEGAL_ESCAPE: '"' ( ESC | ~["\\\r\n] )* '\\' ~[bfrtn"\\/]; // Removed the backslash before /
UNCLOSE_STRING: '"' ( ESC | ~["\\\r\n] )*;
ERROR_CHAR: .;