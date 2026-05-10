import sys

with open('src/semantics/static_checker.py', 'r') as f:
    lines = f.readlines()

new_lines = []

i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('class UnknownType(Type):'):
        i += 5
        continue
    if line.startswith('TyCType ='):
        i += 1
        continue
    
    if line.startswith('class StaticChecker(ASTVisitor):'):
        new_lines.append(line)
        new_lines.append('    class UnknownType(Type):\n')
        new_lines.append('        def __str__(self):\n')
        new_lines.append('            return "UnknownType()"\n')
        new_lines.append('        def accept(self, visitor, o=None):\n')
        new_lines.append('            pass\n\n')
        i += 1
        continue
    
    if 'UnknownType' in line:
        line = line.replace('UnknownType', 'StaticChecker.UnknownType')
    
    new_lines.append(line)
    i += 1

with open('src/semantics/static_checker.py', 'w') as f:
    f.writelines(new_lines)
