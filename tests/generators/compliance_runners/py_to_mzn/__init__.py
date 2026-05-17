import argparse
import ast
import importlib
import re
from collections.abc import Iterable
from pathlib import Path


class Convertor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._enums: list[str] = []
        self._records: list[str] = []
        self._vars: list[str] = []
        self._constraints: list[str] = []

    def convert(self, code: str) -> str:
        self._enums = []
        self._records = []
        self._vars = []
        self._constraints = []
        self.visit(ast.parse(code))
        sections = [
            self._enums,
            self._records,
            self._vars,
            self._constraints,
        ]
        return "\n\n".join("\n".join(section) for section in sections if section) + "\n"

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _is_enum(node):
            self._enums.append(_convert_enum(node))
        else:
            self._records.append(_convert_record(node))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only top-level named variable declarations are supported")
        if not _is_ellipsis(node.value):
            raise ValueError("Top-level annotated assignments must use ellipsis")
        self._vars.append(f"var {_annotation_name(node.annotation)}: {node.target.id};")

    def visit_Expr(self, node: ast.Expr) -> None:
        self._constraints.append(f"constraint {_expr(node.value)};")

    def visit_If(self, node: ast.If) -> None:
        if node.orelse:
            raise ValueError("MiniZinc conversion does not support else clauses")
        body = [_expr(stmt.value) for stmt in node.body if isinstance(stmt, ast.Expr)]
        if len(body) != len(node.body):
            raise ValueError("If bodies may only contain expressions")
        conclusion = body[0] if len(body) == 1 else "/\\".join(f"({expr})" for expr in body)
        self._constraints.append(f"constraint ({_expr(node.test)}) -> ({conclusion});")


def get_solutions(model: str) -> Iterable[dict[str, object]]:
    minizinc = importlib.import_module("minizinc")

    mzn_model = minizinc.Model()
    mzn_model.add_string(model.rstrip() + "\n\nsolve satisfy;\n")
    instance = minizinc.Instance(minizinc.Solver.lookup("gecode"), mzn_model)
    result = instance.solve(all_solutions=True)

    solutions = result.solution
    if solutions is None:
        return
    if not isinstance(solutions, list):
        solutions = [solutions]
    for solution in solutions:
        yield _normalise_solution(solution.p)


def _is_enum(node: ast.ClassDef) -> bool:
    return any(_annotation_name(base) == "Enum" for base in node.bases)


def _convert_enum(node: ast.ClassDef) -> str:
    members = []
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            members.append(stmt.targets[0].id)
    return f"enum {node.name} = {{ {', '.join(members)} }};"


def _convert_record(node: ast.ClassDef) -> str:
    fields = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(f"  {_annotation_name(stmt.annotation)}: {stmt.target.id},")
    return f"type {node.name} = record(\n" + "\n".join(fields) + "\n);"


def _is_ellipsis(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _annotation_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return "bool" if node.id == "bool" else node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    raise ValueError(f"Unsupported annotation: {ast.dump(node)}")


def _expr(node: ast.AST) -> str:
    if isinstance(node, ast.BoolOp):
        op = "/\\" if isinstance(node.op, ast.And) else "\\/"
        return op.join(f"({_expr(value)})" for value in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"not ({_expr(node.operand)})"
    if isinstance(node, ast.Compare):
        return _compare(node.left, node.ops, node.comparators)
    if isinstance(node, ast.Attribute):
        return _attribute(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Set):
        return "{ " + ", ".join(_expr(element) for element in node.elts) + " }"
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "true" if node.value else "false"
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _compare(left: ast.AST, ops: list[ast.cmpop], comparators: list[ast.expr]) -> str:
    if len(ops) != 1 or len(comparators) != 1:
        raise ValueError("Only simple comparisons are supported")
    op = ops[0]
    operator = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.In: "in",
        ast.NotIn: "not in",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }.get(type(op))
    if operator is None:
        raise ValueError(f"Unsupported comparison operator: {ast.dump(op)}")
    right = _expr(comparators[0])
    if isinstance(op, ast.In | ast.NotIn):
        return f"({_expr(left)}) {operator} {right}"
    return f"({_expr(left)}) {operator} ({right})"


def _attribute(node: ast.Attribute) -> str:
    if isinstance(node.value, ast.Name):
        if node.value.id == "p":
            return f"({node.value.id}).{node.attr}"
        return node.attr
    return f"({_expr(node.value)}).{node.attr}"


def _normalise_solution(solution: object) -> dict[str, object]:
    return {key: _parse_value(value) for key, value in _solution_items(solution)}


def _solution_items(solution: object) -> Iterable[tuple[str, object]]:
    if isinstance(solution, dict):
        return solution.items()
    if hasattr(solution, "_asdict"):
        return solution._asdict().items()
    if hasattr(solution, "__dict__"):
        return solution.__dict__.items()
    return _split_record_entries(str(solution).strip("()"))


def _parse_solution(solution: str) -> dict[str, object]:
    match = re.search(r"p\s*=\s*\((.*)\)\s*;", solution, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse MiniZinc solution: {solution}")
    entries = _split_record_entries(match.group(1))
    return {key: _parse_value(value) for key, value in entries}


def _split_record_entries(record: str) -> list[tuple[str, str]]:
    entries = []
    for entry in record.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, value = entry.split(":", 1)
        entries.append((key.strip(), value.strip()))
    return entries


def _parse_value(value: object) -> object:
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "LT":
        return "<"
    if value == "EQ":
        return "="
    if value == "GT":
        return ">"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a validator-state Python model to MiniZinc")
    parser.add_argument("input", help="Python model file")
    parser.add_argument("output", nargs="?", help="MiniZinc output file; stdout if omitted")
    args = parser.parse_args()

    converted = Convertor().convert(Path(args.input).read_text())
    if args.output is None:
        print(converted, end="")
    else:
        Path(args.output).write_text(converted)


if __name__ == "__main__":
    main()
