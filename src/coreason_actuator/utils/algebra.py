import ast


def verify_ast_safety(payload: str) -> bool:
    """
    Mechanistically sandboxes dynamically generated strings by compiling them into an AST
    and rigorously walking the graph to ensure no kinetic execution bleed occurs.
    """
    try:
        tree = ast.parse(payload, mode="eval")
    except SyntaxError as e:
        raise ValueError("Payload is not valid syntax.") from e

    base_allowlist = [
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.BinOp,
        ast.UnaryOp,
        ast.operator,
        ast.unaryop,
        ast.Subscript,
        ast.Slice,
    ]
    allowlist = tuple(base_allowlist)

    for node in ast.walk(tree):
        if not isinstance(node, allowlist):
            raise ValueError(f"Kinetic execution bleed detected. Forbidden AST node: {type(node).__name__}")

    return True
