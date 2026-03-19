with open("src/coreason_actuator/sandbox.py") as f:
    content = f.read()

# Let's fix the script string format so that it simply defines the globals in python
old = """    def patched_solve(*args, **kwargs):
        solver.add(*args)
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            result = {}
            for decl in model.decls():
                name = decl.name()
                if name in expected_proof_schema:
                    val = model[decl]
                    if z3.is_int_value(val):
                        result[name] = val.as_long()
                    elif z3.is_true(val):
                        result[name] = True
                    elif z3.is_false(val):
                        result[name] = False
                    else:
                        result[name] = str(val)
            print(json.dumps(result))
            sys.exit(0)
        if res == z3.unsat:
            sys.stderr.write(f"UNSAT")
            sys.exit(1)
        sys.stderr.write("Timeout")
        sys.exit(2)

    local_vars = {"z3": z3, "Int": z3.Int, "Real": z3.Real, "Bool": z3.Bool, "solve": patched_solve, "solver": solver}
    try:
        exec(formal_grammar_payload, local_vars)
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(3)"""

new = """    def patched_solve(*args, **kwargs):
        solver.add(*args)
        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            result = {}
            for decl in model.decls():
                name = decl.name()
                if name in expected_proof_schema:
                    val = model[decl]
                    if z3.is_int_value(val):
                        result[name] = val.as_long()
                    elif z3.is_true(val):
                        result[name] = True
                    elif z3.is_false(val):
                        result[name] = False
                    else:
                        result[name] = str(val)
            print(json.dumps(result))
            sys.exit(0)
        if res == z3.unsat:
            sys.stderr.write(f"UNSAT")
            sys.exit(1)
        sys.stderr.write("Timeout")
        sys.exit(2)

    # Build globals dynamically inside the script!
    script_globals = {
        "z3": z3,
        "Int": z3.Int,
        "Real": z3.Real,
        "Bool": z3.Bool,
        "solve": patched_solve,
        "solver": solver
    }
    try:
        exec(formal_grammar_payload, script_globals)
    except SystemExit:
        raise
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(3)"""

content = content.replace(old, new)

with open("src/coreason_actuator/sandbox.py", "w") as f:
    f.write(content)
