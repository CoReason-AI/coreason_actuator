with open("tests/test_z3_engine.py") as f:
    content = f.read()

content = content.replace(
    'provider.enforce_filesystem_immutability(["/dev/shm"])',
    'provider.enforce_filesystem_immutability(["/dev/shm"])  # noqa: S108',
)

with open("tests/test_z3_engine.py", "w") as f:
    f.write(content)
