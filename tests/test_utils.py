import importlib
import shutil
from pathlib import Path


def test_logger_mkdir() -> None:
    # Remove logs dir if exists to trigger the mkdir
    logs_dir = Path("logs")
    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    # Reload module to trigger the if not log_path.exists(): branch
    import coreason_actuator.utils.logger

    importlib.reload(coreason_actuator.utils.logger)

    assert logs_dir.exists()
