import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap.sh"


def initialise_repo(tmp_path):
    """
    Create a minimal repository structure and copy the bootstrap script into it.
    
    Parameters:
        tmp_path (Path): Base temporary directory in which a "repo" directory will be created.
    
    Returns:
        Path: Path to the created repository directory.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, repo / "scripts" / "bootstrap.sh")
    return repo


def run_script(repo, env, *args):
    """
    Run the repository's scripts/bootstrap.sh with the provided environment and additional arguments.
    
    Parameters:
        repo (Path): Path to the repository containing a scripts/bootstrap.sh to execute.
        env (Mapping[str, str]): Environment variables to use for the subprocess.
        *args: Additional command-line arguments forwarded to the bootstrap script.
    
    Returns:
        subprocess.CompletedProcess: Completed process object containing `returncode`, `stdout`, and `stderr`.
    """
    cmd = ["bash", str(repo / "scripts" / "bootstrap.sh"), *args]
    return subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)


def test_bootstrap_success_generates_lockfile(tmp_path):
    """
    Verify that running the bootstrap script with installation skipped generates the expected lockfile and virtual environment.
    
    Runs the bootstrap script in a temporary repository with BOOTSTRAP_SKIP_INSTALL=1, asserts the process exits successfully, asserts a requirements.lock file exists and contains "neo4j-graphrag==0.9.0" and "pytest==8.3.2", and asserts a .venv directory was created.
    
    Parameters:
        tmp_path (pathlib.Path): pytest temporary directory fixture used as the repository root for the test.
    """
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update({"BOOTSTRAP_SKIP_INSTALL": "1"})

    result = run_script(repo, env)

    assert result.returncode == 0, result.stderr
    lockfile = repo / "requirements.lock"
    assert lockfile.exists()
    content = lockfile.read_text(encoding="utf-8")
    assert "neo4j-graphrag==0.9.0" in content
    assert "pytest==8.3.2" in content
    venv_path = repo / ".venv"
    assert venv_path.exists()


def test_bootstrap_rejects_incompatible_python(tmp_path):
    shim = tmp_path / "python311"
    script = textwrap.dedent(
        """
        #!/usr/bin/env python3
        import os
        import subprocess
        import sys

        REAL = os.environ.get("PYTHON_REAL", sys.executable)
        args = sys.argv[1:]

        if args and args[0] == "-c" and "sys.version_info" in args[1]:
            print("3.11.9")
            sys.exit(0)

        subprocess.check_call([REAL] + args)
        """
    ).strip() + "\n"
    shim.write_text(script, encoding="utf-8")
    shim.chmod(0o755)

    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "BOOTSTRAP_PYTHON_BIN": str(shim),
            "BOOTSTRAP_SKIP_INSTALL": "1",
            "PYTHON_REAL": sys.executable,
        }
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "must point to a Python 3.12" in result.stderr


def test_bootstrap_supports_custom_venv_path(tmp_path):
    """
    Verifies the bootstrap script accepts a custom virtual environment path and creates the specified venv.
    
    Parameters:
        tmp_path (pathlib.Path): Temporary directory provided by pytest used to create an isolated repository and custom venv path.
    """
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update({"BOOTSTRAP_SKIP_INSTALL": "1"})
    custom = tmp_path / "custom_venv"

    result = run_script(repo, env, "--venv-path", str(custom))

    assert result.returncode == 0, result.stderr
    assert custom.exists()


def test_bootstrap_is_idempotent(tmp_path):
    """
    Verifies that running the bootstrap script twice succeeds and produces an unchanged requirements.lock file.
    
    Runs the script once, records the generated requirements.lock content, runs the script a second time, and asserts both runs exit with code 0 and that the lockfile content is identical.
    """
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update({"BOOTSTRAP_SKIP_INSTALL": "1"})

    first = run_script(repo, env)
    assert first.returncode == 0, first.stderr
    lock_content = (repo / "requirements.lock").read_text(encoding="utf-8")

    second = run_script(repo, env)
    assert second.returncode == 0, second.stderr
    assert (repo / "requirements.lock").read_text(encoding="utf-8") == lock_content


def test_bootstrap_import_validation_success(tmp_path):
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "BOOTSTRAP_SKIP_INSTALL": "1",
            "BOOTSTRAP_TEST_IMPORT": "success",
        }
    )

    result = run_script(repo, env)

    assert result.returncode == 0, result.stderr
    assert "Running import validation" in result.stdout


def test_bootstrap_import_validation_failure(tmp_path):
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "BOOTSTRAP_SKIP_INSTALL": "1",
            "BOOTSTRAP_TEST_IMPORT": "fail",
        }
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "import failed" in result.stderr


def test_bootstrap_output_masks_secret(tmp_path):
    """
    Verify that running the bootstrap script completes successfully, prints the expected follow-up text, and masks sensitive values from both stdout and stderr.
    
    The test sets an example OPENAI_API_KEY in the environment, runs the script with installation skipped, asserts a zero exit code and presence of the "Next steps:" prompt, and confirms that the secret prefix `sk-test` does not appear in either stdout or stderr.
    """
    repo = initialise_repo(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "BOOTSTRAP_SKIP_INSTALL": "1",
            "OPENAI_API_KEY": "sk-test-123",
        }
    )

    result = run_script(repo, env)

    assert result.returncode == 0, result.stderr
    assert "Next steps:" in result.stdout
    assert "sk-test" not in result.stdout
    assert "sk-test" not in result.stderr
