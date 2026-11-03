# lit config for compiler pass tests: `lokinfer-opt ... | filecheck %s`.
import os
import lit.formats

config.name = "lokinfer"
config.test_format = lit.formats.ShTest()
config.suffixes = [".lkir"]
config.test_source_root = os.path.dirname(__file__)

repo = os.path.abspath(os.path.join(config.test_source_root, "..", ".."))
build_dir = os.environ.get("LOKINFER_BUILD_DIR", os.path.join(repo, "build"))
venv_bin = os.path.join(repo, ".venv", "bin")
config.environment["PATH"] = os.pathsep.join([build_dir, venv_bin, os.environ.get("PATH", "")])
