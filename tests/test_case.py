from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import io

from kglobal_analysis import KGlobalCase


class CaseTests(unittest.TestCase):
    def test_case_inspection_and_explicit_parameter_selection(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "movie.bx.015").write_bytes(b"not binary movie data")
            (root / "movie.log.015").touch()
            (root / "p3d.stdout.015").touch()
            (root / "param_one").write_text("#define nx 32\n#define pex 2\n")
            case = KGlobalCase(root)
            self.assertEqual(case.parameters.Nx, 64)
            self.assertEqual(case.variables, ("bx",))
            self.assertEqual(case.suffixes, ("015",))
            summary = StringIO()
            case.inspect(file=summary)
            self.assertIn("Global grid (Nx, Ny, Nz): 64, unknown, unknown", summary.getvalue())
            self.assertIn("015: variables=bx; log=movie.log.015; stdout=p3d.stdout.015",
                          summary.getvalue())
            (root / "param_two").write_text("#define nx 16\n")
            with self.assertRaisesRegex(ValueError, "Multiple parameter files"):
                KGlobalCase(root)
            self.assertEqual(KGlobalCase(root, parameter_file="param_two").parameters.nx, 16)

    def test_no_parameters_and_param_precedence(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "param_backup~").touch()
            case = KGlobalCase(root)
            self.assertIsNone(case.parameter_file)
            self.assertEqual(case.variables, ())
            (root / "param").write_text("#define nx 4\n")
            (root / "param_other").write_text("#define nx 8\n")
            self.assertEqual(KGlobalCase(root).parameters.nx, 4)
            with self.assertRaises(FileNotFoundError):
                KGlobalCase(root, parameter_file="missing")

    def test_not_a_directory(self):
        with self.assertRaises(NotADirectoryError):
            KGlobalCase(__file__)

    def test_only_parameter_contents_are_opened(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            parameter = root / "param"
            parameter.write_text("#define nx 32\n")
            for name in ("movie.bx.000", "movie.log.042", "p3d.stdout.100"):
                (root / name).touch()
            original_open = io.open
            opened = []

            def guarded_open(path, mode="r", *args, **kwargs):
                self.assertEqual(Path(path), parameter)
                self.assertEqual(mode, "r")
                opened.append(Path(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", new=guarded_open), \
                    patch("io.open", side_effect=guarded_open), \
                    patch("builtins.open", side_effect=guarded_open):
                case = KGlobalCase(root)
                case.inspect(file=StringIO())
            self.assertEqual(opened, [parameter])


if __name__ == "__main__":
    unittest.main()
