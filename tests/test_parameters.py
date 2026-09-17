import unittest

from kglobal_analysis.parameters import parse_parameters


class ParameterTests(unittest.TestCase):
    def test_upstream_style_dimensions_and_cadence(self):
        p = parse_parameters('''
! KGlobal parameter example
#define nx 32
#define ny 32
#define nz 1
#define pex 32
#define pey 16
#define pez 1
#define dt .0002
#define n_movieout 1000
#define movie_header "movie_kglobal3.0.h"
#define double_byte
!#define four_byte
''')
        self.assertEqual((p.nx, p.ny, p.nz), (32, 32, 1))
        self.assertEqual((p.pex, p.pey, p.pez), (32, 16, 1))
        self.assertEqual((p.Nx, p.Ny, p.Nz), (1024, 512, 1))
        self.assertAlmostEqual(p.movie_dt, 0.2)
        self.assertEqual(p.movie_header, "movie_kglobal3.0.h")
        self.assertTrue(p.double_byte)
        self.assertFalse(p.four_byte)

    def test_whitespace_comments_parentheses_and_fortran_exponent(self):
        p = parse_parameters('''
/* ignored definitions:
#define nx 999
*/
  # define nx (32) // local grid
#define pex 2 ! processor count
#define dt (-2.5D-5)
#define n_movieout 2000
#define four_byte 0
#define movie_header "folder//movie!header.h" /* preserve string */
''')
        self.assertEqual(p.Nx, 64)
        self.assertAlmostEqual(p.movie_dt, -0.05)
        self.assertTrue(p.four_byte)  # #ifdef tests presence, not value
        self.assertEqual(p.movie_header, "folder//movie!header.h")

    def test_missing_fields_remain_unknown(self):
        p = parse_parameters("#define nx 32\n#define unrelated symbolic_value\n")
        self.assertIsNone(p.Nx)
        self.assertIsNone(p.Ny)
        self.assertIsNone(p.Nz)
        self.assertIsNone(p.movie_dt)
        self.assertFalse(p.double_byte)
        self.assertEqual(p.definitions["unrelated"], "symbolic_value")

    def test_redefinition_and_undef(self):
        p = parse_parameters("#define nx 16\n#define nx 32\n"
                             "#define double_byte\n#undef double_byte\n")
        self.assertEqual(p.nx, 32)
        self.assertFalse(p.double_byte)

    def test_cadence_requires_both_resolved_inputs(self):
        for text in ("", "#define dt .000025", "#define n_movieout 2000",
                     "#define dt .000025\n#define movieout 4.",
                     "#define dt .000025\n#define n_movieout 2000\n#undef dt"):
            with self.subTest(text=text):
                self.assertIsNone(parse_parameters(text).movie_dt)
        self.assertAlmostEqual(parse_parameters(
            "#define dt .000025\n#define n_movieout 2000").movie_dt, 0.05)

    def test_partial_dimensions_are_independent(self):
        p = parse_parameters("#define nx 32\n#define pex 256\n#define ny 32\n"
                             "#define nz 1\n#define pez 1")
        self.assertEqual((p.Nx, p.Ny, p.Nz), (8192, None, 1))

    def test_all_conditional_directives_and_selected_expressions_rejected(self):
        for directive in ("if 0", "ifdef FLAG", "ifndef FLAG", "elif 1", "else", "endif"):
            with self.subTest(directive=directive):
                with self.assertRaises(ValueError):
                    parse_parameters(f"#{directive}\n#define nx 32")
        for name, expression in (("nx", "(16 * 2)"), ("dt", "(1.0 / 1000)"),
                                 ("n_movieout", "INTERVAL")):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    parse_parameters(f"#define {name} {expression}")

    def test_unterminated_block_comment_rejected(self):
        with self.assertRaises(ValueError):
            parse_parameters("/* unfinished comment\n#define nx 999")

    def test_continued_comment_rejected(self):
        # CPP splices lines before removing comments; nx would be commented out.
        with self.assertRaises(ValueError):
            parse_parameters("// continued comment \\\n#define nx 999")

    def test_invalid_or_unsupported_input_is_explicit(self):
        for text in ("#define nx 16*2", "#define nx SIZE", "#define ny 0",
                     "#define nz -1", "#define dt nan", "#define dt 0",
                     "#define dt 1e999", "#define movie_header unquoted.h",
                     "#ifdef X\n#define nx 32\n#endif", '#include "other"',
                     "#define F(x) x", "#define nx 32\\\n", "#undef nx junk"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_parameters(text)


if __name__ == "__main__":
    unittest.main()
