import pytest

from xcdiff.general import comment_zero_lines, GeneralFunctional


@pytest.mark.parametrize(
    'code, expected',
    [
        ("", ""),
        ("\nfoo= (0)*factor;", "\n// foo= (0)*factor;"),
        (
"""
  foo
  ds->df00004 += (0)*factor;
  bar
""",
"""
  foo
  // ds->df00004 += (0)*factor;
  bar
"""
        ),
        (
"""
foo
ds->df00004 += (0)*factor;
bar
""",
"""
foo
// ds->df00004 += (0)*factor;
bar
"""
        ),
    ]
)
def test_strip0(code, expected):

    output = comment_zero_lines(code)
    assert output == expected
