import pytest
import sympy

from xcdiff.general import comment_zero_lines, GeneralFunctional


@pytest.mark.parametrize(
    'code, expected',
    [
        ("", ""),
        ("\nfoo= (0)*factor;", "\n// foo= (0)*factor;"),
        (
            "  foo\n  ds->df00004 += (0)*factor;\n  bar",
            "  foo\n  // ds->df00004 += (0)*factor;\n  bar",
        ),
        (
            "foo\nds->df00004 += (0)*factor;\nbar",
            "foo\n// ds->df00004 += (0)*factor;\nbar",
        ),
    ]
)
def test_strip0(code, expected):

    output = comment_zero_lines(code)
    assert output == expected


@pytest.fixture
def gga2x():
    ra, rb, ga, gb, gab = sympy.symbols(
        "dp->rhoa, dp->rhob, dp->grada, dp->gradb, dp->gradab"
    )
    func = GeneralFunctional("Example2x", ra, rb, ga, gb, gab, ra*rb*(ga*ga + gb*gb + 2*gab))
    return func


def test_gga2x_interface(gga2x):

    assert (
        gga2x.interface()
        == """
/* INTERFACE PART */
static integer example2x_isgga(void) { return 1; }
static integer example2x_read(const char* conf_line);
static real example2x_energy(const FunDensProp* dp);
static void example2x_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
static void example2x_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
static void example2x_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
static void example2x_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

Functional Example2xFunctional = {
  "Example2x",       /* name */
  example2x_isgga,   /* gga-corrected */
   3,
  example2x_read,
  NULL,
  example2x_energy,
  example2x_first,
  example2x_second,
  example2x_third,
  example2x_fourth
};
"""
    )


def test_example2x_energy(gga2x):

    reference = f"""

static real
example2x_energy(const FunDensProp* dp)
{{
  return dp->rhoa*dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2));
}}
"""
    assert gga2x.energy() == reference


def test_gga2x_first_derivatives(gga2x):
    reference = f"""
  ds->df1000 += (dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0100 += (dp->rhoa*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa*dp->rhob)*factor;
  ds->df0001 += (2*dp->gradb*dp->rhoa*dp->rhob)*factor;
  ds->df00001 += (2*dp->rhoa*dp->rhob)*factor;
"""
    assert gga2x.first_derivatives() == reference


def test_gga2x_second_derivatives(gga2x):
    reference = f"""
  ds->df2000 += (0)*factor;
  ds->df1100 += (pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2))*factor;
  ds->df1010 += (2*dp->grada*dp->rhob)*factor;
  ds->df1001 += (2*dp->gradb*dp->rhob)*factor;
  ds->df10001 += (2*dp->rhob)*factor;
  ds->df0200 += (0)*factor;
  ds->df0110 += (2*dp->grada*dp->rhoa)*factor;
  ds->df0101 += (2*dp->gradb*dp->rhoa)*factor;
  ds->df01001 += (2*dp->rhoa)*factor;
  ds->df0020 += (2*dp->rhoa*dp->rhob)*factor;
  ds->df0011 += (0)*factor;
  ds->df00101 += (0)*factor;
  ds->df0002 += (2*dp->rhoa*dp->rhob)*factor;
  ds->df00011 += (0)*factor;
  ds->df00002 += (0)*factor;
"""
    assert gga2x.second_derivatives() == reference


def test_gga2x_third_derivatives(gga2x):
    reference = f"""
  ds->df3000 += (0)*factor;
  ds->df2100 += (0)*factor;
  ds->df2010 += (0)*factor;
  ds->df2001 += (0)*factor;
  ds->df20001 += (0)*factor;
  ds->df1200 += (0)*factor;
  ds->df1110 += (2*dp->grada)*factor;
  ds->df1101 += (2*dp->gradb)*factor;
  ds->df11001 += (2)*factor;
  ds->df1020 += (2*dp->rhob)*factor;
  ds->df1011 += (0)*factor;
  ds->df10101 += (0)*factor;
  ds->df1002 += (2*dp->rhob)*factor;
  ds->df10011 += (0)*factor;
  ds->df10002 += (0)*factor;
  ds->df0300 += (0)*factor;
  ds->df0210 += (0)*factor;
  ds->df0201 += (0)*factor;
  ds->df02001 += (0)*factor;
  ds->df0120 += (2*dp->rhoa)*factor;
  ds->df0111 += (0)*factor;
  ds->df01101 += (0)*factor;
  ds->df0102 += (2*dp->rhoa)*factor;
  ds->df01011 += (0)*factor;
  ds->df01002 += (0)*factor;
  ds->df0030 += (0)*factor;
  ds->df0021 += (0)*factor;
  ds->df00201 += (0)*factor;
  ds->df0012 += (0)*factor;
  ds->df00111 += (0)*factor;
  ds->df00102 += (0)*factor;
  ds->df0003 += (0)*factor;
  ds->df00021 += (0)*factor;
  ds->df00012 += (0)*factor;
  ds->df00003 += (0)*factor;
"""
    assert gga2x.third_derivatives() == reference


def test_gga2x_fourth_derivatives(gga2x):
    reference = f"""
  ds->df4000 += (0)*factor;
  ds->df3100 += (0)*factor;
  ds->df3010 += (0)*factor;
  ds->df3001 += (0)*factor;
  ds->df30001 += (0)*factor;
  ds->df2200 += (0)*factor;
  ds->df2110 += (0)*factor;
  ds->df2101 += (0)*factor;
  ds->df21001 += (0)*factor;
  ds->df2020 += (0)*factor;
  ds->df2011 += (0)*factor;
  ds->df20101 += (0)*factor;
  ds->df2002 += (0)*factor;
  ds->df20011 += (0)*factor;
  ds->df20002 += (0)*factor;
  ds->df1300 += (0)*factor;
  ds->df1210 += (0)*factor;
  ds->df1201 += (0)*factor;
  ds->df12001 += (0)*factor;
  ds->df1120 += (2)*factor;
  ds->df1111 += (0)*factor;
  ds->df11101 += (0)*factor;
  ds->df1102 += (2)*factor;
  ds->df11011 += (0)*factor;
  ds->df11002 += (0)*factor;
  ds->df1030 += (0)*factor;
  ds->df1021 += (0)*factor;
  ds->df10201 += (0)*factor;
  ds->df1012 += (0)*factor;
  ds->df10111 += (0)*factor;
  ds->df10102 += (0)*factor;
  ds->df1003 += (0)*factor;
  ds->df10021 += (0)*factor;
  ds->df10012 += (0)*factor;
  ds->df10003 += (0)*factor;
  ds->df0400 += (0)*factor;
  ds->df0310 += (0)*factor;
  ds->df0301 += (0)*factor;
  ds->df03001 += (0)*factor;
  ds->df0220 += (0)*factor;
  ds->df0211 += (0)*factor;
  ds->df02101 += (0)*factor;
  ds->df0202 += (0)*factor;
  ds->df02011 += (0)*factor;
  ds->df02002 += (0)*factor;
  ds->df0130 += (0)*factor;
  ds->df0121 += (0)*factor;
  ds->df01201 += (0)*factor;
  ds->df0112 += (0)*factor;
  ds->df01111 += (0)*factor;
  ds->df01102 += (0)*factor;
  ds->df0103 += (0)*factor;
  ds->df01021 += (0)*factor;
  ds->df01012 += (0)*factor;
  ds->df01003 += (0)*factor;
  ds->df0040 += (0)*factor;
  ds->df0031 += (0)*factor;
  ds->df00301 += (0)*factor;
  ds->df0022 += (0)*factor;
  ds->df00211 += (0)*factor;
  ds->df00202 += (0)*factor;
  ds->df0013 += (0)*factor;
  ds->df00121 += (0)*factor;
  ds->df00112 += (0)*factor;
  ds->df00103 += (0)*factor;
  ds->df0004 += (0)*factor;
  ds->df00031 += (0)*factor;
  ds->df00022 += (0)*factor;
  ds->df00013 += (0)*factor;
  ds->df00004 += (0)*factor;
"""
    assert gga2x.fourth_derivatives() == reference


def test_gga2x_gradient_function(gga2x):
    reference = f"""
static void
example2x_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0100 += (dp->rhoa*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa*dp->rhob)*factor;
  ds->df0001 += (2*dp->gradb*dp->rhoa*dp->rhob)*factor;
  ds->df00001 += (2*dp->rhoa*dp->rhob)*factor;
}}
"""
    assert gga2x.gradient() == reference


def test_gga2x_hessian(gga2x):
    reference = f"""
static void
example2x_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0100 += (dp->rhoa*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa*dp->rhob)*factor;
  ds->df0001 += (2*dp->gradb*dp->rhoa*dp->rhob)*factor;
  ds->df00001 += (2*dp->rhoa*dp->rhob)*factor;

  // ds->df2000 += (0)*factor;
  ds->df1100 += (pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2))*factor;
  ds->df1010 += (2*dp->grada*dp->rhob)*factor;
  ds->df1001 += (2*dp->gradb*dp->rhob)*factor;
  ds->df10001 += (2*dp->rhob)*factor;
  // ds->df0200 += (0)*factor;
  ds->df0110 += (2*dp->grada*dp->rhoa)*factor;
  ds->df0101 += (2*dp->gradb*dp->rhoa)*factor;
  ds->df01001 += (2*dp->rhoa)*factor;
  ds->df0020 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df0011 += (0)*factor;
  // ds->df00101 += (0)*factor;
  ds->df0002 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df00011 += (0)*factor;
  // ds->df00002 += (0)*factor;

}}
"""
    assert gga2x.hessian() == reference


def test_gga2x_third(gga2x):
    reference = f"""
static void
example2x_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0100 += (dp->rhoa*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa*dp->rhob)*factor;
  ds->df0001 += (2*dp->gradb*dp->rhoa*dp->rhob)*factor;
  ds->df00001 += (2*dp->rhoa*dp->rhob)*factor;

  // ds->df2000 += (0)*factor;
  ds->df1100 += (pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2))*factor;
  ds->df1010 += (2*dp->grada*dp->rhob)*factor;
  ds->df1001 += (2*dp->gradb*dp->rhob)*factor;
  ds->df10001 += (2*dp->rhob)*factor;
  // ds->df0200 += (0)*factor;
  ds->df0110 += (2*dp->grada*dp->rhoa)*factor;
  ds->df0101 += (2*dp->gradb*dp->rhoa)*factor;
  ds->df01001 += (2*dp->rhoa)*factor;
  ds->df0020 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df0011 += (0)*factor;
  // ds->df00101 += (0)*factor;
  ds->df0002 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df00011 += (0)*factor;
  // ds->df00002 += (0)*factor;

  // ds->df3000 += (0)*factor;
  // ds->df2100 += (0)*factor;
  // ds->df2010 += (0)*factor;
  // ds->df2001 += (0)*factor;
  // ds->df20001 += (0)*factor;
  // ds->df1200 += (0)*factor;
  ds->df1110 += (2*dp->grada)*factor;
  ds->df1101 += (2*dp->gradb)*factor;
  ds->df11001 += (2)*factor;
  ds->df1020 += (2*dp->rhob)*factor;
  // ds->df1011 += (0)*factor;
  // ds->df10101 += (0)*factor;
  ds->df1002 += (2*dp->rhob)*factor;
  // ds->df10011 += (0)*factor;
  // ds->df10002 += (0)*factor;
  // ds->df0300 += (0)*factor;
  // ds->df0210 += (0)*factor;
  // ds->df0201 += (0)*factor;
  // ds->df02001 += (0)*factor;
  ds->df0120 += (2*dp->rhoa)*factor;
  // ds->df0111 += (0)*factor;
  // ds->df01101 += (0)*factor;
  ds->df0102 += (2*dp->rhoa)*factor;
  // ds->df01011 += (0)*factor;
  // ds->df01002 += (0)*factor;
  // ds->df0030 += (0)*factor;
  // ds->df0021 += (0)*factor;
  // ds->df00201 += (0)*factor;
  // ds->df0012 += (0)*factor;
  // ds->df00111 += (0)*factor;
  // ds->df00102 += (0)*factor;
  // ds->df0003 += (0)*factor;
  // ds->df00021 += (0)*factor;
  // ds->df00012 += (0)*factor;
  // ds->df00003 += (0)*factor;
}}
"""
    assert gga2x.third() == reference


def test_gga2x_fourth(gga2x):
    reference = f"""
static void
example2x_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0100 += (dp->rhoa*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2)))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa*dp->rhob)*factor;
  ds->df0001 += (2*dp->gradb*dp->rhoa*dp->rhob)*factor;
  ds->df00001 += (2*dp->rhoa*dp->rhob)*factor;

  // ds->df2000 += (0)*factor;
  ds->df1100 += (pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2))*factor;
  ds->df1010 += (2*dp->grada*dp->rhob)*factor;
  ds->df1001 += (2*dp->gradb*dp->rhob)*factor;
  ds->df10001 += (2*dp->rhob)*factor;
  // ds->df0200 += (0)*factor;
  ds->df0110 += (2*dp->grada*dp->rhoa)*factor;
  ds->df0101 += (2*dp->gradb*dp->rhoa)*factor;
  ds->df01001 += (2*dp->rhoa)*factor;
  ds->df0020 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df0011 += (0)*factor;
  // ds->df00101 += (0)*factor;
  ds->df0002 += (2*dp->rhoa*dp->rhob)*factor;
  // ds->df00011 += (0)*factor;
  // ds->df00002 += (0)*factor;

  // ds->df3000 += (0)*factor;
  // ds->df2100 += (0)*factor;
  // ds->df2010 += (0)*factor;
  // ds->df2001 += (0)*factor;
  // ds->df20001 += (0)*factor;
  // ds->df1200 += (0)*factor;
  ds->df1110 += (2*dp->grada)*factor;
  ds->df1101 += (2*dp->gradb)*factor;
  ds->df11001 += (2)*factor;
  ds->df1020 += (2*dp->rhob)*factor;
  // ds->df1011 += (0)*factor;
  // ds->df10101 += (0)*factor;
  ds->df1002 += (2*dp->rhob)*factor;
  // ds->df10011 += (0)*factor;
  // ds->df10002 += (0)*factor;
  // ds->df0300 += (0)*factor;
  // ds->df0210 += (0)*factor;
  // ds->df0201 += (0)*factor;
  // ds->df02001 += (0)*factor;
  ds->df0120 += (2*dp->rhoa)*factor;
  // ds->df0111 += (0)*factor;
  // ds->df01101 += (0)*factor;
  ds->df0102 += (2*dp->rhoa)*factor;
  // ds->df01011 += (0)*factor;
  // ds->df01002 += (0)*factor;
  // ds->df0030 += (0)*factor;
  // ds->df0021 += (0)*factor;
  // ds->df00201 += (0)*factor;
  // ds->df0012 += (0)*factor;
  // ds->df00111 += (0)*factor;
  // ds->df00102 += (0)*factor;
  // ds->df0003 += (0)*factor;
  // ds->df00021 += (0)*factor;
  // ds->df00012 += (0)*factor;
  // ds->df00003 += (0)*factor;

  // ds->df4000 += (0)*factor;
  // ds->df3100 += (0)*factor;
  // ds->df3010 += (0)*factor;
  // ds->df3001 += (0)*factor;
  // ds->df30001 += (0)*factor;
  // ds->df2200 += (0)*factor;
  // ds->df2110 += (0)*factor;
  // ds->df2101 += (0)*factor;
  // ds->df21001 += (0)*factor;
  // ds->df2020 += (0)*factor;
  // ds->df2011 += (0)*factor;
  // ds->df20101 += (0)*factor;
  // ds->df2002 += (0)*factor;
  // ds->df20011 += (0)*factor;
  // ds->df20002 += (0)*factor;
  // ds->df1300 += (0)*factor;
  // ds->df1210 += (0)*factor;
  // ds->df1201 += (0)*factor;
  // ds->df12001 += (0)*factor;
  ds->df1120 += (2)*factor;
  // ds->df1111 += (0)*factor;
  // ds->df11101 += (0)*factor;
  ds->df1102 += (2)*factor;
  // ds->df11011 += (0)*factor;
  // ds->df11002 += (0)*factor;
  // ds->df1030 += (0)*factor;
  // ds->df1021 += (0)*factor;
  // ds->df10201 += (0)*factor;
  // ds->df1012 += (0)*factor;
  // ds->df10111 += (0)*factor;
  // ds->df10102 += (0)*factor;
  // ds->df1003 += (0)*factor;
  // ds->df10021 += (0)*factor;
  // ds->df10012 += (0)*factor;
  // ds->df10003 += (0)*factor;
  // ds->df0400 += (0)*factor;
  // ds->df0310 += (0)*factor;
  // ds->df0301 += (0)*factor;
  // ds->df03001 += (0)*factor;
  // ds->df0220 += (0)*factor;
  // ds->df0211 += (0)*factor;
  // ds->df02101 += (0)*factor;
  // ds->df0202 += (0)*factor;
  // ds->df02011 += (0)*factor;
  // ds->df02002 += (0)*factor;
  // ds->df0130 += (0)*factor;
  // ds->df0121 += (0)*factor;
  // ds->df01201 += (0)*factor;
  // ds->df0112 += (0)*factor;
  // ds->df01111 += (0)*factor;
  // ds->df01102 += (0)*factor;
  // ds->df0103 += (0)*factor;
  // ds->df01021 += (0)*factor;
  // ds->df01012 += (0)*factor;
  // ds->df01003 += (0)*factor;
  // ds->df0040 += (0)*factor;
  // ds->df0031 += (0)*factor;
  // ds->df00301 += (0)*factor;
  // ds->df0022 += (0)*factor;
  // ds->df00211 += (0)*factor;
  // ds->df00202 += (0)*factor;
  // ds->df0013 += (0)*factor;
  // ds->df00121 += (0)*factor;
  // ds->df00112 += (0)*factor;
  // ds->df00103 += (0)*factor;
  // ds->df0004 += (0)*factor;
  // ds->df00031 += (0)*factor;
  // ds->df00022 += (0)*factor;
  // ds->df00013 += (0)*factor;
  // ds->df00004 += (0)*factor;
}}
"""
    assert gga2x.fourth() == reference
