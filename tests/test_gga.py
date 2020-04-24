import sympy

import pytest

from xcdiff import GGAFunctional


@pytest.fixture
def gga2():
    ra, rb, ga, gb = sympy.symbols("dp->rhoa, dp->rhob, dp->grada, dp->gradb")
    func = GGAFunctional("Example2", ra, rb, ga, gb, ra * ga * ga, rb * gb * gb,)
    return func


def test_gga2_interface(gga2):

    assert (
        gga2.interface()
        == """
/* INTERFACE PART */
static integer example2_isgga(void) { return 1; }
static integer example2_read(const char* conf_line);
static real example2_energy(const FunDensProp* dp);
static void example2_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
static void example2_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
static void example2_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
static void example2_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

Functional Example2Functional = {
  "Example2",       /* name */
  example2_isgga,   /* gga-corrected */
   3,
  example2_read,
  NULL,
  example2_energy,
  example2_first,
  example2_second,
  example2_third,
  example2_fourth
};
"""
    )


def test_example2_energy(gga2):
    reference = f"""

static real
example2_energy(const FunDensProp* dp)
{{
  return pow(dp->grada, 2)*dp->rhoa+pow(dp->gradb, 2)*dp->rhob;
}}
"""
    assert gga2.energy() == reference


def test_gga2_gradient(gga2):
    reference = f"""
static void
example2_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (pow(dp->grada, 2))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa)*factor;

  ds->df0100 += (pow(dp->gradb, 2))*factor;
  ds->df0001 += (2*dp->gradb*dp->rhob)*factor;
}}
"""

    assert gga2.gradient() == reference


def test_gga2_hessian(gga2):
    reference = f"""
static void
example2_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (pow(dp->grada, 2))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa)*factor;

  ds->df2000 += (0)*factor;
  ds->df1010 += (2*dp->grada)*factor;
  ds->df0020 += (2*dp->rhoa)*factor;

  ds->df0100 += (pow(dp->gradb, 2))*factor;
  ds->df0001 += (2*dp->gradb*dp->rhob)*factor;

  ds->df0200 += (0)*factor;
  ds->df0101 += (2*dp->gradb)*factor;
  ds->df0002 += (2*dp->rhob)*factor;
}}
"""

    assert gga2.hessian() == reference


def test_gga2_third(gga2):
    reference = f"""
static void
example2_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (pow(dp->grada, 2))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa)*factor;

  ds->df2000 += (0)*factor;
  ds->df1010 += (2*dp->grada)*factor;
  ds->df0020 += (2*dp->rhoa)*factor;

  ds->df3000 += (0)*factor;
  ds->df2010 += (0)*factor;
  ds->df1020 += (2)*factor;
  ds->df0030 += (0)*factor;

  ds->df0100 += (pow(dp->gradb, 2))*factor;
  ds->df0001 += (2*dp->gradb*dp->rhob)*factor;

  ds->df0200 += (0)*factor;
  ds->df0101 += (2*dp->gradb)*factor;
  ds->df0002 += (2*dp->rhob)*factor;

  ds->df0300 += (0)*factor;
  ds->df0201 += (0)*factor;
  ds->df0102 += (2)*factor;
  ds->df0003 += (0)*factor;
}}
"""
    assert gga2.third() == reference


def test_gga2_fourth(gga2):
    reference = f"""
static void
example2_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += (pow(dp->grada, 2))*factor;
  ds->df0010 += (2*dp->grada*dp->rhoa)*factor;

  ds->df2000 += (0)*factor;
  ds->df1010 += (2*dp->grada)*factor;
  ds->df0020 += (2*dp->rhoa)*factor;

  ds->df3000 += (0)*factor;
  ds->df2010 += (0)*factor;
  ds->df1020 += (2)*factor;
  ds->df0030 += (0)*factor;

  ds->df4000 += (0)*factor;
  ds->df3010 += (0)*factor;
  ds->df2020 += (0)*factor;
  ds->df1030 += (0)*factor;
  ds->df0040 += (0)*factor;

  ds->df0100 += (pow(dp->gradb, 2))*factor;
  ds->df0001 += (2*dp->gradb*dp->rhob)*factor;

  ds->df0200 += (0)*factor;
  ds->df0101 += (2*dp->gradb)*factor;
  ds->df0002 += (2*dp->rhob)*factor;

  ds->df0300 += (0)*factor;
  ds->df0201 += (0)*factor;
  ds->df0102 += (2)*factor;
  ds->df0003 += (0)*factor;

  ds->df0400 += (0)*factor;
  ds->df0301 += (0)*factor;
  ds->df0202 += (0)*factor;
  ds->df0103 += (0)*factor;
  ds->df0004 += (0)*factor;
}}
"""

    assert gga2.fourth() == reference
