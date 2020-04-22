import sympy

import pytest

from xcdiff import ExampleFunctional, GGAFunctional, GeneralFunctional


pi = sympy.pi


@pytest.fixture
def example2():
    ra, rb, ga, gb = sympy.symbols("dp->rhoa, dp->rhob, dp->grada, dp->gradb")
    func = ExampleFunctional(
        "Example2",
        ra,
        rb,
        ga,
        gb,
        ra * ga * ga,
        rb * gb * gb,
        const="static const real EPREF= -5e-5;",
        threshold=1e-20,
    )
    return func


@pytest.fixture
def gga2():
    ra, rb, ga, gb = sympy.symbols("dp->rhoa, dp->rhob, dp->grada, dp->gradb")
    func = GGAFunctional("Example2", ra, rb, ga, gb, ra * ga * ga, rb * gb * gb,)
    return func

@pytest.fixture
def gga2x():
    ra, rb, ga, gb, gab = sympy.symbols("dp->rhoa, dp->rhob, dp->grada, dp->gradb, dp->gradab")
    func = GeneralFunctional("Example2x", ra, rb, ga, gb, gab, ra*rb*(ga*ga + gb*gb + 2*gab))
    return func

@pytest.mark.skip
def test_header(example2):
    assert (
        example2.header()
        == """
/*


!
!  Dalton, a molecular electronic structure program
!  Copyright (C) 2018 by the authors of Dalton.
!
!  This program is free software; you can redistribute it and/or
!  modify it under the terms of the GNU Lesser General Public
!  License version 2.1 as published by the Free Software Foundation.
!
!  This program is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!  Lesser General Public License for more details.
!
!  If a copy of the GNU LGPL v2.1 was not distributed with this
!  code, you can obtain one at https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html.
!

!  
*/
/*-*-mode: C; c-indentation-style: "bsd"; c-basic-offset: 4; -*-*/
/* fun-example.c:
   implementation of a test GGA-class functional
   (c) Pawel Salek, pawsa@theochem.kth.se, aug 2001
   NOTE:
   this file may seem unnecessarily complex but the structure really pays off
   when implementing multiple functionals depending on different parameters.
*/

#include <math.h>
#include <stdio.h>
#include "general.h"

#define __CVERSION__

#include "functionals.h"
"""
    )


def test_example2_interface(example2):

    assert (
        example2.interface()
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


def test_example2_read(example2):
    assert (
        example2.read()
        == """
/* IMPLEMENTATION PART */
static integer
example2_read(const char* conf_line)
{
    fun_set_hf_weight(0);
    return 1;
}

/* EXAMPLE2_THRESHOLD Only to avoid numerical problems due to raising 0
 * to a fractional power. */
static const real EXAMPLE2_THRESHOLD = 1e-20;
"""
    )


def test_gga2_read(example2):
    assert (
        example2.read()
        == """
/* IMPLEMENTATION PART */
static integer
example2_read(const char* conf_line)
{
    fun_set_hf_weight(0);
    return 1;
}

/* EXAMPLE2_THRESHOLD Only to avoid numerical problems due to raising 0
 * to a fractional power. */
static const real EXAMPLE2_THRESHOLD = 1e-20;
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


def test_example2x_energy(gga2x):

    reference = f"""

static real
example2x_energy(const FunDensProp* dp)
{{
  return dp->rhoa*dp->rhob*(pow(dp->grada, 2) + 2*dp->gradab + pow(dp->gradb, 2));
}}
"""
    assert gga2x.energy() == reference


def test_example2_gradient(example2):
    reference = f"""
static void
example2_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(pow(dp->grada, 2))*factor;
  ds->df0010 += EPREF*(2*dp->grada*dp->rhoa)*factor;
}}
"""

    assert example2.gradient() == reference


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


def test_gga2x_gradient(gga2x):
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


def test_example2_hessian(example2):
    reference = f"""
static void
example2_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(pow(dp->grada, 2))*factor;
  ds->df0010 += EPREF*(2*dp->grada*dp->rhoa)*factor;
  ds->df1010 += EPREF*(2*dp->grada)*factor;
  ds->df0020 += EPREF*(2*dp->rhoa)*factor;
}}
"""

    assert example2.hessian() == reference


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


def test_example2_third(example2):
    reference = f"""
static void
example2_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(pow(dp->grada, 2))*factor;
  ds->df0010 += EPREF*(2*dp->grada*dp->rhoa)*factor;
  ds->df1010 += EPREF*(2*dp->grada)*factor;
  ds->df0020 += EPREF*(2*dp->rhoa)*factor;

  ds->df1020 += EPREF*(2)*factor;
  ds->df0030 += EPREF*(0)*factor;
}}
"""

    assert example2.third() == reference


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


def test_example2_fourth(example2):
    reference = f"""
static void
example2_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(pow(dp->grada, 2))*factor;
  ds->df0010 += EPREF*(2*dp->grada*dp->rhoa)*factor;
  ds->df1010 += EPREF*(2*dp->grada)*factor;
  ds->df0020 += EPREF*(2*dp->rhoa)*factor;

  ds->df1020 += EPREF*(2)*factor;
  ds->df0030 += EPREF*(0)*factor;

  ds->df4000 += EPREF*(0)*factor;
  ds->df3010 += EPREF*(0)*factor;
  ds->df2020 += EPREF*(0)*factor;
  ds->df1030 += EPREF*(0)*factor;
  ds->df0040 += EPREF*(0)*factor;
}}
"""

    assert example2.fourth() == reference


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
