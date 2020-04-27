import pytest
import sympy

from xcdiff import ExampleFunctional


@pytest.fixture
def example3():
    ra, rb, ga, gb = sympy.symbols("dp->rhoa, dp->rhob, dp->grada, dp->gradb")
    func = ExampleFunctional(
        "Example3",
        ra,
        rb,
        ga,
        gb,
        pow(ga, 1.7),
        pow(gb, 1.7),
        const="static const real EPREF= -5e-2;",
        threshold=1e-20,
    )
    return func



def test_header(example3):
    assert (
        example3.header()
        == """
/*
!
!  Dalton, a molecular electronic structure program
!  Copyright (C) 2020 by the authors of Dalton.
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
/* fun-example3.c:
   implementation of Example3 functional and its derivatives
   (c) Pawel Salek, pawsa@theochem.kth.se, aug 2001
   Z. Rinkevicius adapted for open shell systems: energy, first derivatives.
   NOTE:
   this file may seem unnecessarily complex but the structure really pays off
   when implementing multiple functionals depending on different parameters.

   Derivatives in this file generated with SymPy using xcdiff by Olav Vahtras


*/

#include <math.h>
#include <stdio.h>
#include "general.h"

#define __CVERSION__

#include "functionals.h"
"""
    )


def test_example3_interface(example3):

    assert (
        example3.interface()
        == """
/* INTERFACE PART */
static integer example3_isgga(void) { return 1; }
static integer example3_read(const char* conf_line);
static real example3_energy(const FunDensProp* dp);
static void example3_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
static void example3_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
static void example3_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
static void example3_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

Functional Example3Functional = {
  "Example3",       /* name */
  example3_isgga,   /* gga-corrected */
   3,
  example3_read,
  NULL,
  example3_energy,
  example3_first,
  example3_second,
  example3_third,
  example3_fourth
};
"""
    )


def test_example3_read(example3):
    assert (
        example3.read()
        == """
/* IMPLEMENTATION PART */
static integer
example3_read(const char* conf_line)
{
    fun_set_hf_weight(0);
    return 1;
}

/* EXAMPLE3_THRESHOLD Only to avoid numerical problems due to raising 0
 * to a fractional power. */
static const real EXAMPLE3_THRESHOLD = 1e-20;
"""
    )


def test_example3_energy(example3):

    reference = f"""
static const real EPREF= -5e-2;
static real
example3_energy(const FunDensProp* dp)
{{
  return EPREF*(pow(dp->grada, 1.7)+pow(dp->gradb, 1.7));
}}
"""

    assert example3.energy() == reference


def test_example3_gradient(example3):
    reference = f"""
static void
example3_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(0)*factor;
  ds->df0010 += EPREF*(1.7*pow(dp->grada, 0.69999999999999996))*factor;
}}
"""

    assert example3.gradient() == reference


def test_example3_hessian(example3):
    reference = f"""
static void
example3_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(0)*factor;
  ds->df0010 += EPREF*(1.7*pow(dp->grada, 0.69999999999999996))*factor;
  ds->df1010 += EPREF*(0)*factor;
  ds->df0020 += EPREF*(1.1899999999999999*pow(dp->grada, -0.30000000000000004))*factor;
}}
"""

    assert example3.hessian() == reference


def test_example3_third(example3):
    reference = f"""
static void
example3_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(0)*factor;
  ds->df0010 += EPREF*(1.7*pow(dp->grada, 0.69999999999999996))*factor;
  ds->df1010 += EPREF*(0)*factor;
  ds->df0020 += EPREF*(1.1899999999999999*pow(dp->grada, -0.30000000000000004))*factor;

  ds->df1020 += EPREF*(0)*factor;
  ds->df0030 += EPREF*(-0.35700000000000004*pow(dp->grada, -1.3))*factor;
}}
"""

    assert example3.third() == reference


def test_example3_fourth(example3):
    reference = f"""
static void
example3_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  ds->df1000 += EPREF*(0)*factor;
  ds->df0010 += EPREF*(1.7*pow(dp->grada, 0.69999999999999996))*factor;
  ds->df1010 += EPREF*(0)*factor;
  ds->df0020 += EPREF*(1.1899999999999999*pow(dp->grada, -0.30000000000000004))*factor;

  ds->df1020 += EPREF*(0)*factor;
  ds->df0030 += EPREF*(-0.35700000000000004*pow(dp->grada, -1.3))*factor;

  ds->df4000 += EPREF*(0)*factor;
  ds->df3010 += EPREF*(0)*factor;
  ds->df2020 += EPREF*(0)*factor;
  ds->df1030 += EPREF*(0)*factor;
  ds->df0040 += EPREF*(0.46410000000000007*pow(dp->grada, -2.2999999999999998))*factor;
}}
"""

    assert example3.fourth() == reference
