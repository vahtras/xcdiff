import sympy

import pytest

from xcdiff import ExampleFunctional


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
        info='   Other info here'
    )
    return func


def test_header(example2):
    assert (
        example2.header()
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
/* fun-example2.c:
   implementation of Example2 functional and its derivatives
   (c) Pawel Salek, pawsa@theochem.kth.se, aug 2001
   Z. Rinkevicius adapted for open shell systems: energy, first derivatives.
   NOTE:
   this file may seem unnecessarily complex but the structure really pays off
   when implementing multiple functionals depending on different parameters.

   Other info here
*/

#include <math.h>
#include <stdio.h>
#include "general.h"

#define __CVERSION__

#include "functionals.h"
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
