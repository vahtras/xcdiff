import sympy

import pytest

from xcdiff import Functional


pi = sympy.pi


@pytest.fixture
def slater():
    ra, rb = sympy.symbols("dp->rhoa, dp->rhob")
    # defs = "const real PREF= -3.0/4.0*pow(6/M_PI, 1.0/3.0);"
    PREF = -3 / 4 * (6 / pi) ** (1 / 3)
    func = Functional(
        "Slater", ra, rb, PREF * (ra ** (4 / 3)), PREF * (rb ** (4 / 3)),
        threshold=1e-20,
        info="""
    Functional(
        "Slater", ra, rb, PREF * (ra ** (4 / 3)), PREF * (rb ** (4 / 3)),
        threshold=1e-20,
    )
"""
    )
    return func


def test_header(slater):
    assert (
        slater.header()
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
/* fun-slater.c:
   implementation of Slater functional and its derivatives
   (c) Pawel Salek, pawsa@theochem.kth.se, aug 2001
   Z. Rinkevicius adapted for open shell systems: energy, first derivatives.
   NOTE:
   this file may seem unnecessarily complex but the structure really pays off
   when implementing multiple functionals depending on different parameters.

   Derivatives in this file generated with SymPy using xcdiff by Olav Vahtras


    Functional(
        "Slater", ra, rb, PREF * (ra ** (4 / 3)), PREF * (rb ** (4 / 3)),
        threshold=1e-20,
    )

*/

#include <math.h>
#include <stdio.h>
#include "general.h"

#define __CVERSION__

#include "functionals.h"
"""
    )


def test_slater_interface(slater):

    assert (
        slater.interface()
        == """
/* INTERFACE PART */
static integer slater_isgga(void) { return 0; }
static integer slater_read(const char* conf_line);
static real slater_energy(const FunDensProp* dp);
static void slater_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
static void slater_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
static void slater_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
static void slater_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

Functional SlaterFunctional = {
  "Slater",       /* name */
  slater_isgga,   /* gga-corrected */
   3,
  slater_read,
  NULL,
  slater_energy,
  slater_first,
  slater_second,
  slater_third,
  slater_fourth
};
"""
    )


def test_slater_read(slater):
    assert (
        slater.read()
        == """
/* IMPLEMENTATION PART */
static integer
slater_read(const char* conf_line)
{
    fun_set_hf_weight(0);
    return 1;
}

/* SLATER_THRESHOLD Only to avoid numerical problems due to raising 0
 * to a fractional power. */
static const real SLATER_THRESHOLD = 1e-20;
"""
    )


def test_slater_energy(slater):

    reference = f"""
static real
slater_energy(const FunDensProp* dp)
{{
  real ea = 0.0, eb = 0.0;

  if (dp->rhoa >SLATER_THRESHOLD)
      ea = -1.3628404446241047*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 1.3333333333333333);
  if (dp->rhob >SLATER_THRESHOLD)
      eb = -1.3628404446241047*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 1.3333333333333333);
  return ea + eb;
}}
"""

    assert slater.energy() == reference


def test_slater_gradient(slater):
    reference = f"""
static void
slater_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD)
     ds->df1000 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326))*factor;
  if (dp->rhob>SLATER_THRESHOLD)
     ds->df0100 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326))*factor;
}}
"""

    assert slater.gradient() == reference


def test_slater_hessian(slater):
    reference = f"""
static void
slater_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD) {{
     ds->df1000 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326))*factor;
     ds->df2000 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -0.66666666666666674))*factor;
     }}
  if (dp->rhob>SLATER_THRESHOLD) {{
     ds->df0100 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326))*factor;
     ds->df0200 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -0.66666666666666674))*factor;
     }}
}}
"""

    assert slater.hessian() == reference


def test_slater_third(slater):
    reference = f"""
static void
slater_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD) {{
     ds->df1000 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326))*factor;
     ds->df2000 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -0.66666666666666674))*factor;
     ds->df3000 += (0.40380457618491983*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -1.6666666666666667))*factor;
     }}
  if (dp->rhob>SLATER_THRESHOLD) {{
     ds->df0100 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326))*factor;
     ds->df0200 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -0.66666666666666674))*factor;
     ds->df0300 += (0.40380457618491983*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -1.6666666666666667))*factor;
     }}
}}
"""

    assert slater.third() == reference


def test_slater_fourth(slater):
    reference = f"""
static void
slater_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD) {{
     ds->df1000 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326))*factor;
     ds->df2000 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -0.66666666666666674))*factor;
     ds->df3000 += (0.40380457618491983*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -1.6666666666666667))*factor;
     ds->df4000 += (-0.67300762697486638*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -2.666666666666667))*factor;
     }}
  if (dp->rhob>SLATER_THRESHOLD) {{
     ds->df0100 += (-1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326))*factor;
     ds->df0200 += (-0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -0.66666666666666674))*factor;
     ds->df0300 += (0.40380457618491983*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -1.6666666666666667))*factor;
     ds->df0400 += (-0.67300762697486638*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -2.666666666666667))*factor;
     }}
}}
"""

    assert slater.fourth() == reference
