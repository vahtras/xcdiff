import sympy

import pytest

from dftfunc import Functional


pi = sympy.pi


@pytest.fixture
def slater():
    ra, rb = sympy.symbols('dp->rhoa, dp->rhob')
    # defs = "const real PREF= -3.0/4.0*pow(6/M_PI, 1.0/3.0);"
    PREF = -3/4*(6/pi)**(1/3)
    func = Functional(
        'slater',
        ra,
        rb,
        PREF*(ra**(4/3)),
        PREF*(rb**(4/3)),
    )
    return func


def test_header(slater):
    assert slater.header() == """
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
/* fun-Slater.c:
   implementation of Slater functional and its derivatives 
   (c), Pawel Salek, pawsa@theochem.kth.se, aug 2001
   Z. Rinkevicius adapted for open shell systems: energy, first derivatives.
   NOTE:
   this file may seem unnecessarily complex but the structure really pays off
   when implementing multiple functionals depending on different parameters.
*/

/* strictly conform to XOPEN ANSI C standard */
#if !defined(SYS_DEC)
/* XOPEN compliance is missing on old Tru64 4.0E Alphas and pow() prototype
 * is not specified. */
#define _XOPEN_SOURCE          500
#define _XOPEN_SOURCE_EXTENDED 1
#endif
#include <math.h>
#include <stdio.h>
#include "general.h"

#define __CVERSION__

#include "functionals.h"
"""



def test_slater_energy():

    PREF = -3/4*(6/pi)**(1/3)

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

    ra, rb = sympy.symbols('dp->rhoa, dp->rhob')
    # defs = "const real PREF= -3.0/4.0*pow(6/M_PI, 1.0/3.0);"
    PREF = -3/4*(6/pi)**(1/3)
    func = Functional(
        'slater',
        ra,
        rb,
        PREF*(ra**(4/3)),
        PREF*(rb**(4/3)),
    )
    assert func.energy() == reference


def test_slater_gradient():
    reference = f"""
static void
slater_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD)
     ds->df1000 += -1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326)*factor;
  if (dp->rhob>SLATER_THRESHOLD)
     ds->df0100 += -1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326)*factor;
}}
"""

    ra, rb = sympy.symbols('dp->rhoa, dp->rhob')
    PREF = -3/4*(6/pi)**(1/3)
    func = Functional(
        'slater',
        ra,
        rb,
        PREF*(ra**(4/3)),
        PREF*(rb**(4/3)),
    )
    assert func.gradient() == reference


def test_slater_hessian():
    reference = f"""
static void
slater_second(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{
  if (dp->rhoa>SLATER_THRESHOLD)
     ds->df1000 += -1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, 0.33333333333333326)*factor;
     ds->df2000 += -0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhoa, -0.66666666666666674)*factor;
  if (dp->rhob>SLATER_THRESHOLD)
     ds->df0100 += -1.8171205928321394*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, 0.33333333333333326)*factor;
     ds->df0200 += -0.60570686427737963*pow(M_PI, -0.33333333333333331)*pow(dp->rhob, -0.66666666666666674)*factor;
}}
"""

    ra, rb = sympy.symbols('dp->rhoa, dp->rhob')
    PREF = -3/4*(6/pi)**(1/3)
    func = Functional(
        'slater',
        ra,
        rb,
        PREF*(ra**(4/3)),
        PREF*(rb**(4/3)),
    )
    assert func.hessian() == reference
