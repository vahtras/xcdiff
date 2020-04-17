import textwrap

import sympy
from sympy import Symbol


class BaseFunctional:
    def __init__(
            self, name: str, ra: Symbol, rb: Symbol,
            Fa: Symbol, Fb: Symbol, **kwargs
            ):
        self.name_orig = name
        self.name = name.lower()
        self.ra = ra
        self.rb = rb
        self.Fa = Fa
        self.Fb = Fb
        self.const = kwargs.get('const', '')
        self.gga = 0

    def __str__(self):
        return (
            self.header() +
            self.interface() +
            self.read() +
            self.energy() +
            self.gradient() +
            self.hessian() +
            self.third() +
            self.fourth()
        )

    def header(self):
        return textwrap.dedent(
            """
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
        )

    def interface(self):
        return textwrap.dedent(
            f"""
            /* INTERFACE PART */
            static integer {self.name}_isgga(void) {{ return {self.gga}; }}
            static integer {self.name}_read(const char* conf_line);
            static real {self.name}_energy(const FunDensProp* dp);
            static void {self.name}_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
            static void {self.name}_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
            static void {self.name}_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
            static void {self.name}_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

            Functional {self.name_orig}Functional = {{
              "{self.name_orig}",       /* name */
              {self.name}_isgga,   /* gga-corrected */
               3,
              {self.name}_read, 
              NULL,
              {self.name}_energy, 
              {self.name}_first,
              {self.name}_second,
              {self.name}_third,
              {self.name}_fourth
            }};
            """
        )

    def read(self):
        return textwrap.dedent(
            f"""
            /* IMPLEMENTATION PART */
            static integer
            {self.name}_read(const char* conf_line)
            {{
                fun_set_hf_weight(0);
                return 1;
            }}

            /* {self.name.upper()}_THRESHOLD Only to avoid numerical problems due to raising 0
             * to a fractional power. */
            static const real {self.name.upper()}_THRESHOLD = 1e-20;
            """
        )


class Functional(BaseFunctional):

    def energy(self):
        code = textwrap.dedent(
            f"""
            static real
            {self.name}_energy(const FunDensProp* dp)
            {{
              real ea = 0.0, eb = 0.0;
              {self.const}
              if (dp->rhoa >{self.name.upper()}_THRESHOLD)
                  ea = {sympy.ccode(self.Fa)};
              if (dp->rhob >{self.name.upper()}_THRESHOLD)
                  eb = {sympy.ccode(self.Fb)};
              return ea + eb;
            }}
            """
        )
        return code

    def gradient(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              if (dp->rhoa>{self.name.upper()}_THRESHOLD)
                 ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
              if (dp->rhob>{self.name.upper()}_THRESHOLD)
                 ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
            }}
            """
        )

        return code

    def hessian(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              if (dp->rhoa>{self.name.upper()}_THRESHOLD) {{
                 ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
                 ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
                 ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
                 }}
            }}
            """
        )
        return code

    def third(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              if (dp->rhoa>{self.name.upper()}_THRESHOLD) {{
                 ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
                 ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
                 ds->df3000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))}*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
                 ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
                 ds->df0300 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))}*factor;
                 }}
            }}
            """
        )

        return code

    def fourth(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              if (dp->rhoa>{self.name.upper()}_THRESHOLD) {{
                 ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
                 ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
                 ds->df3000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))}*factor;
                 ds->df4000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))}*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
                 ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
                 ds->df0300 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))}*factor;
                 ds->df0400 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.rb))}*factor;
                 }}
            }}
            """
        )

        return code


class ExampleFunctional(Functional):

    def __init__(
            self, name: str, ra: Symbol, rb: Symbol, ga: Symbol, gb: Symbol,
            Fa: Symbol, Fb: Symbol, **kwargs
            ):
        self.name_orig = name
        self.name = name.lower()
        self.ra = ra
        self.rb = rb
        self.ga = ga
        self.gb = gb
        self.Fa = Fa
        self.Fb = Fb
        self.const = kwargs.get('const', '')
        self.gga = 1

    def energy(self):
        code = textwrap.dedent(
            f"""
            {self.const}
            static real
            {self.name}_energy(const FunDensProp* dp)
            {{
              return EPREF*({sympy.ccode(self.Fa)}+{sympy.ccode(self.Fb)});
            }}
            """
        )
        return code

    def gradient(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += EPREF*{sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga))}*factor;
            }}
            """
        )
        return code

    def hessian(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += EPREF*{sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga))}*factor;
              ds->df1010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ra))}*factor;
              ds->df0020 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;
            }}
            """
        )
        return code

    def third(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += EPREF*{sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga))}*factor;
              ds->df1010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ra))}*factor;
              ds->df0020 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;

              ds->df1020 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ra))}*factor;
              ds->df0030 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))}*factor;
            }}
            """
        )

        return code

    def fourth(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += EPREF*{sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga))}*factor;
              ds->df1010 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ra))}*factor;
              ds->df0020 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;

              ds->df1020 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ra))}*factor;
              ds->df0030 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))}*factor;

              ds->df4000 += EPREF*{sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))}*factor;
              ds->df3010 += EPREF*{sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ga))}*factor;
              ds->df2020 += EPREF*{sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga, self.ga))}*factor;
              ds->df1030 += EPREF*{sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga, self.ga))}*factor;
              ds->df0040 += EPREF*{sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga, self.ga))}*factor;
            }}
            """
        )

        return code


class GGAFunctional(Functional):

    def __init__(
            self, name: str, ra: Symbol, rb: Symbol, ga: Symbol, gb: Symbol,
            Fa: Symbol, Fb: Symbol, **kwargs
            ):
        self.name_orig = name
        self.name = name.lower()
        self.ra = ra
        self.rb = rb
        self.ga = ga
        self.gb = gb
        self.Fa = Fa
        self.Fb = Fb
        self.const = kwargs.get('const', '')
        self.gga = 1

    def energy(self):
        code = textwrap.dedent(
            f"""
            {self.const}
            static real
            {self.name}_energy(const FunDensProp* dp)
            {{
              return {sympy.ccode(self.Fa)}+{sympy.ccode(self.Fb)};
            }}
            """
        )
        return code

    def gradient(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += {sympy.ccode(self.Fa.diff(self.ga))}*factor;

              ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
              ds->df0001 += {sympy.ccode(self.Fb.diff(self.gb))}*factor;
            }}
            """
        )
        return code

    def hessian(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += {sympy.ccode(self.Fa.diff(self.ga))}*factor;

              ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
              ds->df1010 += {sympy.ccode(self.Fa.diff(self.ra, self.ga))}*factor;
              ds->df0020 += {sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;

              ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
              ds->df0001 += {sympy.ccode(self.Fb.diff(self.gb))}*factor;

              ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
              ds->df0101 += {sympy.ccode(self.Fb.diff(self.rb, self.gb))}*factor;
              ds->df0002 += {sympy.ccode(self.Fb.diff(self.gb, self.gb))}*factor;
            }}
            """
        )
        return code

    def third(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += {sympy.ccode(self.Fa.diff(self.ga))}*factor;

              ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
              ds->df1010 += {sympy.ccode(self.Fa.diff(self.ra, self.ga))}*factor;
              ds->df0020 += {sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;

              ds->df3000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))}*factor;
              ds->df2010 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga))}*factor;
              ds->df1020 += {sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga))}*factor;
              ds->df0030 += {sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))}*factor;

              ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
              ds->df0001 += {sympy.ccode(self.Fb.diff(self.gb))}*factor;

              ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
              ds->df0101 += {sympy.ccode(self.Fb.diff(self.rb, self.gb))}*factor;
              ds->df0002 += {sympy.ccode(self.Fb.diff(self.gb, self.gb))}*factor;

              ds->df0300 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))}*factor;
              ds->df0201 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb))}*factor;
              ds->df0102 += {sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb))}*factor;
              ds->df0003 += {sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb))}*factor;
            }}
            """
        )

        return code

    def fourth(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += {sympy.ccode(self.Fa.diff(self.ra))}*factor;
              ds->df0010 += {sympy.ccode(self.Fa.diff(self.ga))}*factor;

              ds->df2000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra))}*factor;
              ds->df1010 += {sympy.ccode(self.Fa.diff(self.ra, self.ga))}*factor;
              ds->df0020 += {sympy.ccode(self.Fa.diff(self.ga, self.ga))}*factor;

              ds->df3000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))}*factor;
              ds->df2010 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga))}*factor;
              ds->df1020 += {sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga))}*factor;
              ds->df0030 += {sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))}*factor;

              ds->df4000 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))}*factor;
              ds->df3010 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ga))}*factor;
              ds->df2020 += {sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga, self.ga))}*factor;
              ds->df1030 += {sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga, self.ga))}*factor;
              ds->df0040 += {sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga, self.ga))}*factor;

              ds->df0100 += {sympy.ccode(self.Fb.diff(self.rb))}*factor;
              ds->df0001 += {sympy.ccode(self.Fb.diff(self.gb))}*factor;

              ds->df0200 += {sympy.ccode(self.Fb.diff(self.rb, self.rb))}*factor;
              ds->df0101 += {sympy.ccode(self.Fb.diff(self.rb, self.gb))}*factor;
              ds->df0002 += {sympy.ccode(self.Fb.diff(self.gb, self.gb))}*factor;

              ds->df0300 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))}*factor;
              ds->df0201 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb))}*factor;
              ds->df0102 += {sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb))}*factor;
              ds->df0003 += {sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb))}*factor;

              ds->df0400 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.rb))}*factor;
              ds->df0301 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.gb))}*factor;
              ds->df0202 += {sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb, self.gb))}*factor;
              ds->df0103 += {sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb, self.gb))}*factor;
              ds->df0004 += {sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb, self.gb))}*factor;
            }}
            """
        )

        return code
