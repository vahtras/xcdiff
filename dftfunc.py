import textwrap

import sympy
from sympy import Symbol


class Functional:
    def __init__(
            self, name: str, ra: Symbol, rb: Symbol,
            Fa: Symbol, Fb: Symbol, **kwargs
            ):
        self.name = name
        self.ra = ra
        self.rb = rb
        self.Fa = Fa
        self.Fb = Fb
        self.const = kwargs.get('const', '')

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
            static integer {self.name}_isgga(void) {{ return 0; }}
            static integer {self.name}_read(const char* conf_line);
            static real {self.name}_energy(const FunDensProp* dp);
            static void {self.name}_first(FunFirstFuncDrv *ds,   real fac, const FunDensProp*);
            static void {self.name}_second(FunSecondFuncDrv *ds, real fac, const FunDensProp*);
            static void {self.name}_third(FunThirdFuncDrv *ds,   real fac, const FunDensProp*);
            static void {self.name}_fourth(FunFourthFuncDrv *ds, real fac, const FunDensProp*);

            Functional SlaterFunctional = {{
              "Slater",       /* name */
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
