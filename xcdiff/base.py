import textwrap

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
        self.threshold = kwargs.get('threshold')
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
            f"""
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
            /* fun-{self.name}.c:
               implementation of {self.name_orig} functional and its derivatives
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
        retstr =  textwrap.dedent(
            f"""
            /* IMPLEMENTATION PART */
            static integer
            {self.name}_read(const char* conf_line)
            {{
                fun_set_hf_weight(0);
                return 1;
            }}
            """
        )
        if self.threshold:
            retstr += textwrap.dedent(
                f"""
                /* {self.name.upper()}_THRESHOLD Only to avoid numerical problems due to raising 0
                 * to a fractional power. */
                static const real {self.name.upper()}_THRESHOLD = {self.threshold};
                """
            )
        return retstr
