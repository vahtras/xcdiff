import textwrap

from sympy import Symbol
import sympy

from .func import Functional


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
              ds->df1000 += ({sympy.ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({sympy.ccode(self.Fa.diff(self.ga))})*factor;

              ds->df0100 += ({sympy.ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({sympy.ccode(self.Fb.diff(self.gb))})*factor;
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
              ds->df1000 += ({sympy.ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({sympy.ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df0100 += ({sympy.ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({sympy.ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb))})*factor;
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
              ds->df1000 += ({sympy.ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({sympy.ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df3000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df1020 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df0030 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;

              ds->df0100 += ({sympy.ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({sympy.ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb))})*factor;

              ds->df0300 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0201 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df0102 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df0003 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb))})*factor;
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
              ds->df1000 += ({sympy.ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({sympy.ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df3000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df1020 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df0030 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;

              ds->df4000 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))})*factor;
              ds->df3010 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ga))})*factor;
              ds->df2020 += ({sympy.ccode(self.Fa.diff(self.ra, self.ra, self.ga, self.ga))})*factor;
              ds->df1030 += ({sympy.ccode(self.Fa.diff(self.ra, self.ga, self.ga, self.ga))})*factor;
              ds->df0040 += ({sympy.ccode(self.Fa.diff(self.ga, self.ga, self.ga, self.ga))})*factor;

              ds->df0100 += ({sympy.ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({sympy.ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb))})*factor;

              ds->df0300 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0201 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df0102 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df0003 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb))})*factor;

              ds->df0400 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.rb))})*factor;
              ds->df0301 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.gb))})*factor;
              ds->df0202 += ({sympy.ccode(self.Fb.diff(self.rb, self.rb, self.gb, self.gb))})*factor;
              ds->df0103 += ({sympy.ccode(self.Fb.diff(self.rb, self.gb, self.gb, self.gb))})*factor;
              ds->df0004 += ({sympy.ccode(self.Fb.diff(self.gb, self.gb, self.gb, self.gb))})*factor;
            }}
            """
        )

        return code
