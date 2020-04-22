import textwrap

from sympy import Symbol, ccode

from .func import Functional


class GGAFunctional(Functional):

    def __init__(
            self, name: str, ra: Symbol, rb: Symbol, ga: Symbol, gb: Symbol,
            Fa: Symbol, Fb: Symbol, **kwargs
            ):
        super().__init__(name, ra, rb, Fa, Fb, **kwargs)
        self.ga = ga
        self.gb = gb
        self.gga = 1

    def energy(self):
        code = textwrap.dedent(
            f"""
            {self.const}
            static real
            {self.name}_energy(const FunDensProp* dp)
            {{
              return {ccode(self.Fa)}+{ccode(self.Fb)};
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
              ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({ccode(self.Fa.diff(self.ga))})*factor;

              ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({ccode(self.Fb.diff(self.gb))})*factor;
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
              ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({ccode(self.Fb.diff(self.gb, self.gb))})*factor;
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
              ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df3000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2010 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df1020 += ({ccode(self.Fa.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df0030 += ({ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;

              ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({ccode(self.Fb.diff(self.gb, self.gb))})*factor;

              ds->df0300 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0201 += ({ccode(self.Fb.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df0102 += ({ccode(self.Fb.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df0003 += ({ccode(self.Fb.diff(self.gb, self.gb, self.gb))})*factor;
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
              ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += ({ccode(self.Fa.diff(self.ga))})*factor;

              ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
              ds->df1010 += ({ccode(self.Fa.diff(self.ra, self.ga))})*factor;
              ds->df0020 += ({ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df3000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2010 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df1020 += ({ccode(self.Fa.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df0030 += ({ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;

              ds->df4000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))})*factor;
              ds->df3010 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ga))})*factor;
              ds->df2020 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ga, self.ga))})*factor;
              ds->df1030 += ({ccode(self.Fa.diff(self.ra, self.ga, self.ga, self.ga))})*factor;
              ds->df0040 += ({ccode(self.Fa.diff(self.ga, self.ga, self.ga, self.ga))})*factor;

              ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
              ds->df0001 += ({ccode(self.Fb.diff(self.gb))})*factor;

              ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
              ds->df0101 += ({ccode(self.Fb.diff(self.rb, self.gb))})*factor;
              ds->df0002 += ({ccode(self.Fb.diff(self.gb, self.gb))})*factor;

              ds->df0300 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0201 += ({ccode(self.Fb.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df0102 += ({ccode(self.Fb.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df0003 += ({ccode(self.Fb.diff(self.gb, self.gb, self.gb))})*factor;

              ds->df0400 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.rb))})*factor;
              ds->df0301 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.gb))})*factor;
              ds->df0202 += ({ccode(self.Fb.diff(self.rb, self.rb, self.gb, self.gb))})*factor;
              ds->df0103 += ({ccode(self.Fb.diff(self.rb, self.gb, self.gb, self.gb))})*factor;
              ds->df0004 += ({ccode(self.Fb.diff(self.gb, self.gb, self.gb, self.gb))})*factor;
            }}
            """
        )

        return code
