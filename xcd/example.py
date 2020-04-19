import textwrap

from sympy import Symbol, ccode

from .func import Functional


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
              return EPREF*({ccode(self.Fa)}+{ccode(self.Fb)});
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
              ds->df1000 += EPREF*({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += EPREF*({ccode(self.Fa.diff(self.ga))})*factor;
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
              ds->df1000 += EPREF*({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += EPREF*({ccode(self.Fa.diff(self.ga))})*factor;
              ds->df1010 += EPREF*({ccode(self.Fa.diff(self.ga, self.ra))})*factor;
              ds->df0020 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga))})*factor;
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
              ds->df1000 += EPREF*({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += EPREF*({ccode(self.Fa.diff(self.ga))})*factor;
              ds->df1010 += EPREF*({ccode(self.Fa.diff(self.ga, self.ra))})*factor;
              ds->df0020 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df1020 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga, self.ra))})*factor;
              ds->df0030 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;
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
              ds->df1000 += EPREF*({ccode(self.Fa.diff(self.ra))})*factor;
              ds->df0010 += EPREF*({ccode(self.Fa.diff(self.ga))})*factor;
              ds->df1010 += EPREF*({ccode(self.Fa.diff(self.ga, self.ra))})*factor;
              ds->df0020 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga))})*factor;

              ds->df1020 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga, self.ra))})*factor;
              ds->df0030 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga, self.ga))})*factor;

              ds->df4000 += EPREF*({ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))})*factor;
              ds->df3010 += EPREF*({ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ga))})*factor;
              ds->df2020 += EPREF*({ccode(self.Fa.diff(self.ra, self.ra, self.ga, self.ga))})*factor;
              ds->df1030 += EPREF*({ccode(self.Fa.diff(self.ra, self.ga, self.ga, self.ga))})*factor;
              ds->df0040 += EPREF*({ccode(self.Fa.diff(self.ga, self.ga, self.ga, self.ga))})*factor;
            }}
            """
        )

        return code
