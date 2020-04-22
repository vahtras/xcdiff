import textwrap

from sympy import Symbol, ccode

from .func import Functional


class GeneralFunctional(Functional):

    def __init__(
            self, name: str, ra: Symbol, rb: Symbol, ga: Symbol, gb: Symbol,
            gab: Symbol, F: Symbol, **kwargs
            ):
        super().__init__(name, ra, rb, None, None, **kwargs)
        self.ga = ga
        self.gb = gb
        self.gab = gab
        self.F = F
        self.gga = 1

    def energy(self):
        code = textwrap.dedent(
            f"""
            {self.const}
            static real
            {self.name}_energy(const FunDensProp* dp)
            {{
              return {ccode(self.F)};
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
              ds->df1000 += ({ccode(self.F.diff(self.ra))})*factor;
              ds->df0100 += ({ccode(self.F.diff(self.rb))})*factor;
              ds->df0010 += ({ccode(self.F.diff(self.ga))})*factor;
              ds->df0001 += ({ccode(self.F.diff(self.gb))})*factor;
              ds->df00001 += ({ccode(self.F.diff(self.gab))})*factor;
            }}
            """
        )
        code = comment_zero_lines(code)
        return code

    def hessian(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += ({ccode(self.F.diff(self.ra))})*factor;
              ds->df0100 += ({ccode(self.F.diff(self.rb))})*factor;
              ds->df0010 += ({ccode(self.F.diff(self.ga))})*factor;
              ds->df0001 += ({ccode(self.F.diff(self.gb))})*factor;
              ds->df00001 += ({ccode(self.F.diff(self.gab))})*factor;

              ds->df2000 += ({ccode(self.F.diff(self.ra, self.ra))})*factor;
              ds->df1100 += ({ccode(self.F.diff(self.ra, self.rb))})*factor;
              ds->df1010 += ({ccode(self.F.diff(self.ra, self.ga))})*factor;
              ds->df1001 += ({ccode(self.F.diff(self.ra, self.gb))})*factor;
              ds->df10001 += ({ccode(self.F.diff(self.ra, self.gab))})*factor;
              ds->df0200 += ({ccode(self.F.diff(self.rb, self.rb))})*factor;
              ds->df0110 += ({ccode(self.F.diff(self.rb, self.ga))})*factor;
              ds->df0101 += ({ccode(self.F.diff(self.rb, self.gb))})*factor;
              ds->df01001 += ({ccode(self.F.diff(self.rb, self.gab))})*factor;
              ds->df0020 += ({ccode(self.F.diff(self.ga, self.ga))})*factor;
              ds->df0011 += ({ccode(self.F.diff(self.ga, self.gb))})*factor;
              ds->df00101 += ({ccode(self.F.diff(self.ga, self.gab))})*factor;
              ds->df0002 += ({ccode(self.F.diff(self.gb, self.gb))})*factor;
              ds->df00011 += ({ccode(self.F.diff(self.gb, self.gab))})*factor;
              ds->df00002 += ({ccode(self.F.diff(self.gab, self.gab))})*factor;

            }}
            """
        )
        code = comment_zero_lines(code)
        return code

    def third(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += ({ccode(self.F.diff(self.ra))})*factor;
              ds->df0100 += ({ccode(self.F.diff(self.rb))})*factor;
              ds->df0010 += ({ccode(self.F.diff(self.ga))})*factor;
              ds->df0001 += ({ccode(self.F.diff(self.gb))})*factor;
              ds->df00001 += ({ccode(self.F.diff(self.gab))})*factor;

              ds->df2000 += ({ccode(self.F.diff(self.ra, self.ra))})*factor;
              ds->df1100 += ({ccode(self.F.diff(self.ra, self.rb))})*factor;
              ds->df1010 += ({ccode(self.F.diff(self.ra, self.ga))})*factor;
              ds->df1001 += ({ccode(self.F.diff(self.ra, self.gb))})*factor;
              ds->df10001 += ({ccode(self.F.diff(self.ra, self.gab))})*factor;
              ds->df0200 += ({ccode(self.F.diff(self.rb, self.rb))})*factor;
              ds->df0110 += ({ccode(self.F.diff(self.rb, self.ga))})*factor;
              ds->df0101 += ({ccode(self.F.diff(self.rb, self.gb))})*factor;
              ds->df01001 += ({ccode(self.F.diff(self.rb, self.gab))})*factor;
              ds->df0020 += ({ccode(self.F.diff(self.ga, self.ga))})*factor;
              ds->df0011 += ({ccode(self.F.diff(self.ga, self.gb))})*factor;
              ds->df00101 += ({ccode(self.F.diff(self.ga, self.gab))})*factor;
              ds->df0002 += ({ccode(self.F.diff(self.gb, self.gb))})*factor;
              ds->df00011 += ({ccode(self.F.diff(self.gb, self.gab))})*factor;
              ds->df00002 += ({ccode(self.F.diff(self.gab, self.gab))})*factor;
              ds->df3000 += ({ccode(self.F.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2100 += ({ccode(self.F.diff(self.ra, self.ra, self.rb))})*factor;
              ds->df2010 += ({ccode(self.F.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df2001 += ({ccode(self.F.diff(self.ra, self.ra, self.gb))})*factor;
              ds->df20001 += ({ccode(self.F.diff(self.ra, self.ra, self.gab))})*factor;
              ds->df1200 += ({ccode(self.F.diff(self.ra, self.rb, self.rb))})*factor;
              ds->df1110 += ({ccode(self.F.diff(self.ra, self.rb, self.ga))})*factor;
              ds->df1101 += ({ccode(self.F.diff(self.ra, self.rb, self.gb))})*factor;
              ds->df11001 += ({ccode(self.F.diff(self.ra, self.rb, self.gab))})*factor;
              ds->df1020 += ({ccode(self.F.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df1011 += ({ccode(self.F.diff(self.ra, self.ga, self.gb))})*factor;
              ds->df10101 += ({ccode(self.F.diff(self.ra, self.ga, self.gab))})*factor;
              ds->df1002 += ({ccode(self.F.diff(self.ra, self.gb, self.gb))})*factor;
              ds->df10011 += ({ccode(self.F.diff(self.ra, self.gb, self.gab))})*factor;
              ds->df10002 += ({ccode(self.F.diff(self.ra, self.gab, self.gab))})*factor;

              ds->df0300 += ({ccode(self.F.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0210 += ({ccode(self.F.diff(self.rb, self.rb, self.ga))})*factor;
              ds->df0201 += ({ccode(self.F.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df02001 += ({ccode(self.F.diff(self.rb, self.rb, self.gab))})*factor;
              ds->df0120 += ({ccode(self.F.diff(self.rb, self.ga, self.ga))})*factor;
              ds->df0111 += ({ccode(self.F.diff(self.rb, self.ga, self.gb))})*factor;
              ds->df01101 += ({ccode(self.F.diff(self.rb, self.ga, self.gab))})*factor;
              ds->df0102 += ({ccode(self.F.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df01011 += ({ccode(self.F.diff(self.rb, self.gb, self.gab))})*factor;
              ds->df01002 += ({ccode(self.F.diff(self.rb, self.gab, self.gab))})*factor;
              ds->df0030 += ({ccode(self.F.diff(self.ga, self.ga, self.ga))})*factor;
              ds->df0021 += ({ccode(self.F.diff(self.ga, self.ga, self.gb))})*factor;
              ds->df00201 += ({ccode(self.F.diff(self.ga, self.ga, self.gab))})*factor;
              ds->df0012 += ({ccode(self.F.diff(self.ga, self.gb, self.gb))})*factor;
              ds->df00111 += ({ccode(self.F.diff(self.ga, self.gb, self.gab))})*factor;
              ds->df00102 += ({ccode(self.F.diff(self.ga, self.gab, self.gab))})*factor;
              ds->df0003 += ({ccode(self.F.diff(self.gb, self.gb, self.gb))})*factor;
              ds->df00021 += ({ccode(self.F.diff(self.gb, self.gb, self.gab))})*factor;
              ds->df00012 += ({ccode(self.F.diff(self.gb, self.gab, self.gab))})*factor;
              ds->df00003 += ({ccode(self.F.diff(self.gab, self.gab, self.gab))})*factor;
            }}
            """
        )

        code = comment_zero_lines(code)
        return code

    def fourth(self):
        code = textwrap.dedent(
            f"""
            static void
            {self.name}_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
            {{
              ds->df1000 += ({ccode(self.F.diff(self.ra))})*factor;
              ds->df0100 += ({ccode(self.F.diff(self.rb))})*factor;
              ds->df0010 += ({ccode(self.F.diff(self.ga))})*factor;
              ds->df0001 += ({ccode(self.F.diff(self.gb))})*factor;
              ds->df00001 += ({ccode(self.F.diff(self.gab))})*factor;

              ds->df2000 += ({ccode(self.F.diff(self.ra, self.ra))})*factor;
              ds->df1100 += ({ccode(self.F.diff(self.ra, self.rb))})*factor;
              ds->df1010 += ({ccode(self.F.diff(self.ra, self.ga))})*factor;
              ds->df1001 += ({ccode(self.F.diff(self.ra, self.gb))})*factor;
              ds->df10001 += ({ccode(self.F.diff(self.ra, self.gab))})*factor;
              ds->df0200 += ({ccode(self.F.diff(self.rb, self.rb))})*factor;
              ds->df0110 += ({ccode(self.F.diff(self.rb, self.ga))})*factor;
              ds->df0101 += ({ccode(self.F.diff(self.rb, self.gb))})*factor;
              ds->df01001 += ({ccode(self.F.diff(self.rb, self.gab))})*factor;
              ds->df0020 += ({ccode(self.F.diff(self.ga, self.ga))})*factor;
              ds->df0011 += ({ccode(self.F.diff(self.ga, self.gb))})*factor;
              ds->df00101 += ({ccode(self.F.diff(self.ga, self.gab))})*factor;
              ds->df0002 += ({ccode(self.F.diff(self.gb, self.gb))})*factor;
              ds->df00011 += ({ccode(self.F.diff(self.gb, self.gab))})*factor;
              ds->df00002 += ({ccode(self.F.diff(self.gab, self.gab))})*factor;
              ds->df3000 += ({ccode(self.F.diff(self.ra, self.ra, self.ra))})*factor;
              ds->df2100 += ({ccode(self.F.diff(self.ra, self.ra, self.rb))})*factor;
              ds->df2010 += ({ccode(self.F.diff(self.ra, self.ra, self.ga))})*factor;
              ds->df2001 += ({ccode(self.F.diff(self.ra, self.ra, self.gb))})*factor;
              ds->df20001 += ({ccode(self.F.diff(self.ra, self.ra, self.gab))})*factor;
              ds->df1200 += ({ccode(self.F.diff(self.ra, self.rb, self.rb))})*factor;
              ds->df1110 += ({ccode(self.F.diff(self.ra, self.rb, self.ga))})*factor;
              ds->df1101 += ({ccode(self.F.diff(self.ra, self.rb, self.gb))})*factor;
              ds->df11001 += ({ccode(self.F.diff(self.ra, self.rb, self.gab))})*factor;
              ds->df1020 += ({ccode(self.F.diff(self.ra, self.ga, self.ga))})*factor;
              ds->df1011 += ({ccode(self.F.diff(self.ra, self.ga, self.gb))})*factor;
              ds->df10101 += ({ccode(self.F.diff(self.ra, self.ga, self.gab))})*factor;
              ds->df1002 += ({ccode(self.F.diff(self.ra, self.gb, self.gb))})*factor;
              ds->df10011 += ({ccode(self.F.diff(self.ra, self.gb, self.gab))})*factor;
              ds->df10002 += ({ccode(self.F.diff(self.ra, self.gab, self.gab))})*factor;

              ds->df0300 += ({ccode(self.F.diff(self.rb, self.rb, self.rb))})*factor;
              ds->df0210 += ({ccode(self.F.diff(self.rb, self.rb, self.ga))})*factor;
              ds->df0201 += ({ccode(self.F.diff(self.rb, self.rb, self.gb))})*factor;
              ds->df02001 += ({ccode(self.F.diff(self.rb, self.rb, self.gab))})*factor;
              ds->df0120 += ({ccode(self.F.diff(self.rb, self.ga, self.ga))})*factor;
              ds->df0111 += ({ccode(self.F.diff(self.rb, self.ga, self.gb))})*factor;
              ds->df01101 += ({ccode(self.F.diff(self.rb, self.ga, self.gab))})*factor;
              ds->df0102 += ({ccode(self.F.diff(self.rb, self.gb, self.gb))})*factor;
              ds->df01011 += ({ccode(self.F.diff(self.rb, self.gb, self.gab))})*factor;
              ds->df01002 += ({ccode(self.F.diff(self.rb, self.gab, self.gab))})*factor;
              ds->df0030 += ({ccode(self.F.diff(self.ga, self.ga, self.ga))})*factor;
              ds->df0021 += ({ccode(self.F.diff(self.ga, self.ga, self.gb))})*factor;
              ds->df00201 += ({ccode(self.F.diff(self.ga, self.ga, self.gab))})*factor;
              ds->df0012 += ({ccode(self.F.diff(self.ga, self.gb, self.gb))})*factor;
              ds->df00111 += ({ccode(self.F.diff(self.ga, self.gb, self.gab))})*factor;
              ds->df00102 += ({ccode(self.F.diff(self.ga, self.gab, self.gab))})*factor;
              ds->df0003 += ({ccode(self.F.diff(self.gb, self.gb, self.gb))})*factor;
              ds->df00021 += ({ccode(self.F.diff(self.gb, self.gb, self.gab))})*factor;
              ds->df00012 += ({ccode(self.F.diff(self.gb, self.gab, self.gab))})*factor;
              ds->df00003 += ({ccode(self.F.diff(self.gab, self.gab, self.gab))})*factor;

              ds->df4000 += ({ccode(self.F.diff(self.ra, self.ra, self.ra, self.ra))})*factor;
              ds->df3100 += ({ccode(self.F.diff(self.ra, self.ra, self.ra, self.rb))})*factor;
              ds->df3010 += ({ccode(self.F.diff(self.ra, self.ra, self.ra, self.ga))})*factor;
              ds->df3001 += ({ccode(self.F.diff(self.ra, self.ra, self.ra, self.gb))})*factor;
              ds->df30001 += ({ccode(self.F.diff(self.ra, self.ra, self.ra, self.gab))})*factor;
              ds->df2200 += ({ccode(self.F.diff(self.ra, self.ra, self.rb, self.rb))})*factor;
              ds->df2110 += ({ccode(self.F.diff(self.ra, self.ra, self.rb, self.ga))})*factor;
              ds->df2101 += ({ccode(self.F.diff(self.ra, self.ra, self.rb, self.gb))})*factor;
              ds->df21001 += ({ccode(self.F.diff(self.ra, self.ra, self.rb, self.gab))})*factor;
              ds->df2020 += ({ccode(self.F.diff(self.ra, self.ra, self.ga, self.ga))})*factor;
              ds->df2011 += ({ccode(self.F.diff(self.ra, self.ra, self.ga, self.gb))})*factor;
              ds->df20101 += ({ccode(self.F.diff(self.ra, self.ra, self.ga, self.gab))})*factor;
              ds->df2002 += ({ccode(self.F.diff(self.ra, self.ra, self.gb, self.gb))})*factor;
              ds->df20011 += ({ccode(self.F.diff(self.ra, self.ra, self.gb, self.gab))})*factor;
              ds->df20002 += ({ccode(self.F.diff(self.ra, self.ra, self.gab, self.gab))})*factor;
              ds->df1300 += ({ccode(self.F.diff(self.ra, self.rb, self.rb, self.rb))})*factor;
              ds->df1210 += ({ccode(self.F.diff(self.ra, self.rb, self.rb, self.ga))})*factor;
              ds->df1201 += ({ccode(self.F.diff(self.ra, self.rb, self.rb, self.gb))})*factor;
              ds->df12001 += ({ccode(self.F.diff(self.ra, self.rb, self.rb, self.gab))})*factor;
              ds->df1120 += ({ccode(self.F.diff(self.ra, self.rb, self.ga, self.ga))})*factor;
              ds->df1111 += ({ccode(self.F.diff(self.ra, self.rb, self.ga, self.gb))})*factor;
              ds->df11101 += ({ccode(self.F.diff(self.ra, self.rb, self.ga, self.gab))})*factor;
              ds->df1102 += ({ccode(self.F.diff(self.ra, self.rb, self.gb, self.gb))})*factor;
              ds->df11011 += ({ccode(self.F.diff(self.ra, self.rb, self.gb, self.gab))})*factor;
              ds->df11002 += ({ccode(self.F.diff(self.ra, self.rb, self.gab, self.gab))})*factor;
              ds->df1030 += ({ccode(self.F.diff(self.ra, self.ga, self.ga, self.ga))})*factor;
              ds->df1021 += ({ccode(self.F.diff(self.ra, self.ga, self.ga, self.gb))})*factor;
              ds->df10201 += ({ccode(self.F.diff(self.ra, self.ga, self.ga, self.gab))})*factor;
              ds->df1012 += ({ccode(self.F.diff(self.ra, self.ga, self.gb, self.gb))})*factor;
              ds->df10111 += ({ccode(self.F.diff(self.ra, self.ga, self.gb, self.gab))})*factor;
              ds->df10102 += ({ccode(self.F.diff(self.ra, self.ga, self.gab, self.gab))})*factor;
              ds->df1003 += ({ccode(self.F.diff(self.ra, self.gb, self.gb, self.gb))})*factor;
              ds->df10021 += ({ccode(self.F.diff(self.ra, self.gb, self.gb, self.gab))})*factor;
              ds->df10012 += ({ccode(self.F.diff(self.ra, self.gb, self.gab, self.gab))})*factor;
              ds->df10003 += ({ccode(self.F.diff(self.ra, self.gab, self.gab, self.gab))})*factor;
              ds->df0400 += ({ccode(self.F.diff(self.rb, self.rb, self.rb, self.rb))})*factor;
              ds->df0310 += ({ccode(self.F.diff(self.rb, self.rb, self.rb, self.ga))})*factor;
              ds->df0301 += ({ccode(self.F.diff(self.rb, self.rb, self.rb, self.gb))})*factor;
              ds->df03001 += ({ccode(self.F.diff(self.rb, self.rb, self.rb, self.gab))})*factor;
              ds->df0220 += ({ccode(self.F.diff(self.rb, self.rb, self.ga, self.ga))})*factor;
              ds->df0211 += ({ccode(self.F.diff(self.rb, self.rb, self.ga, self.gb))})*factor;
              ds->df02101 += ({ccode(self.F.diff(self.rb, self.rb, self.ga, self.gab))})*factor;
              ds->df0202 += ({ccode(self.F.diff(self.rb, self.rb, self.gb, self.gb))})*factor;
              ds->df02011 += ({ccode(self.F.diff(self.rb, self.rb, self.gb, self.gab))})*factor;
              ds->df02002 += ({ccode(self.F.diff(self.rb, self.rb, self.gab, self.gab))})*factor;
              ds->df0130 += ({ccode(self.F.diff(self.rb, self.ga, self.ga, self.ga))})*factor;
              ds->df0121 += ({ccode(self.F.diff(self.rb, self.ga, self.ga, self.gb))})*factor;
              ds->df01201 += ({ccode(self.F.diff(self.rb, self.ga, self.ga, self.gab))})*factor;
              ds->df0112 += ({ccode(self.F.diff(self.rb, self.ga, self.gb, self.gb))})*factor;
              ds->df01111 += ({ccode(self.F.diff(self.rb, self.ga, self.gb, self.gab))})*factor;
              ds->df01102 += ({ccode(self.F.diff(self.rb, self.ga, self.gab, self.gab))})*factor;
              ds->df0103 += ({ccode(self.F.diff(self.rb, self.gb, self.gb, self.gb))})*factor;
              ds->df01021 += ({ccode(self.F.diff(self.rb, self.gb, self.gb, self.gab))})*factor;
              ds->df01012 += ({ccode(self.F.diff(self.rb, self.gb, self.gab, self.gab))})*factor;
              ds->df01003 += ({ccode(self.F.diff(self.rb, self.gab, self.gab, self.gab))})*factor;
              ds->df0040 += ({ccode(self.F.diff(self.ga, self.ga, self.ga, self.ga))})*factor;
              ds->df0031 += ({ccode(self.F.diff(self.ga, self.ga, self.ga, self.gb))})*factor;
              ds->df00301 += ({ccode(self.F.diff(self.ga, self.ga, self.ga, self.gab))})*factor;
              ds->df0022 += ({ccode(self.F.diff(self.ga, self.ga, self.gb, self.gb))})*factor;
              ds->df00211 += ({ccode(self.F.diff(self.ga, self.ga, self.gb, self.gab))})*factor;
              ds->df00202 += ({ccode(self.F.diff(self.ga, self.ga, self.gab, self.gab))})*factor;
              ds->df0013 += ({ccode(self.F.diff(self.ga, self.gb, self.gb, self.gb))})*factor;
              ds->df00121 += ({ccode(self.F.diff(self.ga, self.gb, self.gb, self.gab))})*factor;
              ds->df00112 += ({ccode(self.F.diff(self.ga, self.gb, self.gab, self.gab))})*factor;
              ds->df00103 += ({ccode(self.F.diff(self.ga, self.gab, self.gab, self.gab))})*factor;
              ds->df0004 += ({ccode(self.F.diff(self.gb, self.gb, self.gb, self.gb))})*factor;
              ds->df00031 += ({ccode(self.F.diff(self.gb, self.gb, self.gb, self.gab))})*factor;
              ds->df00022 += ({ccode(self.F.diff(self.gb, self.gb, self.gab, self.gab))})*factor;
              ds->df00013 += ({ccode(self.F.diff(self.gb, self.gab, self.gab, self.gab))})*factor;
              ds->df00004 += ({ccode(self.F.diff(self.gab, self.gab, self.gab, self.gab))})*factor;
            }}
            """
        )

        code = comment_zero_lines(code)
        return code


def comment_zero_lines(code):
    import re
    return re.sub(r'\n(\s*)(.*)= \(0\)\*factor;', r'\n\1// \2= (0)*factor;', code)
