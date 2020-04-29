import textwrap
import functools

from sympy import Symbol, ccode

from .func import Functional
from .prof import Timer

timeme = Timer()


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
        self.args = [ra, rb, ga, gb, gab]

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

    def first_derivatives(self):
        Fa, Fb, Fα, Fβ, Fγ = self.grad()
        code = f"""
  ds->df1000 += ({ccode(Fa)})*factor;
  ds->df0100 += ({ccode(Fb)})*factor;
  ds->df0010 += ({ccode(Fα)})*factor;
  ds->df0001 += ({ccode(Fβ)})*factor;
  ds->df00001 += ({ccode(Fγ)})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def grad(self):
        a, b, α, β, γ = self.ra, self.rb, self.ga, self.gb, self.gab
        Fa = self.F.diff(a)
        Fb = self.F.diff(b)
        Fα = self.F.diff(α)
        Fβ = self.F.diff(β)
        Fγ = self.F.diff(γ)
        return (Fa, Fb, Fα, Fβ, Fγ)

    @timeme
    def second_derivatives(self):
        Faa, Fab, Faα, Faβ, Faγ,\
            Fbb, Fbα, Fbβ, Fbγ, Fαα, Fαβ, Fαγ, Fββ, Fβγ, Fγγ = self.hess()

        code = f"""
  ds->df2000 += ({ccode(Faa)})*factor;
  ds->df1100 += ({ccode(Fab)})*factor;
  ds->df1010 += ({ccode(Faα)})*factor;
  ds->df1001 += ({ccode(Faβ)})*factor;
  ds->df10001 += ({ccode(Faγ)})*factor;
  ds->df0200 += ({ccode(Fbb)})*factor;
  ds->df0110 += ({ccode(Fbα)})*factor;
  ds->df0101 += ({ccode(Fbβ)})*factor;
  ds->df01001 += ({ccode(Fbγ)})*factor;
  ds->df0020 += ({ccode(Fαα)})*factor;
  ds->df0011 += ({ccode(Fαβ)})*factor;
  ds->df00101 += ({ccode(Fαγ)})*factor;
  ds->df0002 += ({ccode(Fββ)})*factor;
  ds->df00011 += ({ccode(Fβγ)})*factor;
  ds->df00002 += ({ccode(Fγγ)})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def hess(self):
        Fa, Fb, Fα, Fβ, Fγ = self.grad()
        a, b, α, β, γ = self.ra, self.rb, self.ga, self.gb, self.gab
        Faa = Fa.diff(a)
        Fab = Fa.diff(b)
        Faα = Fa.diff(α)
        Faβ = Fa.diff(β)
        Faγ = Fa.diff(γ)
        Fbb = Fb.diff(b)
        Fbα = Fb.diff(α)
        Fbβ = Fb.diff(β)
        Fbγ = Fb.diff(γ)
        Fαα = Fα.diff(α)
        Fαβ = Fα.diff(β)
        Fαγ = Fα.diff(γ)
        Fββ = Fβ.diff(β)
        Fβγ = Fβ.diff(γ)
        Fγγ = Fγ.diff(γ)

        return (
            Faa, Fab, Faα, Faβ, Faγ,
            Fbb, Fbα, Fbβ, Fbγ,
            Fαα, Fαβ, Fαγ,
            Fββ, Fβγ,
            Fγγ,
        )

    @timeme
    def third_derivatives(self):
        (
            Faaa, Faab, Faaα, Faaβ, Faaγ, Fabb,
            Fabα, Fabβ, Fabγ, Faαα, Faαβ, Faαγ,
            Faββ, Faβγ, Faγγ, Fbbb, Fbbα, Fbbβ,
            Fbbγ, Fbαα, Fbαβ, Fbαγ, Fbββ, Fbβγ,
            Fbγγ, Fααα, Fααβ, Fααγ, Fαββ, Fαβγ,
            Fαγγ, Fβββ, Fββγ, Fβγγ, Fγγγ
        ) = self.kolm()

        code = f"""
  ds->df3000 += ({ccode(Faaa)})*factor;
  ds->df2100 += ({ccode(Faab)})*factor;
  ds->df2010 += ({ccode(Faaα)})*factor;
  ds->df2001 += ({ccode(Faaβ)})*factor;
  ds->df20001 += ({ccode(Faaγ)})*factor;
  ds->df1200 += ({ccode(Fabb)})*factor;
  ds->df1110 += ({ccode(Fabα)})*factor;
  ds->df1101 += ({ccode(Fabβ)})*factor;
  ds->df11001 += ({ccode(Fabγ)})*factor;
  ds->df1020 += ({ccode(Faαα)})*factor;
  ds->df1011 += ({ccode(Faαβ)})*factor;
  ds->df10101 += ({ccode(Faαγ)})*factor;
  ds->df1002 += ({ccode(Faββ)})*factor;
  ds->df10011 += ({ccode(Faβγ)})*factor;
  ds->df10002 += ({ccode(Faγγ)})*factor;
  ds->df0300 += ({ccode(Fbbb)})*factor;
  ds->df0210 += ({ccode(Fbbα)})*factor;
  ds->df0201 += ({ccode(Fbbβ)})*factor;
  ds->df02001 += ({ccode(Fbbγ)})*factor;
  ds->df0120 += ({ccode(Fbαα)})*factor;
  ds->df0111 += ({ccode(Fbαβ)})*factor;
  ds->df01101 += ({ccode(Fbαγ)})*factor;
  ds->df0102 += ({ccode(Fbββ)})*factor;
  ds->df01011 += ({ccode(Fbβγ)})*factor;
  ds->df01002 += ({ccode(Fbγγ)})*factor;
  ds->df0030 += ({ccode(Fααα)})*factor;
  ds->df0021 += ({ccode(Fααβ)})*factor;
  ds->df00201 += ({ccode(Fααγ)})*factor;
  ds->df0012 += ({ccode(Fαββ)})*factor;
  ds->df00111 += ({ccode(Fαβγ)})*factor;
  ds->df00102 += ({ccode(Fαγγ)})*factor;
  ds->df0003 += ({ccode(Fβββ)})*factor;
  ds->df00021 += ({ccode(Fββγ)})*factor;
  ds->df00012 += ({ccode(Fβγγ)})*factor;
  ds->df00003 += ({ccode(Fγγγ)})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def kolm(self):
        Fa, Fb, Fα, Fβ, Fγ = self.grad()

        (
            Faa, Fab, Faα, Faβ, Faγ,
            Fbb, Fbα, Fbβ, Fbγ,
            Fαα, Fαβ, Fαγ,
            Fββ, Fβγ,
            Fγγ,
        ) = self.hess()

        a, b, α, β, γ = self.ra, self.rb, self.ga, self.gb, self.gab
        Faaa = Faa.diff(a)
        Faab = Faa.diff(b)
        Faaα = Faa.diff(α)
        Faaβ = Faa.diff(β)
        Faaγ = Faa.diff(γ)
        Fabb = Fab.diff(b)
        Fabα = Fab.diff(α)
        Fabβ = Fab.diff(β)
        Fabγ = Fab.diff(γ)
        Faαα = Faα.diff(α)
        Faαβ = Faα.diff(β)
        Faαγ = Faα.diff(γ)
        Faββ = Faβ.diff(β)
        Faβγ = Faβ.diff(γ)
        Faγγ = Faγ.diff(γ)
        Fbbb = Fbb.diff(b)
        Fbbα = Fbb.diff(α)
        Fbbβ = Fbb.diff(β)
        Fbbγ = Fbb.diff(γ)
        Fbαα = Fbα.diff(α)
        Fbαβ = Fbα.diff(β)
        Fbαγ = Fbα.diff(γ)
        Fbββ = Fbβ.diff(β)
        Fbβγ = Fbβ.diff(γ)
        Fbγγ = Fbγ.diff(γ)
        Fααα = Fαα.diff(α)
        Fααβ = Fαα.diff(β)
        Fααγ = Fαα.diff(γ)
        Fαββ = Fαβ.diff(β)
        Fαβγ = Fαβ.diff(γ)
        Fαγγ = Fαγ.diff(γ)
        Fβββ = Fββ.diff(β)
        Fββγ = Fββ.diff(γ)
        Fβγγ = Fβγ.diff(γ)
        Fγγγ = Fγγ.diff(γ)

        return (
            Faaa,
            Faab,
            Faaα,
            Faaβ,
            Faaγ,
            Fabb,
            Fabα,
            Fabβ,
            Fabγ,
            Faαα,
            Faαβ,
            Faαγ,
            Faββ,
            Faβγ,
            Faγγ,
            Fbbb,
            Fbbα,
            Fbbβ,
            Fbbγ,
            Fbαα,
            Fbαβ,
            Fbαγ,
            Fbββ,
            Fbβγ,
            Fbγγ,
            Fααα,
            Fααβ,
            Fααγ,
            Fαββ,
            Fαβγ,
            Fαγγ,
            Fβββ,
            Fββγ,
            Fβγγ,
            Fγγγ,
        )

    @timeme
    def fourth_derivatives(self):
        (
            Faaaa,
            Faaab,
            Faaaα,
            Faaaβ,
            Faaaγ,
            Faabb,
            Faabα,
            Faabβ,
            Faabγ,
            Faaαα,
            Faaαβ,
            Faaαγ,
            Faaββ,
            Faaβγ,
            Faaγγ,
            Fabbb,
            Fabbα,
            Fabbβ,
            Fabbγ,
            Fabαα,
            Fabαβ,
            Fabαγ,
            Fabββ,
            Fabβγ,
            Fabγγ,
            Faααα,
            Faααβ,
            Faααγ,
            Faαββ,
            Faαβγ,
            Faαγγ,
            Faβββ,
            Faββγ,
            Faβγγ,
            Faγγγ,
            Fbbbb,
            Fbbbα,
            Fbbbβ,
            Fbbbγ,
            Fbbαα,
            Fbbαβ,
            Fbbαγ,
            Fbbββ,
            Fbbβγ,
            Fbbγγ,
            Fbααα,
            Fbααβ,
            Fbααγ,
            Fbαββ,
            Fbαβγ,
            Fbαγγ,
            Fbβββ,
            Fbββγ,
            Fbβγγ,
            Fbγγγ,
            Fαααα,
            Fαααβ,
            Fαααγ,
            Fααββ,
            Fααβγ,
            Fααγγ,
            Fαβββ,
            Fαββγ,
            Fαβγγ,
            Fαγγγ,
            Fββββ,
            Fβββγ,
            Fββγγ,
            Fβγγγ,
            Fγγγγ,
        ) = self.neli()

        code = f"""
  ds->df4000 += ({ccode(Faaaa)})*factor;
  ds->df3100 += ({ccode(Faaab)})*factor;
  ds->df3010 += ({ccode(Faaaα)})*factor;
  ds->df3001 += ({ccode(Faaaβ)})*factor;
  ds->df30001 += ({ccode(Faaaγ)})*factor;
  ds->df2200 += ({ccode(Faabb)})*factor;
  ds->df2110 += ({ccode(Faabα)})*factor;
  ds->df2101 += ({ccode(Faabβ)})*factor;
  ds->df21001 += ({ccode(Faabγ)})*factor;
  ds->df2020 += ({ccode(Faaαα)})*factor;
  ds->df2011 += ({ccode(Faaαβ)})*factor;
  ds->df20101 += ({ccode(Faaαγ)})*factor;
  ds->df2002 += ({ccode(Faaββ)})*factor;
  ds->df20011 += ({ccode(Faaβγ)})*factor;
  ds->df20002 += ({ccode(Faaγγ)})*factor;
  ds->df1300 += ({ccode(Fabbb)})*factor;
  ds->df1210 += ({ccode(Fabbα)})*factor;
  ds->df1201 += ({ccode(Fabbβ)})*factor;
  ds->df12001 += ({ccode(Fabbγ)})*factor;
  ds->df1120 += ({ccode(Fabαα)})*factor;
  ds->df1111 += ({ccode(Fabαβ)})*factor;
  ds->df11101 += ({ccode(Fabαγ)})*factor;
  ds->df1102 += ({ccode(Fabββ)})*factor;
  ds->df11011 += ({ccode(Fabβγ)})*factor;
  ds->df11002 += ({ccode(Fabγγ)})*factor;
  ds->df1030 += ({ccode(Faααα)})*factor;
  ds->df1021 += ({ccode(Faααβ)})*factor;
  ds->df10201 += ({ccode(Faααγ)})*factor;
  ds->df1012 += ({ccode(Faαββ)})*factor;
  ds->df10111 += ({ccode(Faαβγ)})*factor;
  ds->df10102 += ({ccode(Faαγγ)})*factor;
  ds->df1003 += ({ccode(Faβββ)})*factor;
  ds->df10021 += ({ccode(Faββγ)})*factor;
  ds->df10012 += ({ccode(Faβγγ)})*factor;
  ds->df10003 += ({ccode(Faγγγ)})*factor;
  ds->df0400 += ({ccode(Fbbbb)})*factor;
  ds->df0310 += ({ccode(Fbbbα)})*factor;
  ds->df0301 += ({ccode(Fbbbβ)})*factor;
  ds->df03001 += ({ccode(Fbbbγ)})*factor;
  ds->df0220 += ({ccode(Fbbαα)})*factor;
  ds->df0211 += ({ccode(Fbbαβ)})*factor;
  ds->df02101 += ({ccode(Fbbαγ)})*factor;
  ds->df0202 += ({ccode(Fbbββ)})*factor;
  ds->df02011 += ({ccode(Fbbβγ)})*factor;
  ds->df02002 += ({ccode(Fbbγγ)})*factor;
  ds->df0130 += ({ccode(Fbααα)})*factor;
  ds->df0121 += ({ccode(Fbααβ)})*factor;
  ds->df01201 += ({ccode(Fbααγ)})*factor;
  ds->df0112 += ({ccode(Fbαββ)})*factor;
  ds->df01111 += ({ccode(Fbαβγ)})*factor;
  ds->df01102 += ({ccode(Fbαγγ)})*factor;
  ds->df0103 += ({ccode(Fbβββ)})*factor;
  ds->df01021 += ({ccode(Fbββγ)})*factor;
  ds->df01012 += ({ccode(Fbβγγ)})*factor;
  ds->df01003 += ({ccode(Fbγγγ)})*factor;
  ds->df0040 += ({ccode(Fαααα)})*factor;
  ds->df0031 += ({ccode(Fαααβ)})*factor;
  ds->df00301 += ({ccode(Fαααγ)})*factor;
  ds->df0022 += ({ccode(Fααββ)})*factor;
  ds->df00211 += ({ccode(Fααβγ)})*factor;
  ds->df00202 += ({ccode(Fααγγ)})*factor;
  ds->df0013 += ({ccode(Fαβββ)})*factor;
  ds->df00121 += ({ccode(Fαββγ)})*factor;
  ds->df00112 += ({ccode(Fαβγγ)})*factor;
  ds->df00103 += ({ccode(Fαγγγ)})*factor;
  ds->df0004 += ({ccode(Fββββ)})*factor;
  ds->df00031 += ({ccode(Fβββγ)})*factor;
  ds->df00022 += ({ccode(Fββγγ)})*factor;
  ds->df00013 += ({ccode(Fβγγγ)})*factor;
  ds->df00004 += ({ccode(Fγγγγ)})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def neli(self):
        (
            Faaa, Faab, Faaα, Faaβ, Faaγ, Fabb, Fabα, Fabβ, Fabγ, Faαα, Faαβ, Faαγ,
            Faββ, Faβγ, Faγγ, Fbbb, Fbbα, Fbbβ, Fbbγ, Fbαα, Fbαβ, Fbαγ, Fbββ, Fbβγ,
            Fbγγ, Fααα, Fααβ, Fααγ, Fαββ, Fαβγ, Fαγγ, Fβββ, Fββγ, Fβγγ, Fγγγ,
        ) = self.kolm()

        a, b, α, β, γ = self.ra, self.rb, self.ga, self.gb, self.gab
        Faaaa = Faaa.diff(a)
        Faaab = Faaa.diff(b)
        Faaaα = Faaa.diff(α)
        Faaaβ = Faaa.diff(β)
        Faaaγ = Faaa.diff(γ)
        Faabb = Faab.diff(b)
        Faabα = Faab.diff(α)
        Faabβ = Faab.diff(β)
        Faabγ = Faab.diff(γ)
        Faaαα = Faaα.diff(α)
        Faaαβ = Faaα.diff(β)
        Faaαγ = Faaα.diff(γ)
        Faaββ = Faaβ.diff(β)
        Faaβγ = Faaβ.diff(γ)
        Faaγγ = Faaγ.diff(γ)
        Fabbb = Fabb.diff(b)
        Fabbα = Fabb.diff(α)
        Fabbβ = Fabb.diff(β)
        Fabbγ = Fabb.diff(γ)
        Fabαα = Fabα.diff(α)
        Fabαβ = Fabα.diff(β)
        Fabαγ = Fabα.diff(γ)
        Fabββ = Fabβ.diff(β)
        Fabβγ = Fabβ.diff(γ)
        Fabγγ = Fabγ.diff(γ)
        Faααα = Faαα.diff(α)
        Faααβ = Faαα.diff(β)
        Faααγ = Faαα.diff(γ)
        Faαββ = Faαβ.diff(β)
        Faαβγ = Faαβ.diff(γ)
        Faαγγ = Faαγ.diff(γ)
        Faβββ = Faββ.diff(β)
        Faββγ = Faββ.diff(γ)
        Faβγγ = Faβγ.diff(γ)
        Faγγγ = Faγγ.diff(γ)
        Fbbbb = Fbbb.diff(b)
        Fbbbα = Fbbb.diff(α)
        Fbbbβ = Fbbb.diff(β)
        Fbbbγ = Fbbb.diff(γ)
        Fbbαα = Fbbα.diff(α)
        Fbbαβ = Fbbα.diff(β)
        Fbbαγ = Fbbα.diff(γ)
        Fbbββ = Fbbβ.diff(β)
        Fbbβγ = Fbbβ.diff(γ)
        Fbbγγ = Fbbγ.diff(γ)
        Fbααα = Fbαα.diff(α)
        Fbααβ = Fbαα.diff(β)
        Fbααγ = Fbαα.diff(γ)
        Fbαββ = Fbαβ.diff(β)
        Fbαβγ = Fbαβ.diff(γ)
        Fbαγγ = Fbαγ.diff(γ)
        Fbβββ = Fbββ.diff(β)
        Fbββγ = Fbββ.diff(γ)
        Fbβγγ = Fbβγ.diff(γ)
        Fbγγγ = Fbγγ.diff(γ)
        Fαααα = Fααα.diff(α)
        Fαααβ = Fααα.diff(β)
        Fαααγ = Fααα.diff(γ)
        Fααββ = Fααβ.diff(β)
        Fααβγ = Fααβ.diff(γ)
        Fααγγ = Fααγ.diff(γ)
        Fαβββ = Fαββ.diff(β)
        Fαββγ = Fαββ.diff(γ)
        Fαβγγ = Fαβγ.diff(γ)
        Fαγγγ = Fαγγ.diff(γ)
        Fββββ = Fβββ.diff(β)
        Fβββγ = Fβββ.diff(γ)
        Fββγγ = Fββγ.diff(γ)
        Fβγγγ = Fβγγ.diff(γ)
        Fγγγγ = Fγγγ.diff(γ)

        return (
            Faaaa,
            Faaab,
            Faaaα,
            Faaaβ,
            Faaaγ,
            Faabb,
            Faabα,
            Faabβ,
            Faabγ,
            Faaαα,
            Faaαβ,
            Faaαγ,
            Faaββ,
            Faaβγ,
            Faaγγ,
            Fabbb,
            Fabbα,
            Fabbβ,
            Fabbγ,
            Fabαα,
            Fabαβ,
            Fabαγ,
            Fabββ,
            Fabβγ,
            Fabγγ,
            Faααα,
            Faααβ,
            Faααγ,
            Faαββ,
            Faαβγ,
            Faαγγ,
            Faβββ,
            Faββγ,
            Faβγγ,
            Faγγγ,
            Fbbbb,
            Fbbbα,
            Fbbbβ,
            Fbbbγ,
            Fbbαα,
            Fbbαβ,
            Fbbαγ,
            Fbbββ,
            Fbbβγ,
            Fbbγγ,
            Fbααα,
            Fbααβ,
            Fbααγ,
            Fbαββ,
            Fbαβγ,
            Fbαγγ,
            Fbβββ,
            Fbββγ,
            Fbβγγ,
            Fbγγγ,
            Fαααα,
            Fαααβ,
            Fαααγ,
            Fααββ,
            Fααβγ,
            Fααγγ,
            Fαβββ,
            Fαββγ,
            Fαβγγ,
            Fαγγγ,
            Fββββ,
            Fβββγ,
            Fββγγ,
            Fβγγγ,
            Fγγγγ,
        )

    @timeme
    def gradient(self):
        code = f"""
static void
{self.name}_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}}}
"""
        code = comment_zero_lines(code)
        return code

    @timeme
    def hessian(self):
        code = f"""
static void
{self.name}_second(FunSecondFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}{self.second_derivatives()}
}}
"""
        code = comment_zero_lines(code)
        return code

    @timeme
    def third(self):
        code = f"""
static void
{self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}{self.second_derivatives()}{self.third_derivatives()}}}
"""
        code = comment_zero_lines(code)
        return code

    @timeme
    def fourth(self):
        code = f"""
static void
{self.name}_fourth(FunFourthFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}{self.second_derivatives()}{self.third_derivatives()}{self.fourth_derivatives()}}}
"""
        code = comment_zero_lines(code)
        return code


def comment_zero_lines(code):
    import re
    return re.sub(r'\n(\s*)(.*)= \(0\)\*factor;', r'\n\1// \2= (0)*factor;', code)
