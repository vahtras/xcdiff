import textwrap
import functools
import concurrent.futures

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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(ccode, self.grad())
        Fa, Fb, Fα, Fβ, Fγ = results
        code = f"""
  ds->df1000 += ({Fa})*factor;
  ds->df0100 += ({Fb})*factor;
  ds->df0010 += ({Fα})*factor;
  ds->df0001 += ({Fβ})*factor;
  ds->df00001 += ({Fγ})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def grad(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.F.diff, x) for x in self.args]
            results = (future.result() for future in futures)

        return tuple(results)

    @timeme
    @functools.lru_cache(None)
    def second_derivatives(self):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(ccode, self.hess())

        Faa, Fab, Faα, Faβ, Faγ,\
            Fbb, Fbα, Fbβ, Fbγ, Fαα, Fαβ, Fαγ, Fββ, Fβγ, Fγγ = results

        code = f"""
  ds->df2000 += ({Faa})*factor;
  ds->df1100 += ({Fab})*factor;
  ds->df1010 += ({Faα})*factor;
  ds->df1001 += ({Faβ})*factor;
  ds->df10001 += ({Faγ})*factor;
  ds->df0200 += ({Fbb})*factor;
  ds->df0110 += ({Fbα})*factor;
  ds->df0101 += ({Fbβ})*factor;
  ds->df01001 += ({Fbγ})*factor;
  ds->df0020 += ({Fαα})*factor;
  ds->df0011 += ({Fαβ})*factor;
  ds->df00101 += ({Fαγ})*factor;
  ds->df0002 += ({Fββ})*factor;
  ds->df00011 += ({Fβγ})*factor;
  ds->df00002 += ({Fγγ})*factor;
"""
        return code

    @timeme
    @functools.lru_cache(None)
    def hess(self):
        Fa, Fb, Fα, Fβ, Fγ = self.grad()
        a, b, α, β, γ = self.args
        """
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
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = \
                [executor.submit(Fa.diff, x) for x in self.args] + \
                [executor.submit(Fb.diff, x) for x in self.args[1:]] + \
                [executor.submit(Fα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fγ.diff, x) for x in self.args[4:]]

            results = (future.result() for future in futures)

        return tuple(results)

    @timeme
    @functools.lru_cache(None)
    def third_derivatives(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(ccode, self.kolm())
        (
            Faaa, Faab, Faaα, Faaβ, Faaγ, Fabb,
            Fabα, Fabβ, Fabγ, Faαα, Faαβ, Faαγ,
            Faββ, Faβγ, Faγγ, Fbbb, Fbbα, Fbbβ,
            Fbbγ, Fbαα, Fbαβ, Fbαγ, Fbββ, Fbβγ,
            Fbγγ, Fααα, Fααβ, Fααγ, Fαββ, Fαβγ,
            Fαγγ, Fβββ, Fββγ, Fβγγ, Fγγγ
        ) = results

        code = f"""
  ds->df3000 += ({Faaa})*factor;
  ds->df2100 += ({Faab})*factor;
  ds->df2010 += ({Faaα})*factor;
  ds->df2001 += ({Faaβ})*factor;
  ds->df20001 += ({Faaγ})*factor;
  ds->df1200 += ({Fabb})*factor;
  ds->df1110 += ({Fabα})*factor;
  ds->df1101 += ({Fabβ})*factor;
  ds->df11001 += ({Fabγ})*factor;
  ds->df1020 += ({Faαα})*factor;
  ds->df1011 += ({Faαβ})*factor;
  ds->df10101 += ({Faαγ})*factor;
  ds->df1002 += ({Faββ})*factor;
  ds->df10011 += ({Faβγ})*factor;
  ds->df10002 += ({Faγγ})*factor;
  ds->df0300 += ({Fbbb})*factor;
  ds->df0210 += ({Fbbα})*factor;
  ds->df0201 += ({Fbbβ})*factor;
  ds->df02001 += ({Fbbγ})*factor;
  ds->df0120 += ({Fbαα})*factor;
  ds->df0111 += ({Fbαβ})*factor;
  ds->df01101 += ({Fbαγ})*factor;
  ds->df0102 += ({Fbββ})*factor;
  ds->df01011 += ({Fbβγ})*factor;
  ds->df01002 += ({Fbγγ})*factor;
  ds->df0030 += ({Fααα})*factor;
  ds->df0021 += ({Fααβ})*factor;
  ds->df00201 += ({Fααγ})*factor;
  ds->df0012 += ({Fαββ})*factor;
  ds->df00111 += ({Fαβγ})*factor;
  ds->df00102 += ({Fαγγ})*factor;
  ds->df0003 += ({Fβββ})*factor;
  ds->df00021 += ({Fββγ})*factor;
  ds->df00012 += ({Fβγγ})*factor;
  ds->df00003 += ({Fγγγ})*factor;
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
        """
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
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = \
                [executor.submit(Faa.diff, x) for x in self.args] + \
                [executor.submit(Fab.diff, x) for x in self.args[1:]] + \
                [executor.submit(Faα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Faβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Faγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fbb.diff, x) for x in self.args[1:]] + \
                [executor.submit(Fbα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fbβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fbγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fαα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fαβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fαγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fββ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fβγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fγγ.diff, x) for x in self.args[4:]]

            results = (future.result() for future in futures)

        return tuple(results)

        """
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
        """

    @timeme
    @functools.lru_cache(None)
    def fourth_derivatives(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(ccode, self.neli())
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
        ) = results

        code = f"""
  ds->df4000 += ({Faaaa})*factor;
  ds->df3100 += ({Faaab})*factor;
  ds->df3010 += ({Faaaα})*factor;
  ds->df3001 += ({Faaaβ})*factor;
  ds->df30001 += ({Faaaγ})*factor;
  ds->df2200 += ({Faabb})*factor;
  ds->df2110 += ({Faabα})*factor;
  ds->df2101 += ({Faabβ})*factor;
  ds->df21001 += ({Faabγ})*factor;
  ds->df2020 += ({Faaαα})*factor;
  ds->df2011 += ({Faaαβ})*factor;
  ds->df20101 += ({Faaαγ})*factor;
  ds->df2002 += ({Faaββ})*factor;
  ds->df20011 += ({Faaβγ})*factor;
  ds->df20002 += ({Faaγγ})*factor;
  ds->df1300 += ({Fabbb})*factor;
  ds->df1210 += ({Fabbα})*factor;
  ds->df1201 += ({Fabbβ})*factor;
  ds->df12001 += ({Fabbγ})*factor;
  ds->df1120 += ({Fabαα})*factor;
  ds->df1111 += ({Fabαβ})*factor;
  ds->df11101 += ({Fabαγ})*factor;
  ds->df1102 += ({Fabββ})*factor;
  ds->df11011 += ({Fabβγ})*factor;
  ds->df11002 += ({Fabγγ})*factor;
  ds->df1030 += ({Faααα})*factor;
  ds->df1021 += ({Faααβ})*factor;
  ds->df10201 += ({Faααγ})*factor;
  ds->df1012 += ({Faαββ})*factor;
  ds->df10111 += ({Faαβγ})*factor;
  ds->df10102 += ({Faαγγ})*factor;
  ds->df1003 += ({Faβββ})*factor;
  ds->df10021 += ({Faββγ})*factor;
  ds->df10012 += ({Faβγγ})*factor;
  ds->df10003 += ({Faγγγ})*factor;
  ds->df0400 += ({Fbbbb})*factor;
  ds->df0310 += ({Fbbbα})*factor;
  ds->df0301 += ({Fbbbβ})*factor;
  ds->df03001 += ({Fbbbγ})*factor;
  ds->df0220 += ({Fbbαα})*factor;
  ds->df0211 += ({Fbbαβ})*factor;
  ds->df02101 += ({Fbbαγ})*factor;
  ds->df0202 += ({Fbbββ})*factor;
  ds->df02011 += ({Fbbβγ})*factor;
  ds->df02002 += ({Fbbγγ})*factor;
  ds->df0130 += ({Fbααα})*factor;
  ds->df0121 += ({Fbααβ})*factor;
  ds->df01201 += ({Fbααγ})*factor;
  ds->df0112 += ({Fbαββ})*factor;
  ds->df01111 += ({Fbαβγ})*factor;
  ds->df01102 += ({Fbαγγ})*factor;
  ds->df0103 += ({Fbβββ})*factor;
  ds->df01021 += ({Fbββγ})*factor;
  ds->df01012 += ({Fbβγγ})*factor;
  ds->df01003 += ({Fbγγγ})*factor;
  ds->df0040 += ({Fαααα})*factor;
  ds->df0031 += ({Fαααβ})*factor;
  ds->df00301 += ({Fαααγ})*factor;
  ds->df0022 += ({Fααββ})*factor;
  ds->df00211 += ({Fααβγ})*factor;
  ds->df00202 += ({Fααγγ})*factor;
  ds->df0013 += ({Fαβββ})*factor;
  ds->df00121 += ({Fαββγ})*factor;
  ds->df00112 += ({Fαβγγ})*factor;
  ds->df00103 += ({Fαγγγ})*factor;
  ds->df0004 += ({Fββββ})*factor;
  ds->df00031 += ({Fβββγ})*factor;
  ds->df00022 += ({Fββγγ})*factor;
  ds->df00013 += ({Fβγγγ})*factor;
  ds->df00004 += ({Fγγγγ})*factor;
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
        """
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
        """

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = \
                [executor.submit(Faaa.diff, x) for x in self.args] + \
                [executor.submit(Faab.diff, x) for x in self.args[1:]] + \
                [executor.submit(Faaα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Faaβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Faaγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fabb.diff, x) for x in self.args[1:]] + \
                [executor.submit(Fabα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fabβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fabγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Faαα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Faαβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Faαγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Faββ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Faβγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Faγγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fbbb.diff, x) for x in self.args[1:]] + \
                [executor.submit(Fbbα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fbbβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fbbγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fbαα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fbαβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fbαγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fbββ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fbβγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fbγγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fααα.diff, x) for x in self.args[2:]] + \
                [executor.submit(Fααβ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fααγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fαββ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fαβγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fαγγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fβββ.diff, x) for x in self.args[3:]] + \
                [executor.submit(Fββγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fβγγ.diff, x) for x in self.args[4:]] + \
                [executor.submit(Fγγγ.diff, x) for x in self.args[4:]]

            results = [future.result() for future in futures]

        return results

        """
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
    """

    @timeme
    @functools.lru_cache(None)
    def gradient(self):
        code = f"""
static void
{self.name}_first(FunFirstFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}}}
"""
        code = comment_zero_lines(code)
        return code

    @timeme
    @functools.lru_cache(None)
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
    @functools.lru_cache(None)
    def third(self):
        code = f"""
static void
{self.name}_third(FunThirdFuncDrv *ds, real factor, const FunDensProp* dp)
{{{self.first_derivatives()}{self.second_derivatives()}{self.third_derivatives()}}}
"""
        code = comment_zero_lines(code)
        return code

    @timeme
    @functools.lru_cache(None)
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
