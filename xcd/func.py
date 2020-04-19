import textwrap

from sympy import ccode

from .base import BaseFunctional


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
                  ea = {ccode(self.Fa)};
              if (dp->rhob >{self.name.upper()}_THRESHOLD)
                  eb = {ccode(self.Fb)};
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
                 ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
              if (dp->rhob>{self.name.upper()}_THRESHOLD)
                 ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
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
                 ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
                 ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
                 ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
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
                 ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
                 ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
                 ds->df3000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
                 ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
                 ds->df0300 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
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
                 ds->df1000 += ({ccode(self.Fa.diff(self.ra))})*factor;
                 ds->df2000 += ({ccode(self.Fa.diff(self.ra, self.ra))})*factor;
                 ds->df3000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra))})*factor;
                 ds->df4000 += ({ccode(self.Fa.diff(self.ra, self.ra, self.ra, self.ra))})*factor;
                 }}
              if (dp->rhob>{self.name.upper()}_THRESHOLD) {{
                 ds->df0100 += ({ccode(self.Fb.diff(self.rb))})*factor;
                 ds->df0200 += ({ccode(self.Fb.diff(self.rb, self.rb))})*factor;
                 ds->df0300 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb))})*factor;
                 ds->df0400 += ({ccode(self.Fb.diff(self.rb, self.rb, self.rb, self.rb))})*factor;
                 }}
            }}
            """
        )

        return code
