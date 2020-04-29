import time


def timeme(f):
    """
    Simple timer decorator

    Usage

    >>> @timeme
    ... def foo():
    ...     time.sleep(1)
    >>> foo()
    Running in foo...
    ...1.00s
    """

    def wrap(*args, **kwargs):
        print(f'Running in {f.__name__}...')
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()
        print(f'...{t2-t1:.2f}s')
        return res
    return wrap


class Timer():
    """
    Class timer decorator with indented output for nested calls

    Usage

    >>> timethis = Timer()
    >>> @timethis
    ... def foo():
    ...     bar()
    ...     time.sleep(.1)
    >>> @timethis
    ... def bar():
    ...     time.sleep(.2)
    >>> foo()
    Running in foo...
        Running in bar...
        0.20s
    0.30s
    """

    indent = -4

    def __call__(self, f):
        def wrap(*args, **kwargs):
            Timer.indent += 4
            print(Timer.indent*' ' + f'Running in {f.__name__}...')
            t1 = time.time()
            res = f(*args, **kwargs)
            t2 = time.time()
            print(Timer.indent*' ' + f'{t2-t1:.2f}s')
            Timer.indent -= 4
            return res
        return wrap


if __name__ == "__main__":

    tm = Timer()

    @tm
    def foo():
        bar()
        time.sleep(.1)

    @tm
    def bar():
        baz()
        time.sleep(.2)

    @tm
    def baz():
        time.sleep(.3)

    baz()
    foo()
