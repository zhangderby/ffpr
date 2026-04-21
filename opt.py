"""Various optimization algorithms from prysm with minor modifications to L-BFGS-B to make it compatible with cupy."""
import warnings

from utils import ensure_np

from scipy.optimize import _lbfgsb

from prysm.mathops import (np,
                           _np)

from prysm.x.optym.optimizers import (runN,
                                      GradientDescent,
                                      AdaGrad,
                                      RMSProp,
                                      Adam,
                                      RAdam,
                                      AdaMomentum,
                                      Yogi)

class F77LBFGSB:
    """Limited Memory Broyden Fletcher Goldfarb Shannon optimizer, variant B (L-BFGS-B).

    L-BFGS-B is a Quasi-Newton method which uses the previous m gradient vectors
    to perform the BFGS update, which itself is an approximation of Newton's
    Method.

    The "L" in L-BFGS is Limited Memory, due to this m*n storage requirement,
    where m is a small integer (say 10 to 30), and n is the number of variables.

    At its core, L-BFGS solves the BFGS update using an adaptive line search,
    satisfying the strong Wolfe conditions, which guarantee that it does not
    move uphill.

    Variant B (BFGS-B) incorporates subspace minimization, which further
    accelerates convergence.

    Subspace minimization is the practice of forming a lower-dimensional "manifold"
    (essentially, enclosing Euclidean geometry) for the problem at a given
    iteration, and then exactly solving for the minimum of that manifold.

    The combination of subspace minimization and a quasi-newton update give
    L-BFGS-B exponential convergence, where it may converge by an order of
    magnitude in cost or more on each iteration.

    This wrapper around Jorge Nocedal's Fortran code made available through
    SciPy attenpts to defeat the built-in convergence tests of lbfgsb.f, but
    is not always successful due to the nature of floating point arithmetic.
    Unlike all other classes in this file, L-BFGS-B may refuse to step(), and
    may stop early in a runN or run_to call.  A warning will be generated in
    such instances.

    References
    ----------
    [1] Jorge Nocedal, "Updating Quasi-Newton Matricies with Limited Storage"
        https://doi.org/10.2307/2006193

    [2] Richard H. Byrd, Peihuang Lu, and Jorge Nocedal "A Limited-Memory
        Algorithm For Bound-Constrained Optimization"
        https://doi.org/10.1137/0916069

    [3] Ciyou Zhu, Richard H. Byrd, Peihuang Lu, and Jorge Nocedal "Algorithm 778:
        L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization"
        https://doi.org/10.1145/279232.279236

    [4] José Luis Morales and Jorge Nocedal, "Remark on "algorithm 778: L-BFGS-B:
        Fortran subroutines for large-scale bound constrained optimization"
        https://doi.org/10.1145/2049662.2049669

    """
    def __init__(self, fg, x0, memory=10, lower_bounds=None, upper_bounds=None):
        """Create a new L-BFGS-B optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        memory : int
            the number of recent gradient vectors to use in performing the
            approximate Newton's step
        lower_bounds : numpy.ndarray, optional
            vector of same size as x0 containing the hard lower bounds for the
            variables; if None, unconstrained lb
        upper_bounds : numpy.ndarray, optional
            vector of same size as x0 containing the hard upper bounds for the
            variables; if None, unconstrained ub

        """
        self.fg = fg
        self.x0 = ensure_np(x0)
        self.n = len(x0)  # n = n vars
        self.m = memory

        # create the work arrays Fortran needs
        fint_dtype = _np.int32 # _lbfgsb.types.intvar.dtype
#         ffloat_dtype = x0.dtype  maybe can uncomment this someday, but probably not.
        ffloat_dtype = _np.float64

        # todo: f77 code explodes for f32 dtype?
        if lower_bounds is None:
            lower_bounds = _np.full(self.n, -_np.Inf, dtype=ffloat_dtype)

        if upper_bounds is None:
            upper_bounds = _np.full(self.n, _np.Inf, dtype=ffloat_dtype)

        # nbd is an array of integers for Fortran
        #         nbd(i)=0 if x(i) is unbounded,
        #                1 if x(i) has only a lower bound,
        #                2 if x(i) has both lower and upper bounds, and
        #                3 if x(i) has only an upper bound.
        nbd = _np.zeros(self.n, dtype=fint_dtype)
        self.l = lower_bounds  # NOQA
        self.u = upper_bounds
        finite_lower_bound = _np.isfinite(self.l)
        finite_upper_bound = _np.isfinite(self.u)
        # unbounded case handled in init as zeros
        lower_but_not_upper_bound = finite_lower_bound & ~finite_upper_bound
        upper_but_not_lower_bound = finite_upper_bound & ~finite_lower_bound
        both_bounds = finite_lower_bound & finite_upper_bound
        nbd[lower_but_not_upper_bound] = 1
        nbd[both_bounds]               = 2  # NOQA
        nbd[upper_but_not_lower_bound] = 3
        self.nbd = nbd

        # much less complicated initializations
        m, n = self.m, self.n
        self.x = x0.copy()
        self.f = _np.array(0.0, dtype=ffloat_dtype)
        self.g = _np.zeros((self.n), dtype=ffloat_dtype)
        # see lbfgsb.f for this size
        # error in the docstring, see line 240 to 252
        self.wa = _np.zeros(2 * m * n + 11 * m ** 2 + 5 * n + 8 * m, dtype=ffloat_dtype)
        self.iwa = _np.zeros(3 * n, dtype=fint_dtype)
        self.f = _np.array([0.], dtype=ffloat_dtype)
        self.g = _np.zeros([self.n], dtype=ffloat_dtype)
        self.task = _np.zeros(1, dtype='S60')  # S60 = <= 60 character wide byte array
        self.csave = _np.zeros(1, dtype='S60')
        self.lsave = _np.zeros(4, dtype=fint_dtype)
        self.isave = _np.zeros(44, dtype=fint_dtype)
        self.dsave = _np.zeros(29, dtype=ffloat_dtype)
        self.task[:] = 'START'

        self.iter = 0

        # try to prevent F77 driver from ever stopping on its own
        # cannot use NaN or Inf, Fortran comparisons do not work
        # properly, so pick unreasonably small numbers.
        # TODO: would a negative number be better here?
        self.factr = 1e-999
        self.pgtol = 1e-999

        # other stuff to be added to the interface later
        self.maxls = 20
        self.iprint = 1

    def _call_fortran(self):
        _lbfgsb.setulb(self.m, self.x, self.l, self.u, self.nbd, self.f, self.g,
                       self.factr, self.pgtol, self.wa, self.iwa, self.task, self.iprint,
                       self.csave, self.lsave, self.isave, self.dsave, self.maxls)

    def _view_s(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[0:m*n].reshape(m, n)[:self._valid_space_sy]

    def _view_y(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[m*n:2*m*n].reshape(m, n)[:self._valid_space_sy]

    @property
    def _nbfgs_updates(self):
        return self.isave[30]

    @property
    def _valid_space_sy(self):
        return min(self._nbfgs_updates, self.m)

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1  # increment first so that while loop is self-breaking
        x = self.x.copy()
        while self._nbfgs_updates < self.iter:
            # call F77 mutates all of the class's state
            self._call_fortran()
            # strip null bytes/termination and any ASCII white space
            task = self.task.tobytes().strip(b'\x00').strip()
            if task.startswith(b'FG'):
                f, g = self.fg(self.x)
                if hasattr(g, 'get'):
                    f = float(f)
                    g = g.get()
                if g.ndim != 1:
                    g = g.ravel()

                self.f[:] = f
                self.g[:] = g
                self._call_fortran()

            if _fortran_died(task):
                msg = task.decode('UTF-8')
                raise ValueError("the Fortran L-BFGS-B driver thinks something is wrong with the problem and gave the message " + msg)

            # TODO: fix this properly
            if _fortran_converged(task):
                raise StopIteration
                # break

            if _fortran_major_iter_complete(task):
                break

        return x, self.f, self.g

    def run_to(self, N):
        """Run the optimizer until its iteration count equals N."""
        while self.iter < N:
            try:
                yield self.step()
            except StopIteration:
                warnings.warn(f'L-BFGS-B can make no further progress; stopped on iteration {self.iter}/N iterations')
                break


def _fortran_died(task):
    return task.startswith(b'STOP')


def _fortran_converged(task):
    return task.startswith(b'CONV')


def _fortran_major_iter_complete(task):
    return task.startswith(b'NEW_X')