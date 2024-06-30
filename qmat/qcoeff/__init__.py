#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base module for Q coefficients generation
"""
import numpy as np

from qmat.utils import checkOverriding, storeClass, importAll
from qmat.lagrange import LagrangeApproximation

class QGenerator(object):
    r"""
    Base class to generate all the Q-coefficients of a Butcher table
    
    .. math::
        \begin{array}
            {c|c}
            \tau & Q \\
            \hline
            & w^\top
        \end{array}

    where weights, nodes and coefficients itself are included.
    """

    @classmethod
    def getInstance(cls):
        """
        Getter for a instance of a class.

        Returns
        -------
        obj :
            Instance.
        """
        try:
            return cls()
        except TypeError:
            return cls(**cls.DEFAULT_PARAMS)

    @property
    def nodes(self):
        """Nodes of Q-coefficients."""
        raise NotImplementedError("mouahahah")

    @property
    def Q(self):
        """Coefficient matrix."""
        raise NotImplementedError("mouahahah")

    @property
    def weights(self):
        """Weights of Q-coefficients."""
        raise NotImplementedError("mouahahah")

    @property
    def weightsEmbedded(self):
        """
        These weights can be used to construct a secondary lower order method from the same stages.
        """
        raise NotImplementedError("no embedded weights implemented for {type(self).__name__}")

    @property
    def nNodes(self):
        """
        Property to get number of nodes.
        
        Returns
        -------
        int :
            Size of nodes array.
        """
        return self.nodes.size

    @property
    def rightIsNode(self):
        """
        If right node corresponds to end of interval, last node is set to 1.
        
        Returns
        -------
        numpy.1darray :
            Nodes with last node equal to 1.
        """
        return self.nodes[-1] == 1.

    @property
    def T(self):
        """
        Transfer matrix from zero-to-node to node-to-node integration.
        
        Returns
        -------
        numpy.2darray :
            Transfer matrix.
        """
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T

    @property
    def S(self):
        """
        Matrix for node-to-node integration that can be computed using the zero-to-node
        integration matrix.
        
        Returns
        -------
        numpy.2darray :
            Node-to-node integration matrix.
        """
        Q = np.asarray(self.Q)
        M = self.Q.shape[0]
        T = np.eye(M)
        T[1:,:-1][np.diag_indices(M-1)] = -1
        return T @ Q

    @property
    def Tinv(self):
        """
        Transfer matrix from node-to-node to zero-to-node.
        
        Returns
        -------
        numpy.2darray :
            Transfer matrix.
        """
        M = self.Q.shape[0]
        return np.tri(M)

    @property
    def hCoeffs(self):
        """
        Interpolation coefficients for update at end of interval.
        
        Returns
        -------
        numpy.1darray :
            Coefficients.
        """
        approx = LagrangeApproximation(self.nodes)
        return approx.getInterpolationMatrix([1]).ravel()

    def genCoeffs(self, withS=False, hCoeffs=False, embedded=False):
        r"""
        Generates the coefficients together with requested output, i.e., node-to-node integration
        matrix, interpolation coefficients for update at the end of interval. If ``embedded=True``
        output corresponds to coefficients for an embedded scheme.

        Parameters
        ----------
        withS : bool, optional
            If ``True``, coefficients of S-matrix are also added to the output denoting coefficients for node-to-node
            integration. By default ``False``.
        hCoeffs : bool, optional
            If ``True`` interpolation coefficients for the end-interval update are also added to the output. By default
            ``False``.
        embedded : bool, optional
            Used to generate coefficients for an embedded scheme. By default ``False``.

        Returns
        -------
        out : list
            Generated coefficients.
        """
        out = [self.nodes, self.weights, self.Q]

        if embedded:
            out[1] = np.vstack([out[1], self.weightsEmbedded])
        if withS:
            out.append(self.S)
        if hCoeffs:
            out.append(self.hCoeffs)
        return out

    @property
    def order(self):
        """Order of scheme."""
        raise NotImplementedError("mouahahah")

    @property
    def orderEmbedded(self):
        """
        Order of embedded scheme.
        
        Returns
        -------
        int :
            Order of accuracy.
        """
        return self.order - 1

    def solveDahlquist(self, lam, u0, T, nSteps, useEmbeddedWeights=False):
        r"""
        Solves Dahlquist equation numerically by a scheme that uses the Q-coefficients.

        Parameters
        ----------
        lam : float
            Problem parameter :math:`\lambda` of test equation.
        u0 : int
            Initial condition.
        T : float
            End time of simulation.
        nSteps : int
            Number of time steps used for simulation.
        useEmbeddedWeights : bool, optional
            If ``True`` weights of an embedded scheme are used. By default ``False``.

        Returns
        -------
        uNum : np.1darray
            Contains the numerical solution at each time.
        """
        nodes, weights, Q = self.nodes, self.weights, self.Q

        if useEmbeddedWeights:
            weights = self.weightsEmbedded

        uNum = np.zeros(nSteps+1, dtype=complex)
        uNum[0] = u0

        dt = T/nSteps
        A = np.eye(nodes.size) - lam*dt*Q
        for i in range(nSteps):
            b = np.ones(nodes.size)*uNum[i]
            uStages = np.linalg.solve(A, b)
            uNum[i+1] = uNum[i] + lam*dt*weights.dot(uStages)

        return uNum

    def errorDahlquist(self, lam, u0, T, nSteps, uNum=None, useEmbeddedWeights=False):
        r"""
        Error between numerical solution and exact solution of Dahlquist equation is computed by
        executing the ``solveDahlquist`` function for desired parameters.

        Parameters
        ----------
        lam : float
            Problem parameter :math:`\lambda` of test equation.
        u0 : int
            Initial condition.
        T : float
            End time of simulation.
        nSteps : int
            Number of time steps used for simulation.
        useEmbeddedWeights : bool, optional
            If ``True`` weights of an embedded scheme are used. By default ``False``.

        Returns
        -------
        float :
            Error.
        """
        if uNum is None:
            uNum = self.solveDahlquist(lam, u0, T, nSteps, useEmbeddedWeights=useEmbeddedWeights)
        times = np.linspace(0, T, nSteps+1)
        uExact = u0 * np.exp(lam*times)
        return np.linalg.norm(uNum-uExact, ord=np.inf)


Q_GENERATORS = {}

def register(cls:QGenerator)->QGenerator:
    r"""
    For registration of a new subclass of ``QGenerator``, the class will be checked for correct
    overriding, i.e., it is proven whether the properties of the base class are correctly overwritten
    (and not returning a ``NotImplementedError``). For instantiation of a new class, default parameters
    needs to be defined. Finally, the subclass is stored in the dictionary ``Q_GENERATORS`` providing
    a collection of current generators.

    Parameters
    ----------
    cls : QGenerator
        New subclass.

    Returns
    -------
    QGenerator
        Subclass that passed the checks.
    """
    # Check for correct overriding
    for name in ["nodes", "Q", "weights", "order"]:
        checkOverriding(cls, name)
    # Check that TEST_PARAMS are given and valid if no default constructor
    try:
        cls()
    except TypeError:
        try:
            params = cls.DEFAULT_PARAMS
        except AttributeError:
            raise AttributeError(
                f"{cls.__name__} requires DEFAULT_PARAMS attribute"
                " since it has no default constructor")
        try:
            cls(**params)
        except:
            raise TypeError(
                f"{cls.__name__} could not be instantiated with DEFAULT_PARAMS")
    # Store class (and aliases)
    storeClass(cls, Q_GENERATORS)
    return cls

def genQCoeffs(qType, withS=False, hCoeffs=False, embedded=False, **params):
    r"""
    Generates Q-coefficients, i.e., nodes, weights, and coefficients matrix of a collocation or a Runge-Kutta method.

    Parameters
    ----------
    qType : str
        Type of coefficients generated. Can be ``'Collocation'`` in case of a collocation method. For a list of available
        coefficients, see keys of the ``Q_GENERATORS`` dictionary.
    withS : bool, optional
        If ``True``, coefficients of S-matrix are also added to the output denoting coefficients for node-to-node
        integration. By default ``False``.
    hCoeffs : bool, optional
        If ``True`` interpolation coefficients for the end-interval update are also added to the output. By default
        ``False``.
    embedded : bool, optional
        Used to generate coefficients for an embedded scheme. By default ``False``.

    Returns
    -------
    numpy.1darray's
        Nodes, weights and coefficients matrix.
    """
    try:
        Generator = Q_GENERATORS[qType]
    except KeyError:
        raise ValueError(f"{qType=!r} is not available")
    gen = Generator(**params)
    return gen.genCoeffs(withS, hCoeffs, embedded)


# Import all local submodules
__all__ = importAll(locals(), __path__, __name__, __import__)
