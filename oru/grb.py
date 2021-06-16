import dataclasses
import json
import dacite

from typing import ClassVar, Dict, Any, Tuple, Union, NewType, Callable
from .constants import _GUROBI_MODEL_ATTR, INFO_ATTR_TO_MODEL_ATTR, EPS
from .core import take
from gurobi import *
from functools import wraps
from collections import deque
from .json import JSONSerialisableDataclass

VarDict = Dict[Union[int, Tuple[int, ...]], Var]


class BinVarDict(VarDict):
    pass


class IntVarDict(VarDict):
    pass


class CtsVarDict(VarDict):
    pass


class CtsVar(Var):
    pass


class BinVar(Var):
    pass


class IntVar(Var):
    pass


@dataclasses.dataclass
class GurobiModelInformation(JSONSerialisableDataclass):
    barr_iters: int
    is_multiobj: bool
    max_coeff: float
    max_obj_coeff: float
    max_q_obj_coeff: float
    max_qc_coeff: float
    max_qc_lin_coeff: float
    max_qc_rhs: float
    max_rhs: float
    max_var_bnd: float
    min_coeff: float
    min_obj_coeff: float
    min_q_obj_coeff: float
    min_qc_coeff: float
    min_qc_lin_coeff: float
    min_qc_rhs: float
    min_rhs: float
    min_var_bnd: float
    model_name: str
    model_sense: int
    model_time: float
    num_bin_vars: int
    num_constr: int
    num_gen_constr: int
    num_int_vars: int
    num_nodes: int
    num_nz: int
    num_pwl_vars: int
    num_quad_coeff_nz: int
    num_quad_constr: int
    num_quad_nz: int
    num_sos: int
    num_start: int
    num_vars: int
    obj_bnd: float
    obj_const: float
    obj_val: float
    simplex_iters: int
    sol_count: int
    status: int

    # MIP-only
    mip_gap: float
    obj_bnd_raw: float
    pool_obj_bnd: float
    pool_obj_val: float

    # LP-only
    kappa: float


def _wrap_callback(callback):
    def wrapped_callback(model: Model, where):
        return callback(model._parent, where)

    return wrapped_callback


def pprint_constraint(cons: Constr, model: Model, eps=EPS):
    lhs = ''
    for var in model.getVars():
        a = model.getCoeff(cons, var)
        if abs(a) > eps:
            if a < 0:
                sgn = '-'
            else:
                sgn = '+'
            a = abs(a)
            if abs(a - 1) < eps:
                a = ''
            else:
                a = f'{a:.4g}'
            lhs += f' {sgn} {a}{var.VarName}'

    print(' '.join([lhs.rstrip().rstrip('+ '), cons.sense, str(cons.RHS)]))


class ModelWrapper:
    model: Model

    def __init__(self, name=""):
        self.model = Model(name=name)
        self.model._parent = self

    @property
    def NumConstrs(self):
        try:
            return self.model.getAttr("NumConstrs")
        except AttributeError:
            return None

    @property
    def NumVars(self):
        try:
            return self.model.getAttr("NumVars")
        except AttributeError:
            return None

    @property
    def NumSOS(self):
        try:
            return self.model.getAttr("NumSOS")
        except AttributeError:
            return None

    @property
    def NumQConstrs(self):
        try:
            return self.model.getAttr("NumQConstrs")
        except AttributeError:
            return None

    @property
    def NumGenConstrs(self):
        try:
            return self.model.getAttr("NumGenConstrs")
        except AttributeError:
            return None

    @property
    def NumNZs(self):
        try:
            return self.model.getAttr("NumNZs")
        except AttributeError:
            return None

    @property
    def DNumNZs(self):
        try:
            return self.model.getAttr("DNumNZs")
        except AttributeError:
            return None

    @property
    def NumQNZs(self):
        try:
            return self.model.getAttr("NumQNZs")
        except AttributeError:
            return None

    @property
    def NumQCNZs(self):
        try:
            return self.model.getAttr("NumQCNZs")
        except AttributeError:
            return None

    @property
    def NumIntVars(self):
        try:
            return self.model.getAttr("NumIntVars")
        except AttributeError:
            return None

    @property
    def NumBinVars(self):
        try:
            return self.model.getAttr("NumBinVars")
        except AttributeError:
            return None

    @property
    def NumPWLObjVars(self):
        try:
            return self.model.getAttr("NumPWLObjVars")
        except AttributeError:
            return None

    @property
    def ModelName(self):
        try:
            return self.model.getAttr("ModelName")
        except AttributeError:
            return None

    @property
    def ModelSense(self):
        try:
            return self.model.getAttr("ModelSense")
        except AttributeError:
            return None

    @property
    def ObjCon(self):
        try:
            return self.model.getAttr("ObjCon")
        except AttributeError:
            return None

    @property
    def ObjVal(self):
        """
        The objective value for the current solution. If the model was solved to optimality, then this attribute gives
        the optimal objective value.  Otherwise, ``None`` is returned.
        """
        try:
            return self.model.getAttr("ObjVal")
        except AttributeError:
            return None

    @property
    def ObjBound(self):
        """
        The best known bound on the optimal objective. When solving a MIP model, the algorithm maintains both a lower
        bound and an upper bound on the optimal objective value. For a minimization model, the upper bound is the
        objective of the best known feasible solution, while the lower bound gives a bound on the best possible
        objective.

        In contrast to ObjBoundC, this attribute takes advantage of objective integrality information to round to a
        tighter bound. For example, if the objective is known to take an integral value and the current best bound is
        1.5, ObjBound will return 2.0 while ObjBoundC will return 1.5.
        """
        try:
            return self.model.getAttr("ObjBound")
        except AttributeError:
            return None

    @property
    def ObjBoundC(self):
        """
        The best known bound on the optimal objective. When solving a MIP model, the algorithm maintains both a lower
        bound and an upper bound on the optimal objective value. For a minimization model, the upper bound is the
        objective of the best known feasible solution, while the lower bound gives a bound on the best possible
        objective.

        In contrast to ObjBound, this attribute does not take advantage of objective integrality information to round
        to a tighter bound. For example, if the objective is known to take an integral value and the current best
        bound is 1.5, ObjBound will return 2.0 while ObjBoundC will return 1.5.
        """
        try:
            return self.model.getAttr("ObjBoundC")
        except AttributeError:
            return None

    @property
    def PoolObjBound(self):
        try:
            return self.model.getAttr("PoolObjBound")
        except AttributeError:
            return None

    @property
    def PoolObjVal(self):
        try:
            return self.model.getAttr("PoolObjVal")
        except AttributeError:
            return None

    @property
    def MIPGap(self):
        try:
            return self.model.getAttr("MIPGap")
        except AttributeError:
            return None

    @property
    def Runtime(self):
        try:
            return self.model.getAttr("Runtime")
        except AttributeError:
            return None

    @property
    def Status(self):
        try:
            return self.model.getAttr("Status")
        except AttributeError:
            return None

    @property
    def SolCount(self):
        try:
            return self.model.getAttr("SolCount")
        except AttributeError:
            return None

    @property
    def IterCount(self):
        try:
            return self.model.getAttr("IterCount")
        except AttributeError:
            return None

    @property
    def BarIterCount(self):
        try:
            return self.model.getAttr("BarIterCount")
        except AttributeError:
            return None

    @property
    def NodeCount(self):
        try:
            return self.model.getAttr("NodeCount")
        except AttributeError:
            return None

    @property
    def IsMIP(self):
        try:
            return self.model.getAttr("IsMIP")
        except AttributeError:
            return None

    @property
    def IsQP(self):
        try:
            return self.model.getAttr("IsQP")
        except AttributeError:
            return None

    @property
    def IsQCP(self):
        try:
            return self.model.getAttr("IsQCP")
        except AttributeError:
            return None

    @property
    def IsMultiObj(self):
        try:
            return self.model.getAttr("IsMultiObj")
        except AttributeError:
            return None

    @property
    def IISMinimal(self):
        try:
            return self.model.getAttr("IISMinimal")
        except AttributeError:
            return None

    @property
    def MaxCoeff(self):
        try:
            return self.model.getAttr("MaxCoeff")
        except AttributeError:
            return None

    @property
    def MinCoeff(self):
        try:
            return self.model.getAttr("MinCoeff")
        except AttributeError:
            return None

    @property
    def MaxBound(self):
        try:
            return self.model.getAttr("MaxBound")
        except AttributeError:
            return None

    @property
    def MinBound(self):
        try:
            return self.model.getAttr("MinBound")
        except AttributeError:
            return None

    @property
    def MaxObjCoeff(self):
        try:
            return self.model.getAttr("MaxObjCoeff")
        except AttributeError:
            return None

    @property
    def MinObjCoeff(self):
        try:
            return self.model.getAttr("MinObjCoeff")
        except AttributeError:
            return None

    @property
    def MaxRHS(self):
        try:
            return self.model.getAttr("MaxRHS")
        except AttributeError:
            return None

    @property
    def MinRHS(self):
        try:
            return self.model.getAttr("MinRHS")
        except AttributeError:
            return None

    @property
    def MaxQCCoeff(self):
        try:
            return self.model.getAttr("MaxQCCoeff")
        except AttributeError:
            return None

    @property
    def MinQCCoeff(self):
        try:
            return self.model.getAttr("MinQCCoeff")
        except AttributeError:
            return None

    @property
    def MaxQCLCoeff(self):
        try:
            return self.model.getAttr("MaxQCLCoeff")
        except AttributeError:
            return None

    @property
    def MinQCLCoeff(self):
        try:
            return self.model.getAttr("MinQCLCoeff")
        except AttributeError:
            return None

    @property
    def MaxQCRHS(self):
        try:
            return self.model.getAttr("MaxQCRHS")
        except AttributeError:
            return None

    @property
    def MinQCRHS(self):
        try:
            return self.model.getAttr("MinQCRHS")
        except AttributeError:
            return None

    @property
    def MaxQObjCoeff(self):
        try:
            return self.model.getAttr("MaxQObjCoeff")
        except AttributeError:
            return None

    @property
    def MinQObjCoeff(self):
        try:
            return self.model.getAttr("MinQObjCoeff")
        except AttributeError:
            return None

    @property
    def Kappa(self):
        try:
            return self.model.getAttr("Kappa")
        except AttributeError:
            return None

    @property
    def KappaExact(self):
        try:
            return self.model.getAttr("KappaExact")
        except AttributeError:
            return None

    @property
    def FarkasProof(self):
        try:
            return self.model.getAttr("FarkasProof")
        except AttributeError:
            return None

    @property
    def TuneResultCount(self):
        try:
            return self.model.getAttr("TuneResultCount")
        except AttributeError:
            return None

    @property
    def NumStart(self):
        try:
            return self.model.getAttr("NumStart")
        except AttributeError:
            return None

    @property
    def LicenseExpiration(self):
        try:
            return self.model.getAttr("LicenseExpiration")
        except AttributeError:
            return None

    @property
    def JobID(self):
        try:
            return self.model.getAttr("JobID")
        except AttributeError:
            return None

    @property
    def Server(self):
        try:
            return self.model.getAttr("Server")
        except AttributeError:
            return None

    def addConstr(self, lhs, sense=None, rhs=None, name=""):
        """Add a constraint to a model. """
        return self.model.addConstr(lhs, sense, rhs, name)

    def addConstrs(self, generator, name=""):
        """
        Add multiple constraints to a model using a Python generator expression. Returns a Gurobi tupledict that
        contains the newly created constraints, indexed by the values generated by the generator expression.
        """
        return self.model.addConstrs(generator, name=name)

    def addGenConstrAbs(self, resvar, argvar, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_ABS to a model."""
        return self.model.addGenConstrAbs(resvar, argvar, name)

    def addGenConstrAnd(self, resvar, vars, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_AND to a model. """
        return self.model.addGenConstrAnd(resvar, vars, name)

    def addGenConstrIndicator(self, binvar, binval, lhs, sense=None, rhs=None, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_INDICATOR to a model. """
        return self.model.addGenConstrIndicator(binvar, binval, lhs, sense, rhs, name)

    def addGenConstrMax(self, resvar, vars, constant=None, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_MAX to a model."""
        return self.model.addGenConstrMax(resvar, vars, constant, name)

    def addGenConstrMin(self, resvar, vars, constant=None, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_MIN to a model. """
        return self.model.addGenConstrMin(resvar, vars, constant, name)

    def addGenConstrOr(self, resvar, vars, name=""):
        """Add a new general constraint of type GRB.GENCONSTR_OR to a model. """
        return self.model.addGenConstrOr(resvar, vars, name)

    def addLConstr(self, lhs, sense=None, rhs=None, name=""):
        """
        Add a linear constraint to a model. This method is faster than addConstr()
        (as much as 50% faster for very sparse constraints), but can only be used to add linear constraints.
        """
        return self.model.addLConstr(lhs, sense, rhs, name)

    def addQConstr(self, lhs, sense=None, rhs=None, name=""):
        """
         Add a quadratic constraint to a model. Important note: the algorithms that Gurobi uses to solve quadratically
         constrained problems can only handle certain types of quadratic constraints.
        """
        return self.model.addQConstr(lhs, sense, rhs, name)

    def addRange(self, expr, lower, upper, name=""):
        """
         Add a range constraint to a model. A range constraint states that the value of the input expression must be
         between the specified lower and upper bounds in any solution. Note that range constraints are stored
         internally as equality constraints. We add an extra variable to the model to capture the range information.
         Thus, the Sense attribute on a range constraint will always be GRB.EQUAL.
        """
        return self.model.addRange(expr, lower, upper, name)

    def addSOS(self, type, vars, wts=None):
        """
        Add an SOS constraint to the model.
        """
        return self.model.addSOS(type, vars, wts)

    def addVar(self, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None):
        """Add a decision variable to a model. """
        return self.model.addVar(lb, ub, obj, vtype, name, column)

    def addVars(self, *indexes, lb=0.0, ub=None, obj=0.0, vtype=None, name=""):
        """ Add multiple decision variables to a model. Returns a Gurobi tupledict object that contains the newly
        created variables. The keys for the tupledict are derived from the indices argument(s). """
        return self.model.addVars(*indexes, lb=lb, ub=ub, obj=obj, vtype=vtype, name=name)

    def cbCut(self, lhs, sense=None, rhs=None):
        """
        Add a new cutting plane to a MIP model from within a callback function. Note that this method can only be
        invoked when the where value on the callback function is equal to GRB.Callback.MIPNODE (see the Callback Codes
        section for more information). Cutting planes can be added at any node of the branch-and-cut tree.
        However, they should be added sparingly, since they increase the size of the relaxation model that is solved
        at each node and can significantly degrade node processing speed.  Cutting planes are typically used to cut
        off the current relaxation solution. To retrieve the relaxation solution at the current node, you should
        first call cbGetNodeRel.  When adding your own cuts, you must set parameter PreCrush to value 1. This setting
        shuts off a few presolve reductions that sometimes prevent cuts on the original model from being applied to
        the presolved model.
        """
        return self.model.cbCut(lhs, rhs, sense)

    def cbGet(self, what):
        """Query the optimizer from within the user callback. """
        return self.model.cbGet(what)

    def cbGetNodeRel(self, vars):
        """
        Retrieve values from the node relaxation solution at the current node. Note that this method can only be
        invoked when the where value on the callback function is equal to GRB.Callback.MIPNODE, and
        GRB.Callback.MIPNODE_STATUS is equal to GRB.OPTIMAL (see the Callback Codes section for more information).
        """
        return self.model.cbGetNodeRel(vars)

    def cbGetSolution(self, vars):
        """
        Retrieve values from the new MIP solution. Note that this method can only be invoked when the where value on the
        callback function is equal to GRB.Callback.MIPSOL or GRB.Callback.MULTIOBJ (see the Callback Codes
        section for more information).
        """
        return self.model.cbGetSolution(vars)

    def cbLazy(self, lhs, sense=None, rhs=None):
        """
        Add a new lazy constraint to a MIP model from within a callback function. Note that this method can only be
        invoked when the where value on the callback function is GRB.Callback.MIPNODE or GRB.Callback.MIPSOL
        (see the Callback Codes section for more information).

        Lazy constraints are typically used when the full set of constraints for a MIP model is too large to represent
        explicitly. By only including the constraints that are actually violated by solutions found during the
        branch-and-cut search, it is sometimes possible to find a proven optimal solution while only adding a fraction
        of the full set of constraints.

        You would typically add a lazy constraint by first querying the current node solution (by calling cbGetSolution
        from a GRB.CB_MIPSOL callback, or cbGetNodeRel from a GRB.CB_MIPNODE callback), and then calling cbLazy() to
        add a constraint that cuts off the solution. Gurobi guarantees that you will have the opportunity to cut off any
        solutions that would otherwise be considered feasible.

        Your callback should be prepared to cut off solutions that violate any of your lazy constraints, including those that
        have already been added. Node solutions will usually respect previously added lazy constraints, but not always.

        Note that you must set the LazyConstraints parameter to 1 if you want to use lazy constraints.
        """
        return self.model.cbLazy(lhs, rhs, sense)

    def cbSetSolution(self, vars, values):
        """
        Import solution values for a heuristic solution. Only available when the where value on the callback function
        is equal to GRB.CB_MIPNODE. (see the Callback Codes section for more information).

        When you specify a heuristic solution from a callback, variables initially take undefined values. You should use
        this method to specify variable values. You can make multiple calls to cbSetSolution from one callback
        invocation to specify values for multiple sets of variables. After the callback, if values have been specified
        for any variables, the Gurobi optimizer will try to compute a feasible solution from the specified values,
        possibly filling in values for variables whose values were left undefined. You can also optionally call
        cbUseSolution within your callback function to try to immediately compute a feasible solution from the
        specified values.
        """
        return self.model.cbSetSolution(vars, values)

    def cbStopOneMultiObj(self, objnum):
        """Interrupt the optimization process of one of the optimization steps in a multi-objective MIP problem without
        stopping the hierarchical optimization process. Only available for multi-objective MIP models and when the
        where member variable is not equal to GRB.Callback.MULTIOBJ (see the Callback Codes section for more
        information).
         """
        return self.model.cbStopOneMultiObj(objnum)

    def cbUseSolution(self):
        """
        Once you have imported solution values using cbSetSolution, you can optionally call cbUseSolution to
        immediately use these values to try to compute a heuristic solution.  Returns the objective value for the
        solution obtained from your solution values (or GRB.INFINITY if no improved solution is found).
        """
        return self.model.cbUseSolution()

    def chgCoeff(self, constr, var, newvalue):
        """
        Change one coefficient in the model. The desired change is captured using a Var object, a Constr object, and a
        desired coefficient for the specified variable in the specified constraint. If you make multiple changes to
        the same coefficient, the last one will be applied.

        Note that, due to our lazy update approach, the change won't actually take effect until you update the model
        (using Model.update), optimize the model (using Model.optimize), or write the model to disk (using Model.write).
        """
        return self.model.chgCoeff(constr, var, newvalue)

    def computeIIS(self):
        """
        Compute an Irreducible Inconsistent Subsystem (IIS). An IIS is a subset of the constraints and variable bounds
        with the following properties:
            - the subsystem represented by the IIS is infeasible, and
            - if any of the constraints or bounds of the IIS is removed, the subsystem becomes feasible.

        Note that an infeasible model may have multiple IISs. The one returned by Gurobi is not necessarily the one
        with minimum cardinality; there may exist others with fewer constraints or bounds.

        If an IIS computation is interrupted before completion, Gurobi will return the smallest IIS found to that point.

        This method populates the IISCONSTR, IISQCONSTR, and IISGENCONSTR constraint attributes, the IISSOS SOS
        attribute, and the IISLB, and IISUB variable attributes. You can also obtain information about the results of
        the IIS computation by writing an .ilp format file (see Model.write). This file contains only the IIS from the
        original model.

        Note that this method can be used to compute IISs for both continuous and MIP models.
        """
        return self.model.computeIIS()

    def copy(self):
        """
        Copy a model. Note that due to the lazy update approach in Gurobi, you have to call update() before copying it.
        """
        return self.model.copy()

    def discardConcurrentEnvs(self):
        """
        Discard concurrent environments for a model.

        The concurrent environments created by getConcurrentEnv will be used by every subsequent call to the
        concurrent optimizer until the concurrent environments are discarded.
        """
        return self.model.discardConcurrentEnvs()

    def discardMultiobjEnvs(self):
        """
        Discard all multi-objective environments associated with the model, thus restoring multi objective optimization
        to its default behavior.

        Please refer to the discussion of Multiple Objectives for information on how to specify multiple objective
        functions and control the trade-off between them.

        Use getMultiobjEnv() to create a multi-objective environment.
        """
        return self.model.discardMultiobjEnvs()

    def display(self, *args, **kwargs):
        return self.model.display(*args, **kwargs)

    def feasibility(self):
        """
        Return the feasibility version of the MIP model.
        """
        return self.model.feasibility()

    def feasRelax(self, relaxobjtype, minrelax, vars, lbpen, ubpen, constrs, rhspen):
        """
        Modifies the Model object to create a feasibility relaxation. Note that you need to call optimize on the result
        to compute the actual relaxed solution. Note also that this is a more complex version of this method -
        use feasRelaxS for a simplified version.

        The feasibility relaxation is a model that, when solved, minimizes the amount by which the solution violates
        the bounds and linear constraints of the original model. This method provides a number of options for
        specifying the relaxation.

        If you specify `relaxobjtype=0`, the objective of the feasibility relaxation is to minimize the sum of the
        weighted magnitudes of the bound and constraint violations. The lbpen, ubpen, and rhspen arguments specify
        the cost per unit violation in the lower bounds, upper bounds, and linear constraints, respectively.

        If you specify `relaxobjtype=1`, the objective of the feasibility relaxation is to minimize the weighted sum of
        the squares of the bound and constraint violations. The lbpen, ubpen, and rhspen arguments specify the
        coefficients on the squares of the lower bound, upper bound, and linear constraint violations, respectively.

        If you specify `relaxobjtype=2`, the objective of the feasibility relaxation is to minimize the weighted count of
        bound and constraint violations. The lbpen, ubpen, and rhspen arguments specify the cost of violating a lower
         bound, upper bound, and linear constraint, respectively.

        To give an example, if a constraint with rhspen value p is violated by 2.0, it would contribute `2*p` to the
        feasibility relaxation objective for `relaxobjtype=0`, it would contribute `2*2*p` for `relaxobjtype=1`, and it would
        contribute p for `relaxobjtype=2`.

        The `minrelax` argument is a boolean that controls the type of feasibility relaxation that is created. If
        `minrelax=False`, optimizing the returned model gives a solution that minimizes the cost of the violation.
        If `minrelax=True`, optimizing the returned model finds a solution that minimizes the original objective, but
        only from among those solutions that minimize the cost of the violation. Note that feasRelax must solve an
        optimization problem to find the minimum possible relaxation when `minrelax=True`, which can be quite expensive.

        Note that this is a destructive method: it modifies the model on which it is invoked. If you don't want to
        modify your original model, use copy() to create a copy before invoking this method.
        """
        return self.model.feasRelax(relaxobjtype, minrelax, vars, lbpen, ubpen, constrs, rhspen)

    def feasRelaxS(self, relaxobjtype, minrelax, vrelax, crelax):
        """
        Modifies the Model object to create a feasibility relaxation. Note that you need to call optimize on the
        result to compute the actual relaxed solution. Note also that this is a simplified version of this method -
        use feasRelax() for more control over the relaxation performed.

        The feasibility relaxation is a model that, when solved, minimizes the amount by which the solution violates
        the bounds and linear constraints of the original model. This method provides a number of options for
        specifying the relaxation.

        If you specify `relaxobjtype=0`, the objective of the feasibility relaxation is to minimize the sum of the
        magnitudes of the bound and constraint violations.

        If you specify `relaxobjtype=1`, the objective of the feasibility relaxation is to minimize the sum of the squares
        of the bound and constraint violations.

        If you specify `relaxobjtype=2`, the objective of the feasibility relaxation is to minimize the total number of
        bound and constraint violations.

        To give an example, if a constraint is violated by 2.0, it would contribute 2.0 to the feasibility relaxation
        objective for `relaxobjtype=0`, it would contribute 2.0*2.0 for `relaxobjtype=1`, and it would contribute 1.0
        for `relaxobjtype=2`.

        The `minrelax` argument is a boolean that controls the type of feasibility relaxation that is created.
        If `minrelax=False`, optimizing the returned model gives a solution that minimizes the cost of the violation.
        If `minrelax=True`, optimizing the returned model finds a solution that minimizes the original objective,
        but only from among those solutions that minimize the cost of the violation. Note that feasRelaxS() must solve
        an optimization problem to find the minimum possible relaxation when `minrelax=True`, which can be quite
        expensive.

        Note that this is a destructive method: it modifies the model on which it is invoked. If you don't want to
        modify your original model, use copy() to create a copy before invoking this method.
        """
        return self.model.feasRelaxS(relaxobjtype, minrelax, vrelax, crelax)

    def fixed(self):
        """
        Create the fixed model associated with a MIP model. The MIP model must have a solution loaded
        (e.g., after a call to the optimize method). In the fixed model, each integer variable is fixed to the value
        that variable takes in the MIP solution.
        """
        return self.model.fixed()

    def getAttr(self, attrname, objs=None):
        """
        Query the value of an attribute. When called with a single argument, it returns the value of a model attribute.
        When called with two arguments, it returns the value of an attribute for either a list or a dictionary
        containing either variables or constraints. If called with a list, the result is a list. If called with a
        dictionary, the result is a dictionary that uses the same keys, but is populated with the requested attribute
        values. The full list of available attributes can be found in the Attributes section.

        :raises AttributeError: Raises an AttributeError if the requested attribute doesn't exist or can't be queried.
        :param attrname: Name of the attribute.
        :param objs: List or dictionary containing either constraints or variables.
        :return: A list, dictionary or value of attribute.
        """
        return self.model.getAttr(attrname, objs)

    def getCoeff(self, constr, var):
        """
         Query the coefficient of variable var in linear constraint constr (note that the result can be zero).
        """
        return self.model.getCoeff(constr, var)

    def getCol(self, var):
        """
        Retrieve the list of constraints in which a variable participates, and the associated coefficients. The result
        is returned as a Column object.
        """
        return self.model.getCol(var)

    def getConcurrentEnv(self, num: int):
        """
        Create/retrieve a concurrent environment for a model.

        This method provides fine-grained control over the concurrent optimizer. By creating your own concurrent
        environments and setting appropriate parameters on these environments (e.g., the Method parameter), you can
        control exactly which strategies the concurrent optimizer employs. For example, if you create two concurrent
        environments, and set Method to primal simplex for one and dual simplex for the other, subsequent concurrent
        optimizer runs will use the two simplex algorithms rather than the default choices.

        Note that you must create contiguously numbered concurrent environments, starting with num=0. For example,
        if you want three concurrent environments, they must be numbered 0, 1, and 2.

        Once you create concurrent environments, they will be used for every subsequent concurrent optimization on that
        model. Use discardConcurrentEnvs to revert back to default concurrent optimizer behavior.
        """
        return self.model.getConcurrentEnv(num)

    def getConstrByName(self, name: str):
        """
        Retrieve a linear constraint from its name. If multiple linear constraints have the same name, this method
        chooses one arbitrarily.
        """
        return self.model.getConstrByName(name)

    def getConstrs(self):
        """ Retrieve a list of all linear constraints in the model. """
        return self.model.getConstrs()

    def getGenConstrAbs(self, genconstr):
        return self.model.getGenConstrAbs(genconstr)

    def getGenConstrAnd(self, genconstr):
        return self.model.getGenConstrAnd(genconstr)

    def getGenConstrIndicator(self, genconstr):
        return self.model.getGenConstrIndicator(genconstr)

    def getGenConstrMax(self, genconstr):
        return self.model.getGenConstrMax(genconstr)

    def getGenConstrMin(self, genconstr):
        return self.model.getGenConstrMin(genconstr)

    def getGenConstrOr(self, genconstr):
        return self.model.getGenConstrOr(genconstr)

    def getGenConstrs(self):
        return self.model.getGenConstrs()

    def getMultiobjEnv(self):
        return self.model.getMultiobjEnv()

    def getObjective(self):
        return self.model.getObjective()

    def getParamInfo(self, paramname):
        return self.model.getParamInfo(paramname)

    def getPWLObj(self, var):
        return self.model.getPWLObj(var)

    def getQConstrs(self):
        return self.model.getQConstrs()

    def getQCRow(self, qc):
        return self.model.getQCRow(qc)

    def getRow(self, constr):
        return self.model.getRow(constr)

    def getSOS(self, sos):
        return self.model.getSOS(sos)

    def getSOSs(self):
        return self.model.getSOSs()

    def getTuneResult(self):
        return self.model.getTuneResult()

    def getVarByName(self, name: str):
        return self.model.getVarByName(name)

    def getVars(self):
        return self.model.getVars()

    def linearize(self):
        return self.model.linearize()

    def message(self, msg):
        return self.model.message(msg)

    def optimize(self, callback=None):
        if callback is None:
            return self.model.optimize()
        else:
            return self.model.optimize(_wrap_callback(callback))

    def presolve(self):
        return self.model.presolve()

    def printAttr(self, attrname, filter):
        return self.model.printAttr(attrname, filter)

    def printQuality(self):
        return self.model.printQuality()

    def printStats(self):
        return self.model.printStats()

    def read(self, filename):
        return self.model.read(filename)

    def relax(self):
        return self.model.relax()

    def remove(self, items):
        return self.model.remove(items)

    def reset(self):
        return self.model.reset()

    def resetParams(self):
        return self.model.resetParams()

    def setAttr(self, attrname, newvalue):
        return self.model.setAttr(attrname, newvalue)

    def setObjective(self, expression, sense=None):
        return self.model.setObjective(expression, sense=sense)

    def setObjectiveN(self, expression, index):
        return self.model.setObjectiveN(expression, index)

    def setParam(self, paramname, newvalue):
        return self.model.setParam(paramname, newvalue)

    def setPWLObj(self, var, x, y):
        return self.model.setPWLObj(var, x, y)

    def terminate(self):
        return self.model.terminate()

    def tune(self):
        return self.model.tune()

    def update(self):
        self.model.update()

    def write(self, filename):
        self.model.write(filename)


class BaseGurobiModel(ModelWrapper):
    def __init__(self, name=""):
        super().__init__(name=name)
        self.__lonevars__ = []
        self.__intvars__ = []
        self.__binvars__ = []
        self.__ctsvars__ = []
        for attrname, attrtype in self.__annotations__.items():  # Black 3.7 magic
            if attrtype == BinVarDict:
                self.__binvars__.append(attrname)
            elif attrtype == BinVar:
                self.__binvars__.append(attrname)
                self.__lonevars__.append(attrname)
            elif attrtype == IntVarDict:
                self.__intvars__.append(attrname)
            elif attrtype == IntVar:
                self.__intvars__.append(attrname)
                self.__lonevars__.append(attrname)
            elif attrtype == CtsVarDict:
                self.__ctsvars__.append(attrname)
            elif attrtype == CtsVar:
                self.__ctsvars__.append(attrname)
                self.__lonevars__.append(attrname)
            elif attrtype in (VarDict, Var):
                raise TypeError("VarDict/Var should not be used to denote variables, use BinVarDict, IntVarDict or "
                                "CtsVarDict, or their *Var equivalents instead.")
            else:
                continue
            if attrname in self.__lonevars__:
                setattr(self, attrname, None)
                setattr(self, attrname + 'v', None)
            else:
                setattr(self, attrname, attrtype())
                setattr(self, attrname + 'v', dict())

        self.__intvars__ = tuple(self.__intvars__)
        self.__binvars__ = tuple(self.__binvars__)
        self.__ctsvars__ = tuple(self.__ctsvars__)
        self.__vars__ = self.__ctsvars__ + self.__binvars__ + self.__intvars__
        self.cut_cache = dict()
        self.cut_cache_size = 0
        self.cons: Dict[str, Dict[Any, Constr]] = dict()

    @property
    def cons_size(self):
        return {key: 1 if isinstance(val, Constr) else len(val) for key, val in self.cons.items()}

    def set_vars_attrs(self, **kwargs):
        """Convenience function to set the variable dictionaries on a model, checking if all have been provided. """
        a = set(kwargs.keys())
        b = set(self.__vars__)
        missing = b - a
        if len(missing) > 0:
            raise ValueError("Missing variables: ", ",".join(missing))
        extra = a - b
        if len(extra) > 0:
            raise ValueError("Undefined variables: ", ",".join(extra))
        for varname, vardict in kwargs.items():
            setattr(self, varname, vardict)

    def set_variables_continuous(self):
        for var_attr in self.__intvars__ + self.__binvars__:
            if var_attr in self.__lonevars__:
                getattr(self, var_attr).vtype = GRB.CONTINUOUS
            else:
                for var in getattr(self, var_attr).values():
                    var.vtype = GRB.CONTINUOUS

    def set_variables_integer(self):
        for var_attr in self.__intvars__ + self.__binvars__:
            vtype = GRB.BINARY if var_attr in self.__binvars__ else GRB.INTEGER
            if var_attr in self.__lonevars__:
                getattr(self, var_attr).vtype = vtype
            else:
                for var in getattr(self, var_attr).values():
                    var.vtype = vtype

    def get_iis_constraints(self):
        iis_keys = dict()
        for name, constrdict in self.cons.items():
            if isinstance(constrdict, Constr):
                if constrdict.IISConstr > 0:
                    iis_keys[name] = None
            else:
                keys = [k for k, cons in constrdict.items() if cons.IISConstr > 0]
                if len(keys) > 0:
                    iis_keys[name] = keys

        return iis_keys

    def update_var_values(self, where=None, eps=EPS):
        for var_attr in self.__vars__:
            val_attr = var_attr + 'v'
            if where is None:
                if var_attr in self.__lonevars__:
                    setattr(self, val_attr, getattr(self, var_attr).X)
                else:
                    setattr(self, val_attr,
                            {k: var.X for k, var in getattr(self, var_attr).items() if var.X > eps})

            elif where == GRB.Callback.MIPSOL:
                if var_attr in self.__lonevars__:
                    setattr(self, val_attr, self.cbGetSolution(getattr(self, var_attr)))
                else:
                    vardict = getattr(self, var_attr)

                    setattr(self, val_attr,
                            dict((k, v) for k, v in zip(vardict.keys(), self.cbGetSolution(list(vardict.values()))) if
                                 v > eps))
            elif where == GRB.Callback.MIPNODE:
                if var_attr in self.__lonevars__:
                    setattr(self, val_attr, self.cbGetNodeRel(getattr(self, var_attr)))
                else:
                    vardict = getattr(self, var_attr)
                    setattr(self, val_attr,
                            dict((k, v) for k, v in zip(vardict.keys(), self.cbGetNodeRel(list(vardict.values()))) if
                                 v > eps))
            else:
                raise ValueError("`where` is must be one of: None, GRB.Callback.MIPSOL, GRB.Callback.MIPNODE")

    def flush_cut_cache(self):
        total = 0
        for constraint_name in self.cut_cache:
            if isinstance(self.cut_cache[constraint_name], dict):
                self.cons[constraint_name] = dict()
                for idx, cut in self.cut_cache[constraint_name].items():
                    self.cons[constraint_name][idx] = self.addConstr(cut)
                    total += 1
            else:
                self.cons[constraint_name] = [self.addConstr(cut) for cut in self.cut_cache[constraint_name]]
                total += 1
        self.cut_cache.clear()
        self.cut_cache_size = 0
        return total

    def get_gurobi_model_information(self) -> GurobiModelInformation:
        kwargs = {}
        for attr in dataclasses.fields(GurobiModelInformation):

            try:
                val = attr.type(self.model.getAttr(INFO_ATTR_TO_MODEL_ATTR[attr.name]))
            except AttributeError:
                val = None
            kwargs[attr.name] = val
        return GurobiModelInformation(**kwargs)

    def _add_cut_to_cache(self, cut, cache, cache_key=None):
        if cache_key is not None:
            if cache not in self.cut_cache:
                self.cut_cache[cache] = dict()
            self.cut_cache[cache][cache_key] = cut
            self.cut_cache_size += 1
        else:
            if cache not in self.cut_cache:
                self.cut_cache[cache] = deque()
            self.cut_cache[cache].append(cut)
            self.cut_cache_size += 1

    def cbCut(self, cut: TempConstr, cache: str = None, cache_key=None):
        super().cbCut(cut)
        if cache is not None:
            self._add_cut_to_cache(cut, cache, cache_key)

    def cbLazy(self, cut: TempConstr, cache: str = None, cache_key=None):
        super().cbLazy(cut)
        if cache is not None:
            self._add_cut_to_cache(cut, cache, cache_key)

    def temp_params(self, **param_val_pairs):
        return TempModelParameters(self, **param_val_pairs)


class TempModelParameters:
    """
    Context manager which temporarily sets parameters and restores them after.
    """

    def __init__(self, model: BaseGurobiModel, **param_val_pairs):
        self.model = model
        self.old_parameters = dict()
        self.new_parameters = param_val_pairs
        for param, new_val in param_val_pairs.items():
            _, _, old_val, _, _, _ = model.getParamInfo(param)
            self.old_parameters[param] = old_val

    def __enter__(self):
        for param, val in self.new_parameters.items():
            self.model.setParam(param, val)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for param, val in self.old_parameters.items():
            self.model.setParam(param, val)
