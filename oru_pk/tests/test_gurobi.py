import gurobi
import random
from oru import grb
import tempfile
import os
from gurobi import GRB

class ExampleModel(grb.BaseGurobiModel):
    X: grb.BinVarDict

    def __init__(self):
        super().__init__()
        random.seed(1337)
        X = [self.addVar(vtype=GRB.BINARY, obj=(1 + random.random() * 10)) for _ in range(100)]
        constr = []
        for _ in range(5):
            var_mask = [random.random() < 0.3 for _ in range(len(X))]
            constr.append(self.addConstr(
                gurobi.quicksum(x for x, m in zip(X, var_mask) if m) <= sum(var_mask) - 1
            ))
        constr.append(self.addConstr(gurobi.quicksum(x * random.randint(5, 20) for x in X) <= 100))
        self.set_vars_attrs(X=dict(enumerate(X)))
        self.setParam("OutputFlag", 0)
        self.cons['cons'] = constr


def test_info_serialisation():
    model = ExampleModel()
    model.optimize()
    info = model.get_gurobi_model_information()
    fp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    fp.close()
    info.to_json_file(fp.name)
    info_copy = grb.GurobiModelInformation.from_json_file(fp.name)
    os.remove(fp.name)
    assert info == info_copy

def test_variable_switching():
    model = ExampleModel()
    model.optimize()
    assert model.IsMIP == 1
    assert all(var.vtype == GRB.BINARY for _, var in model.X.items())
    model.set_variables_continuous()
    model.optimize()
    assert model.IsMIP == 0
    assert all(var.vtype == GRB.CONTINUOUS for _, var in model.X.items())
    model.set_variables_integer()
    model.optimize()
    assert model.IsMIP == 1
    assert all(var.vtype == GRB.BINARY for _, var in model.X.items())
