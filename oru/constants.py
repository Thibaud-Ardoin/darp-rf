import subprocess
import json

def _get_conda_info():
    return json.loads(subprocess.check_output(["conda", "info", "--json"], text=True))

_GUROBI_MODEL_ATTR = {
    "NumConstrs": "num_constr",
    "NumVars": "num_vars",
    "NumSOS": "num_sos",
    "NumQConstrs": "num_quad_constr",
    "NumGenConstrs": "num_gen_constr",
    "DNumNZs": "num_nz",
    "NumQNZs": "num_quad_nz",
    "NumQCNZs": "num_quad_coeff_nz",
    "NumIntVars": "num_int_vars",
    "NumBinVars": "num_bin_vars",
    "NumPWLObjVars": "num_pwl_vars",
    "ModelName": "model_name",
    "ModelSense": "model_sense",
    "ObjCon": "obj_const",
    "ObjVal": "obj_val",
    "ObjBound": "obj_bnd",
    "ObjBoundC": "obj_bnd_raw",
    "PoolObjBound": "pool_obj_bnd",
    "PoolObjVal": "pool_obj_val",
    "MIPGap": "mip_gap",
    "Runtime": "model_time",
    "Status": "status",
    "SolCount": "sol_count",
    "IterCount": "simplex_iters",
    "BarIterCount": "barr_iters",
    "NodeCount": "num_nodes",
    "IsMIP": "is_mip",
    "IsQP": "is_qp",
    "IsQCP": "is_qcp",
    "IsMultiObj": "is_multiobj",
    "IISMinimal": "iis_minimal",
    "MaxCoeff": "max_coeff",
    "MinCoeff": "min_coeff",
    "MaxBound": "max_var_bnd",
    "MinBound": "min_var_bnd",
    "MaxObjCoeff": "max_obj_coeff",
    "MinObjCoeff": "min_obj_coeff",
    "MaxRHS": "max_rhs",
    "MinRHS": "min_rhs",
    "MaxQCCoeff": "max_qc_coeff",
    "MinQCCoeff": "min_qc_coeff",
    "MaxQCLCoeff": "max_qc_lin_coeff",
    "MinQCLCoeff": "min_qc_lin_coeff",
    "MaxQCRHS": "max_qc_rhs",
    "MinQCRHS": "min_qc_rhs",
    "MaxQObjCoeff": "max_q_obj_coeff",
    "MinQObjCoeff": "min_q_obj_coeff",
    "KappaExact": "kappa",
    "FarkasProof": "farkas_proof",
    "TuneResultCount": "tune_result_cnt",
    "NumStart": "num_start",
}

INFO_ATTR_TO_MODEL_ATTR = {v: k for k, v in _GUROBI_MODEL_ATTR.items()}
EPS=1e-4
CONDA_INFO=_get_conda_info()
