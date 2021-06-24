import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from .utils.data import modify, DARP_Data
from darp import *
from .oru.grb import BinVarDict, IntVar, CtsVar
from .oru import slurm
from gurobi import *
from typing import Tuple
from collections import OrderedDict

from darp_restricted_fragments import (DARPRestrictedFragmentsExperiment,
                                       RestrictedFragmentGenerator,
                                       DARP_RF_Model_Params,
                                       Network,
                                       write_info_file,
                                       DARP_RF_Model,
                                       main_callback,
                                       build_chains,
                                       HEURISTIC_MODEL_PARAMETERS,
                                       MASTER_MODEL_PARAMETERS,
                                       F_DBG)


def run_rf_algo(file_nb):

    Chain = Tuple[ResFrag]

    CUT_FUNC_LOOKUP = dict()
    VALID_INEQUALITIES = set()
    LAZY_CUTS = set()

    def cut(name: str, valid_inequality : bool):
        def decorator(func):
            global CUT_FUNC_LOOKUP, VALID_INEQUALITIES, LAZY_CUTS
            assert name not in CUT_FUNC_LOOKUP, f"{name} is already defined."
            CUT_FUNC_LOOKUP[name] = func
            if valid_inequality:
                VALID_INEQUALITIES.add(name)
            else:
                LAZY_CUTS.add(name)
            return func
        return decorator


    stopwatch = Stopwatch()
    exp = DARPRestrictedFragmentsExperiment.from_cl_args(file_nb)
    exp.parameters['NS_heuristic'] = False
    exp.parameters['FR_heuristic'] = False
    exp.parameters['heuristic_stop_gap'] = 1
    exp.parameters['triad_cuts'] = False
    exp.parameters['AF_cuts'] = False
    exp.parameters['FF_cuts'] = False
    exp.parameters['FA_cuts'] = False
    exp.parameters['cut_violate'] = False
    exp.parameters['ap2c'] = False
    # get_output_path

    exp.print_summary_table()
    exp.write_index_file()
    data = exp.data
    # noinspection PyArgumentList
    model_custom_params = DARP_RF_Model_Params(
        sdarp=exp.inputs['sdarp'],
        ns_heuristic=exp.parameters['NS_heuristic'],
        fr_heuristic=exp.parameters['FR_heuristic'],
        heuristic_stop_gap=exp.parameters['heuristic_stop_gap'],
        triad_cuts=exp.parameters['triad_cuts'],
        AF_cuts=exp.parameters['AF_cuts'],
        FF_cuts=exp.parameters['FF_cuts'],
        FA_cuts=exp.parameters['FA_cuts'],
        cut_violate=exp.parameters['cut_violate'],
        apriori_2cycle=exp.parameters['ap2c']
    )

    stopwatch.start()
    data = tighten_time_windows(data)
    data = remove_arcs(data)
    stopwatch.lap('data_preprocess')
    network = Network(data,
                      fragment_domination=exp.parameters['domination'],
                      filter_fragments=exp.parameters['filter_rf'])
    network.build()
    stopwatch.lap('network_build')
    obj_info = {}
    write_info_file(exp, stopwatch.times.copy(),network, obj_info)

    model = DARP_RF_Model(network, model_custom_params, cpus=exp.parameters['cpus'])
    stopwatch.lap('model_build')
    model.set_variables_continuous()
    model.optimize()
    obj_info['lb_root_lp'] = model.ObjVal
    cut_counter = defaultdict(int)

    write_info_file(exp, stopwatch.times.copy(),network, obj_info,model=model)

    num_cuts_added = float('inf')
    output = TablePrinter(list(CUT_FUNC_LOOKUP.keys()) + ['objective'])


    with model.temp_params(OutputFlag=0):
        while True:
            if num_cuts_added == float('inf'):  # first iteration only.
                output.print_line(model.ObjVal, pad_left=True)
            else:
                model.optimize()

            num_cuts_added = 0
            model.update_var_values()
            cuts_this_iter = []
            for cutname in CUT_FUNC_LOOKUP:
                if getattr(model.parameters, cutname):
                    separate_func = CUT_FUNC_LOOKUP[cutname]
                    cuts = separate_func(model, violate=model.parameters.cut_violate)
                    if cutname in VALID_INEQUALITIES:
                        add_valid_inequalities(model, None, cuts, cutname)
                    else:
                        add_lazy_cuts(model, None, cuts, cutname)
                    ncuts = len(cuts)
                    cut_counter[cutname + '_lp'] += ncuts
                    num_cuts_added += ncuts
                else:
                    ncuts = 0
                cuts_this_iter.append(ncuts)

            if num_cuts_added == 0:
                break

            output.print_line(*[x if x > 0 else '' for x in cuts_this_iter], model.ObjVal)

    obj_info['lb_lifted_lp'] = model.ObjVal
    stopwatch.lap('rootcut')

    # write_info_file(exp, stopwatch.times.copy(),network, obj_info,cut_counter, model)

    # Reduced-cost heuristic
    for param, value in MASTER_MODEL_PARAMETERS.items():
        model.setParam(param, value)
    for param, value in exp.parameters['gurobi'].items():
        model.setParam(param, value)

    if model.parameters.sdarp:
        for w in model.W.values():
            w.BranchPriority = 10
    else:
        model.Z.BranchPriority = 10

    if exp.parameters['rc_frac'] > 0:
        reduced_costs = sorted([(var.RC, f) for f, var in model.X.items()], key=lambda x: x[0])
        reduced_costs_cutoff_idx = int(exp.parameters['rc_frac'] * len(reduced_costs))
        rc_heur = model.addConstr(quicksum(model.X[f] for _, f in reduced_costs[reduced_costs_cutoff_idx:]) == 0)

        model.nonzero_fragments = set(f for _, f in reduced_costs[:reduced_costs_cutoff_idx])
        model.set_variables_integer()
        model.update()
        with model.temp_params(TimeLimit=(exp.parameters['timelimit']-stopwatch.time)/2, MIPGap=0.01):
            model.optimize(main_callback)

        # noinspection PyUnreachableCode
        if __debug__:
            if model.Status == GRB.OPTIMAL:
                model.update_var_values()
                rc_heur_soln = {'X': model.Xv.copy(), 'Y': model.Yv.copy()}
                rc_heur_chains, _ = build_chains(model)
                for F, _, _ in rc_heur_chains:
                    pprint_path(tuple(i for f in F for i in f.path), data)

        model.remove(rc_heur)
        model.nonzero_fragments = None
        print(f'Adding {model.cut_cache_size:d} constraints as hard constraints.')
        for cutname, cutdict in model.cut_cache.items():
            cut_counter[cutname + '_rc_mip'] = len(cutdict)


        ub_rc_mip = model.lc_best_obj

        model.flush_cut_cache()
        if exp.parameters['rc_fix'] and ub_rc_mip < float('inf'):
            model.set_variables_continuous()
            with model.temp_params(OutputFlag=0):
                model.optimize()
            gap = ub_rc_mip - model.ObjVal
            n_fixed = 0
            for var in model.X.values():
                if var.rc > gap + 0.01:
                    var.ub = 0
                    n_fixed += 1
            print(f"RC-fixing: fixed {n_fixed} variables")
            model.set_variables_integer()

        obj_info['ub_rc_mip'] = model.lc_best_obj
    else:
        model.set_variables_integer()

    stopwatch.lap('rc_mip')

    write_info_file(exp, stopwatch.times.copy(),network, obj_info,cut_counter, model)
    if __debug__:
        model.set_variables_continuous()
        model.optimize()
        model.update_var_values()
        chains, cycles = build_chains(model)
        for F, A, val in chains:
            print(f"{val:.5f} " + " ".join(map(lambda x : f"{str(x):30s}", F)))
        model.set_variables_integer()

    with model.temp_params(TimeLimit=exp.parameters['timelimit']-stopwatch.time):
        model.optimize(main_callback)
    stopwatch.stop('main_mip')

    for cutname, cutdict in model.cut_cache.items():
        cut_counter[cutname + '_main_mip'] = len(cutdict)

    obj_info['ub_final'] = model.ObjVal
    obj_info['lb_final'] = model.ObjBound

    if model.SolCount > 0:
        soln = DARPSolution(model.ObjVal)
        model.update_var_values()
        chains, cycles = build_chains(model)
        paths = {}
        assert len(cycles) == 0
        for k, r in enumerate(chains):
            r_frags, r_arcs, _ = r
            paths[k] = tuple(i for f in r_frags for i in f.path)
            print("Vehicle", k)
            if len(r) == 0:
                continue
            print(*map(str, r_frags))
            pprint_path(paths[k], data, add_depots=True)
            assert get_early_schedule(paths[k], data) is not None, "illegal solution"
            soln.add_route(paths[k])
        soln.to_json_file(exp.outputs['solution'])

    print()
    output = TablePrinter(('Section', ' Time (s)'), min_col_width=24)
    times = stopwatch.times.copy()
    times['total'] = sum(times.values())
    for k, v in times.items():
        output.print_line(k, v)

    write_info_file(exp, times,network, obj_info,cut_counter, model)
    return exp.outputs['solution']
