from __future__ import annotations
import random
from dataclasses import dataclass, field
from itertools import product
from cost_oracle import AlgorithmConfig, PublicProblem
from ant_colony_system import geometric_distances, geometric_nn, cycle_cost, actual_cost

# Adapted from Rego, Gamboa and Glover (2016).

ALGORITHM_ID = 'DR16-Core-Geo-NoDB-v1'

SOURCE_ID = 'Rego_Gamboa_Glover_2016_10.1002_net.21676__Glover1996_DR'

SOURCE_VERSION = 'directed_DR_core__ADAPTATION_operational_spec_1.0.1'

POLICIES = {'initialization_policy': 'geometric_NN_actual_n_arcs',
            'candidate_policy': 'all_legitimate_directed_DR_ejections',
            'stopping_policy': 'declared_root_schedule_or_explicit_round_budget'}

ADAPTATIONS = ('geometric_NN_initial_tour', 'one_geometric_non_tour_chord_per_origin',
               'deterministic_greedy_chain_and_canonical_ties',
               'explicit_chain_depth_and_outer_round_budget',
               'declared_pair_history_and_terminal_closure_scope',
               'separate_Double_Bridge_omitted')

def make_config(*, seed, chain_depth, max_rounds):
    if type(seed) is not int or any(type(v) is not int or v <= 0 for v in (chain_depth, max_rounds)):
        raise ValueError('Explicit integer seed and positive chain_depth/max_rounds required')
    return AlgorithmConfig(ALGORITHM_ID, SOURCE_ID, SOURCE_VERSION, 'G', 'ADAPTATION', seed,
        **POLICIES, parameters={'chain_depth': chain_depth, 'max_rounds': max_rounds,
                               'operational_spec_version': '1.0.1'}, adaptations=ADAPTATIONS)

def validate_inputs(public, config, rng):
    if not isinstance(public, dict) or set(public) != {'node_ids','coordinates','depot','scenario','cost_unit'}:
        raise ValueError('Only the public problem fields are permitted')
    problem = PublicProblem(tuple(public['node_ids']), tuple(map(tuple, public['coordinates'])),
                            public['depot'], public['scenario'], public['cost_unit'])
    fixed = {'algorithm_id': ALGORITHM_ID, 'source_id': SOURCE_ID, 'source_version': SOURCE_VERSION,
             'mode': 'G', 'fidelity': 'ADAPTATION', **POLICIES}
    if any(config.get(k) != v for k,v in fixed.items()) or tuple(config.get('adaptations',())) != ADAPTATIONS:
        raise ValueError('Identity, source and adaptation policies are fixed')
    params = config.get('parameters',{})
    if not isinstance(params,dict) or set(params) != {'chain_depth','max_rounds','operational_spec_version'}:
        raise ValueError('Missing or unauthorized operational parameter')
    if params['operational_spec_version'] != '1.0.1' or any(type(params[k]) is not int or params[k] <= 0 for k in ('chain_depth','max_rounds')):
        raise ValueError('Positive explicit budgets and fixed specification version required')
    if type(config.get('seed')) is not int or type(rng) is not random.Random or rng.getstate() != random.Random(config['seed']).getstate():
        raise ValueError('Configured fresh random.Random required; DR does not consume it')
    return problem, params

def component(nodes, arcs, start):
    adjacency = {v: set() for v in nodes}
    for a,b in arcs:
        adjacency[a].add(b); adjacency[b].add(a)
    seen = {start}; pending = [start]
    while pending:
        v = pending.pop()
        for w in adjacency[v]-seen:
            seen.add(w); pending.append(w)
    return frozenset(seen)

@dataclass(frozen=True)
class Reference:
    nodes: tuple
    arcs: frozenset
    incoming_root: int = field(init=False)
    outgoing_root: int = field(init=False)

    def __post_init__(self):
        nodes = tuple(self.nodes); raw = tuple(tuple(x) for x in self.arcs)
        if len(nodes) < 3 or any(type(v) is not int for v in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError('Distinct integer reference nodes required')
        if len(raw) != len(nodes)+1 or len(set(raw)) != len(raw):
            raise ValueError('Reference must have exactly n+1 distinct directed arcs')
        if any(len(a) != 2 or any(type(v) is not int or v not in nodes for v in a) or a[0] == a[1] for a in raw):
            raise ValueError('Invalid directed reference arc')
        indegree = {v: 0 for v in nodes}; outdegree = dict(indegree)
        for a,b in raw: outdegree[a] += 1; indegree[b] += 1
        ins = [v for v in nodes if indegree[v] == 2]
        outs = [v for v in nodes if outdegree[v] == 2]
        if len(ins) != 1 or len(outs) != 1 or any(indegree[v] not in (1,2) or outdegree[v] not in (1,2) for v in nodes):
            raise ValueError('Reference must have one incoming and one outgoing excess root')
        if len(component(nodes,raw,nodes[0])) != len(nodes):
            raise ValueError('Disconnected reference')
        object.__setattr__(self,'nodes',tuple(sorted(nodes)))
        object.__setattr__(self,'arcs',frozenset(raw))
        object.__setattr__(self,'incoming_root',ins[0]); object.__setattr__(self,'outgoing_root',outs[0])

    @property
    def form(self):
        if self.incoming_root == self.outgoing_root: return 'coincident_bicycle'
        for role,root,s,old in incidences(self):
            if len(component(self.nodes,self.arcs-{old},root)) != len(self.nodes): return 'bicycle'
        return 'tricycle'

def incidences(state):

    for a,b in sorted(state.arcs):
        if b == state.incoming_root: yield 0,b,a,(a,b)
    for a,b in sorted(state.arcs):
        if a == state.outgoing_root: yield 1,a,b,(a,b)

@dataclass(frozen=True)
class History:
    deleted: frozenset = frozenset()
    inserted: frozenset = frozenset()
    protected: frozenset = frozenset()

    def advance(self, move):

        protect = {move.added} if move.removed in self.inserted else set()
        return History(self.deleted|{move.removed}, self.inserted|{move.added}, self.protected|protect)

@dataclass(frozen=True)
class Ejection:
    role: int
    root: int
    subroot: int
    replacement: int
    removed: tuple
    added: tuple
    reference: Reference
    incidence_type: str

    @property
    def tie_key(self): return self.role,self.subroot,self.replacement

def ejections(state, history):

    for role,root,s,old in incidences(state):
        if old in history.protected: continue
        part = component(state.nodes,state.arcs-{old},root)
        cyclic = len(part) == len(state.nodes)
        for j in state.nodes:
            new = (s,j) if role == 0 else (j,s)
            if j == s or new in state.arcs or new in history.deleted: continue
            if not cyclic and j not in part: continue
            changed = Reference(state.nodes,state.arcs-{old}|{new})
            expected_i = j if role == 0 else state.incoming_root
            expected_o = j if role == 1 else state.outgoing_root
            if (changed.incoming_root,changed.outgoing_root) != (expected_i,expected_o):
                raise AssertionError('Source ejection did not move the declared directed root')
            yield Ejection(role,root,s,j,old,new,changed,'cycle' if cyclic else 'noncycle')

def cycle_from_arcs(nodes, arcs, depot):

    nodes = tuple(nodes); arcs = tuple(arcs)
    if depot not in nodes or len(arcs) != len(nodes): raise ValueError('Invalid cycle size/start')
    successor = {}; incoming = set()
    for a,b in arcs:
        if a not in nodes or b not in nodes or a == b or a in successor or b in incoming:
            raise ValueError('Cycle degrees/arc invalid')
        successor[a] = b; incoming.add(b)
    walked = [depot]
    for _ in range(len(nodes)-1): walked.append(successor[walked[-1]])
    if len(set(walked)) != len(nodes) or successor[walked[-1]] != depot:
        raise ValueError('Disconnected subtours')
    return tuple(walked)

@dataclass(frozen=True)
class Trial:
    removed: tuple
    added: tuple
    cycle: tuple

def trials(state, depot):

    incoming = sorted(a for a in state.arcs if a[1] == state.incoming_root)
    outgoing = sorted(a for a in state.arcs if a[0] == state.outgoing_root)
    definitions = []
    root_arc = state.outgoing_root,state.incoming_root
    if root_arc in state.arcs: definitions.append(((root_arc,),()))
    for old_in,old_out in product(incoming,outgoing):
        new = old_in[0],old_out[1]
        if old_in == old_out or new[0] == new[1] or new in state.arcs: continue
        definitions.append(((old_in,old_out),(new,)))
    seen = set()
    for removed,added in definitions:
        final = state.arcs-set(removed)|set(added)
        try: cycle = cycle_from_arcs(state.nodes,final,depot)
        except ValueError: continue
        if cycle not in seen:
            seen.add(cycle); yield Trial(removed,added,cycle)

def score_ejection(move, client, context):
    return actual_cost(client,*move.added,'search',{**context,'role':'added'})-actual_cost(client,*move.removed,'search',{**context,'role':'removed'})

def score_trial(trial, reference_cost, client, context):
    value = reference_cost
    for a,b in trial.added: value += actual_cost(client,a,b,'search',{**context,'role':'trial_added'})
    for a,b in trial.removed: value -= actual_cost(client,a,b,'search',{**context,'role':'trial_removed'})
    return value

def run_chain(cycle, tour_cost, chord, client, *, depth, context):
    if type(depth) is not int or depth <= 0: raise ValueError('Positive explicit chain depth required')
    base = frozenset(zip(cycle,cycle[1:]+cycle[:1]))
    if chord in base or chord[0] == chord[1] or any(v not in cycle for v in chord):
        raise ValueError('A valid directed non-tour chord is required')
    state = Reference(tuple(cycle),base|{chord})
    reference_cost = tour_cost+actual_cost(client,*chord,'search',{**context,'phase':'initial_DR_chord'})
    history = History(inserted=frozenset({chord}))
    counts = {'evaluated_ejections':0,'evaluated_trials':0,'committed_ejections':0}
    best = (tour_cost,tuple(cycle),-1,-1)
    client.record('start',{**context,'phase':'chain','chord':list(chord),'base_tour':list(cycle),
        'base_cost':tour_cost,'reference_arcs':sorted(state.arcs),'reference_cost':reference_cost})
    reason = 'explicit_chain_depth'
    for level in range(depth+1):
        for order,trial in enumerate(trials(state,cycle[0])):
            details = {**context,'phase':'trial','level':level,'trial_order':order}
            value = score_trial(trial,reference_cost,client,details)
            counts['evaluated_trials'] += 1
            client.record('candidate',{**details,'removed':trial.removed,'added':trial.added,
                'reference_cost':reference_cost,'tour':list(trial.cycle),'actual_cost':value})
            key = value,trial.cycle,level,order
            if key < best: best = key
        if level == depth: break
        scored = []
        for order,move in enumerate(ejections(state,history)):
            details = {**context,'phase':'ejection_candidate','level':level+1,'order':order}
            delta = score_ejection(move,client,details)
            counts['evaluated_ejections'] += 1
            client.record('candidate',{**details,'role_index':move.role,'root':move.root,'subroot':move.subroot,
                'replacement':move.replacement,'incidence_type':move.incidence_type,
                'removed':move.removed,'added':move.added,'delta':delta})
            scored.append(((delta,*move.tie_key),move))
        if not scored:
            reason = 'empty_legitimate_ejection_set'; break
        key,chosen = min(scored,key=lambda row:row[0])
        before = reference_cost; reference_cost += key[0]
        history = history.advance(chosen); state = chosen.reference
        counts['committed_ejections'] += 1
        client.record('candidate',{**context,'phase':'ejection_commit','level':level+1,
            'removed':chosen.removed,'added':chosen.added,'delta':key[0],
            'reference_cost_before':before,'reference_cost':reference_cost,
            'reference_arcs':sorted(state.arcs),'incoming_root':state.incoming_root,
            'outgoing_root':state.outgoing_root,'form':state.form,
            'deleted':sorted(history.deleted),'inserted':sorted(history.inserted),'protected':sorted(history.protected)})
    client.record('stop',{**context,'phase':'chain','reason':reason,'best_cost':best[0],
        'best_tour':list(best[1]),'best_level':best[2],'counts':counts})
    return best[1],best[0],counts

def root_seeds(cycle, distances):
    successor = dict(zip(cycle,cycle[1:]+cycle[:1]))
    return tuple((u,min((v for v in cycle if v != u and v != successor[u]),
                        key=lambda v:(distances[u,v],v))) for u in sorted(cycle))

def solve(public_problem, config, client, rng):
    problem,parameters = validate_inputs(public_problem,config,rng)
    distances = geometric_distances(problem); cycle = geometric_nn(problem,distances)
    value = cycle_cost(cycle,client,'initialization',{'phase':'geometric_NN_actual_length'})
    client.record('start',{'phase':'solver','algorithm_id':ALGORITHM_ID,'fidelity':'ADAPTATION',
        'initial_tour':list(cycle),'actual_cost':value,'separate_double_bridge':False})
    totals = {'evaluated_ejections':0,'evaluated_trials':0,'committed_ejections':0,
              'chains':0,'accepted_compound_moves':0}
    reason = 'explicit_round_budget_exhausted'
    for round_id in range(parameters['max_rounds']):
        seeds = root_seeds(cycle,distances); accepted = False
        client.record('start',{'phase':'round','round':round_id,'root_seeds':seeds,'tour':list(cycle),'actual_cost':value})
        for seed_index,chord in enumerate(seeds):
            changed,new_cost,stats = run_chain(cycle,value,chord,client,depth=parameters['chain_depth'],
                context={'round':round_id,'seed_index':seed_index})
            totals['chains'] += 1
            for key,number in stats.items(): totals[key] += number
            if new_cost < value:
                client.record('move_accepted',{'phase':'compound','round':round_id,'seed_index':seed_index,
                    'old_tour':list(cycle),'new_tour':list(changed),'old_cost':value,'new_cost':new_cost,
                    'delta':new_cost-value})
                cycle,value = changed,new_cost; totals['accepted_compound_moves'] += 1; accepted = True
                client.record('restart',{'phase':'outer_round','after_round':round_id})
                break
            client.record('move_rejected',{'phase':'compound','round':round_id,'seed_index':seed_index,
                'reason':'no_strict_actual_improvement','candidate_cost':new_cost,'current_cost':value})
        if not accepted:
            reason = 'declared_root_schedule_exhausted'; break
    client.record('stop',{'phase':'solver','reason':reason,'actual_cost':value,'tour':list(cycle),'counts':totals})
    return {'tour':list(cycle)+[problem.depot],'stopping_reason':reason,
        'diagnostics':{'algorithm_id':ALGORITHM_ID,'fidelity':'ADAPTATION','mode':'G',
            'actual_objective_ticks':value,'counts':totals,'rounds_attempted':round_id+1,
            'certificate_scope':'none_global_or_unrestricted_DR_local','separate_double_bridge':False,
            'rng_draws':0}}
