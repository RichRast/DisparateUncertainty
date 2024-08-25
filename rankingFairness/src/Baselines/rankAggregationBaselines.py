import numpy as np
import random
import math
import pandas as pd
import itertools
from itertools import combinations, permutations
from copy import deepcopy
from tqdm import tqdm

"""
Reference: Kcachel et al 2023, Paper:"Fairer Together"
"""

def calc_exposure_ratio(ranking, group_ids):
    """
    Reference: Singh et al, 2018; Kcachel et al 2023
    """

    unique_grps, grp_count_items = np.unique(group_ids, return_counts=True)
    num_items = len(ranking)
    exp_vals = exp_at_position_array(num_items)
    grp_exposures = np.zeros_like(unique_grps, dtype=np.float64)
    for i in range(0,num_items):
        grp_of_item = group_ids[i]
        exp_of_item = exp_vals[i]
        #update total group exp
        grp_exposures[grp_of_item] += exp_of_item

    avg_exp_grp = grp_exposures / grp_count_items
    expdpp = np.min(avg_exp_grp)/np.max(avg_exp_grp) #ratio based
    return expdpp, avg_exp_grp

def exp_at_position_array(num_items):
    return np.array([(1/(np.log2(i+1))) for i in range(1,num_items+1)])    

def epik(base_ranks, item_ids, group_ids, bnd):
    """
    Reference: Kcachel et al 2023
    Function perform fair exposure kemeny rank aggregation.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :param bnd: Desired minimum exposure ratio of consensus ranking
    :return: consensus: A numpy array of item ides representing the consensus ranking. ranking_group_ids: a numy array of
    group ids corresponding to the group membership of each item in the consensus.
    """

    # Declare and initialize model
    m = gp.Model('EPiK')

    # Create decision variable (every pair)
    items = np.unique(base_ranks[0]).tolist()
    num_voters, num_items = np.shape(base_ranks)
    item_strings = [str(var) for var in item_ids]
    group_strings = [str(var) for var in group_ids]
    item_grpid_combo_strings = list(zip(item_strings, group_strings))
    pair_combinations = [(i, j) for i in item_strings for j in item_strings] #all possible
    x = m.addVars(pair_combinations, vtype = GRB.BINARY,  name="pair")
    x = m.addVars(pair_combinations, name="pair", vtype=gp.GRB.BINARY)
    m.addConstrs((x[r,r] == 0 for r in item_strings), name="zeroselfpairs")


    # Ordering constraint Xab +Xba = 1
    print("making strict ordering.....")
    unique_pairs = [(str(i), str(j)) for i, j in combinations(range(np.shape(base_ranks)[1]), 2)] #count 01, and 10 only once
    m.addConstrs((x[a,b]+x[b,a] == 1 for a,b in unique_pairs if a != b), name='strict_order')

    # Prevent cycles constraint
    print("making cycle prevention.....")
    m.addConstrs((x[a, b] + x[b, c] + x[c,a] <= 2 for a, b in unique_pairs if a != b for c in item_strings if c != a and c != b), name='stopcycles')

    # Objective function
    print("starting objective function.....")
    pair_agreements = precedence_matrix_agreement(base_ranks)
    pair_agreement_list = pair_agreements.ravel()

    print("making objective function dictionary.....")
    #Make dictionary for Gurobi
    pair_weights = {}
    iter = 0
    for (i, j) in pair_combinations:
        pair_weights[(i, j)] = pair_agreement_list[iter]
        iter += 1
    pair_combinations, scores = gp.multidict(pair_weights)

    print("setting objective function.....")
    m.setObjective(x.prod(scores), GRB.MAXIMIZE)


    #Group Fairness
    unique_grp_ids, size_grp = np.unique(group_ids, return_counts = True)
    num_groups = len(unique_grp_ids)


    posofitem = m.addVars([i for i in item_strings], name="posofitem-id")
    m.addConstrs(((num_items - x.sum(r, '*')) + 1 == posofitem[r] for r in item_strings ),
                name='pair2pos')  # add one for the log term
    l = m.addVars([i for i in item_strings], name="logofposforitem-id")
    for r in item_strings:
        m.addGenConstrLogA(posofitem[r], l[r], 2, "logarithm" + str(r))


    #make exposure variables e[item, grp]
    e = m.addVars(item_grpid_combo_strings, name="expofitem-id-grp") #E[item][group]

    m.addConstrs((l[r]*e[r,grp] == 1 for r, grp in item_grpid_combo_strings),
                name='exposure')

    g = m.addVars([str(grp) for grp in unique_grp_ids], name = "groupexp-grp")


    m.addConstrs((e.sum('*', str(grp)) == g[str(grp)] for grp in unique_grp_ids),
                    name='sumgrpexposure')

    ag = m.addVars([str(grp) for grp in unique_grp_ids], name="avggroupexp-grp")

    m.addConstrs((g[str(grp)] / size_grp[np.argwhere(unique_grp_ids == grp).flatten()[0]] == ag[str(grp)] for grp in
                unique_grp_ids), name="avgexpofgroup")


    group_tuples = list(permutations([str(g) for g in unique_grp_ids], 2))
    g_ratio = m.addVars(group_tuples, name = "ratioavgexpgrps-grp-grp")

    m.addConstrs((ag[j]*g_ratio[i,j] == ag[i] for i, j in group_tuples), name = 'ratio-avg-grp-exps')

    m.addConstrs((g_ratio[i,j] >= bnd for i, j in group_tuples), name= "lowerb-groupexpratio")
    m.addConstrs((g_ratio[i, j] <= (1/bnd) for i, j in group_tuples), name="upperb-groupexpratio")

    # m.write('epikCS.lp')
    print("starting optimization.....")
    # Run optimization engine
    m.params.NonConvex = 2
    m.optimize()

    # Uncoment to display optimal values of decision variables
    # print("Printing variables....")
    # for v in m.getVars():
    #     if v.x > 1e-6:
    #         print(v.varName, v.x)

    #extract solution
    rank_pairs = [var.varName for var in m.getVars() if var.x == 1 and var.varName.startswith('pair')]
    winning_items = [(var.split(',')[0]).split('[')[1] for var in rank_pairs]
    result = [item for items, c in Counter(winning_items).most_common()
            for item in [items] * c]
    consensus = list(unique_everseen(result))
    consensus = list(map(int, consensus))
    bottom_candidate = [item for item in range(0, num_items) if item not in consensus]
    consensus.append(bottom_candidate[0])

    ranking_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus]
    return np.asarray(consensus), ranking_group_ids

    def precedence_matrix_disagreement(baseranks):
        """
        :param baseranks: num_rankers x num_items
        :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # disagreements with i over j
        """
        num_rankers, num_items = baseranks.shape


        weight = np.zeros((num_items, num_items))

        pwin_cand = np.unique(baseranks[0]).tolist()
        plose_cand = np.unique(baseranks[0]).tolist()
        combos = [(i, j) for i in pwin_cand for j in plose_cand]
        for combo in combos:
            i = combo[0]
            j = combo[1]
            h_ij = 0 #prefer i to j
            h_ji = 0 #prefer j to i
            for r in range(num_rankers):
                if np.argwhere(baseranks[r] == i)[0][0] > np.argwhere(baseranks[r] == j)[0][0]:
                    h_ij += 1
                else:
                    h_ji += 1

            weight[i, j] = h_ij
            weight[j, i] = h_ji
            np.fill_diagonal(weight, 0)
        return weight



def precedence_matrix_agreement(baseranks):
    """
    :param baseranks: num_rankers x num_items
    :return: precedence matrix of disagreeing pair weights. Index [i,j] shows # agreements with i over j
    """
    num_rankers, num_items = baseranks.shape


    weight = np.zeros((num_items, num_items))

    pwin_cand = np.unique(baseranks[0]).tolist()
    plose_cand = np.unique(baseranks[0]).tolist()
    combos = [(i, j) for i in pwin_cand for j in plose_cand]
    for combo in combos:
        i = combo[0]
        j = combo[1]
        h_ij = 0 #prefer i to j
        h_ji = 0 #prefer j to i
        for r in range(num_rankers):
            if np.argwhere(baseranks[r] == i)[0][0] < np.argwhere(baseranks[r] == j)[0][0]:
                h_ij += 1
            else:
                h_ji += 1

        weight[i, j] = h_ij
        weight[j, i] = h_ji
        np.fill_diagonal(weight, 0)
    return weight

def pre_proc_kem(base_ranks, item_ids, group_ids, bnd):
    """
    Function to pre-process input rankings to Kemeny to be fair.
    :param base_ranks: Assumes zero index
    :param item_ids: Assumes zero index
    :param group_ids: Assumes zero index
    :return: consensus: A numpy array
    """

    # perform exposure fairness on each ranking
    n_voters, n_items = np.shape(base_ranks)
    fair_base_ranks = np.zeros_like(base_ranks, dtype = int)
    for r in range(0,n_voters):
        base_rank = base_ranks[r,:]
        base_rank = np.reshape(base_rank, (1, len(base_rank)))
        fair_base_rank, _ = epiRA(base_rank, item_ids, group_ids, bnd, True, "Copeland")
        print("fair base rank", fair_base_rank)
        fair_base_ranks[r, :] = fair_base_rank

    print("fair base ranks in pre fair", fair_base_ranks)

    #perform kemeny
    result, ranking_group_ids = kemeny(fair_base_ranks, item_ids, group_ids)

    return np.asarray(result), ranking_group_ids

def epiRA(base_ranks, item_ids, group_ids, bnd, grporder, current_ranking=None, current_group_ids=None, agg_method=None, verbose=False):
    """
    Function to perform fair exposure rank aggregation via post-processing a voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
        in item_ids.
    :param bnd: Desired minimum exposure ratio of consensus ranking
    :param grporder: True - re orders consensus ranking to preserve within group order. False does not preserve within group order.
    :param agg_method: String indicating which voting rule to use. 'Kemeny', 'Copeland', 'Schulze', 'Borda', 'Maximin'.
    :return: consensus: A numpy array of item ides representing the consensus ranking. ranking_group_ids: a numy array of
        group ids corresponding to the group membership of each item in the consensus.
    """

    
    if agg_method is None:
        assert (current_ranking is not None) and (current_group_ids is not None)
        num_items=len(item_ids)
        consensus = deepcopy(current_ranking)
        consensus_group_ids = deepcopy(current_group_ids)

    else:
        num_voters, num_items = np.shape(base_ranks)
    if agg_method == "Copeland":
        consensus, consensus_group_ids, current_ranking, current_group_ids = copeland(base_ranks, item_ids, group_ids)

    if agg_method == "Kemeny":
        kemeny_r, kemeny_group_ids = kemeny(base_ranks, item_ids, group_ids)
        consensus = list(kemeny_r)
        current_ranking = np.asarray(consensus)
        consensus_group_ids = np.asarray(kemeny_group_ids)
        current_group_ids = consensus_group_ids

    if agg_method == "Borda":
        consensus, consensus_group_ids, current_ranking, current_group_ids = borda(base_ranks, item_ids, group_ids)

    if agg_method == "Schulze":
        consensus, consensus_group_ids, current_ranking, current_group_ids = schulze(base_ranks, item_ids, group_ids)
    if agg_method == "Maximin":
        consensus, consensus_group_ids, current_ranking, current_group_ids = maximin(base_ranks, item_ids, group_ids)



    cur_exp, avg_exps = calc_exposure_ratio(current_ranking, current_group_ids)
    exp_at_position = np.array([(1 / (np.log2(i + 1))) for i in range(1, num_items + 1)])
    repositions = 0


    swapped = np.full(len(current_ranking), False) #hold items that have been swapped
    while( cur_exp < bnd ):

        # Prevent infinite loops
        if repositions > ((num_items * (num_items - 1)) / 2):
            print("Try increasing the bound")
            # return current_ranking, current_group_ids
            break

        max_avg_exp = np.max(avg_exps)
        grp_min_avg_exp = np.argmin(avg_exps) #group id of group with lowest avg exposure
        grp_max_avg_exp = np.argmax(avg_exps)  # group id of group with lowest avg exposure
        grp_min_size = np.sum(group_ids == grp_min_avg_exp)
        Gmin_positions = np.argwhere(current_group_ids == grp_min_avg_exp).flatten()
        Gmax_positions = np.argwhere(current_group_ids == grp_max_avg_exp).flatten()

        indx_highest_grp_min_item = np.min(Gmin_positions)
        valid_Gmax_items = Gmax_positions < indx_highest_grp_min_item

        if np.sum(valid_Gmax_items) == 0:
            Gmin_counter = 1
            while np.sum(valid_Gmax_items) == 0:
                next_highest_ranked_Gmin = np.min(Gmin_positions[Gmin_counter:, ])
                valid_Gmax_items = Gmax_positions < next_highest_ranked_Gmin
                Gmin_counter += 1
            indx_highest_grp_min_item = next_highest_ranked_Gmin
        if swapped[indx_highest_grp_min_item] == True: #swapping same item
            #valid_grp_min = np.argwhere(~swapped & current_group_ids == grp_min_avg_exp).flatten()
            valid_grp_min = np.intersect1d(np.argwhere(~swapped).flatten(),np.argwhere(current_group_ids == grp_min_avg_exp).flatten())
            if len(valid_grp_min) != 0: indx_highest_grp_min_item = np.min(valid_grp_min)  # index of highest ranked item that was not swapped
        highest_item_exp = exp_at_position[indx_highest_grp_min_item]
        exp_grp_min_without_highest = (np.min(avg_exps) * grp_min_size) - highest_item_exp

        boost = (grp_min_size*max_avg_exp*bnd) - exp_grp_min_without_highest

        exp = np.copy(exp_at_position) #deep copy
        exp[np.argwhere(current_group_ids == grp_min_avg_exp).flatten()] = np.Inf
        exp[indx_highest_grp_min_item] = np.Inf #added 11/21
        indx = (np.abs(exp - boost)).argmin() #find position with closest exposure to boost

        min_grp_item = current_ranking[indx_highest_grp_min_item]
        # print("min_grp_item",min_grp_item)
        swapping_item = current_ranking[indx]
        # print("swapping_item", swapping_item)
        #put swapping item in min_grp_item position
        current_ranking[indx_highest_grp_min_item] = swapping_item
        #put min_group_item at indx
        current_ranking[indx] = min_grp_item
        repositions += 1
        swapped[indx_highest_grp_min_item] = True
        swapped[indx] = True
        #update group ids
        current_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in current_ranking]
        #set up next loop
        cur_exp, avg_exps = calc_exposure_ratio(current_ranking, current_group_ids)
        # print("exposure after swap:", cur_exp)

    if grporder == True: #Reorder to preserve consensus
        # pass
        consensus = np.asarray(consensus)
        current_ranking = np.ones(num_items, dtype = int)
        current_group_ids = np.asarray(current_group_ids)
        for g in np.unique(group_ids).tolist():
            where_to_put_g = np.argwhere(current_group_ids == g).flatten()
            g_ordered = consensus[np.argwhere(consensus_group_ids == g).flatten()] #order in copeland
            current_ranking[where_to_put_g] = g_ordered
        current_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in current_ranking]
        
        if verbose:
            print("exposure achieved with RA:", cur_exp)
        return current_ranking, np.asarray(current_group_ids), cur_exp



    return np.asarray(current_ranking), np.asarray(current_group_ids)



def copeland(base_ranks, item_ids, group_ids):
    """
    Function to perform copeland voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    copelandDict = {key: 0 for key in items_list}
    pair_agreements = precedence_matrix_agreement(base_ranks)
    for item in items_list:
        for comparison_item in items_list:
            if item != comparison_item:
                num_item_wins = pair_agreements[comparison_item, item]
                num_comparison_item_wins = pair_agreements[item, comparison_item]
                if num_item_wins < num_comparison_item_wins:
                    copelandDict[item] += 1

    items = list(copelandDict.keys())
    copeland_pairwon_cnt = list(copelandDict.values())
    zip_scores_items = zip(copeland_pairwon_cnt, items)
    sorted_pairs = sorted(zip_scores_items, reverse=True)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids


def borda(base_ranks, item_ids, group_ids):
    """
    Function to perform borda voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    item_list = list(item_ids)
    bordaDict = {key: 0 for key in item_list}
    num_rankings, num_items = base_ranks.shape
    points_per_pos_legend = list(range(num_items - 1, -1, -1))

    for ranking in range(0, num_rankings):
        for item_pos in range(0, num_items):
            item = base_ranks[ranking, item_pos]
            bordaDict[item] += points_per_pos_legend[item_pos]

    candidates = list(bordaDict.keys())
    borda_scores = list(bordaDict.values())
    zip_scores_items = zip(borda_scores, candidates)
    sorted_pairs = sorted(zip_scores_items, reverse=True)
    consensus = [element for _, element in sorted_pairs] #borda
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids


def maximin(base_ranks, item_ids, group_ids):
    """
    Function to perform borda voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    maximinDict = {key: 0 for key in items_list}
    pair_agreements = precedence_matrix_agreement(base_ranks)
    for item in items_list:
        max_item_wins = 0
        for comparison_item in items_list:
           if item != comparison_item:
              num_item_wins = pair_agreements[comparison_item, item]
              max_item_wins = max(max_item_wins, num_item_wins)

        maximinDict[item] += max_item_wins

    items = list(maximinDict.keys())
    maximin_score = list(maximinDict.values())
    zip_scores_items = zip(maximin_score, items)
    sorted_pairs = sorted(zip_scores_items, reverse=False)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids



def schulze(base_ranks, item_ids, group_ids):
    """
    Function to perform schulze voting rule.
    :param base_ranks: Assumes zero index. Numpy array of # voters x # items representing the base rankings.
    :param item_ids: Assumes zero index. Numpy array of item ids.
    :param group_ids: Assumes zero index. Numyp array of group ids correpsonding to the group membership of each item
    in item_ids.
    :return: Consensus: list of items ids in copeland ranking, consensus_group_ids: numpy array of group ids corresponding to consensus list,
    current_ranking: numpy array of ids of copeland ranking, current_group_ids: numpy array of group ids for copeland ranking
    """
    items_list = list(item_ids)
    Qmat = precedence_matrix_agreement(base_ranks)
    Pmat = np.zeros_like(Qmat)

    for i in items_list:
       for j in items_list:
           if i != j:
              if Qmat[j, i] > Qmat[i, j]:
               Pmat[j, i] = Qmat[j, i]
              else:
               Pmat[j, i] = 0

    for i in items_list:
       for j in items_list:
           if i != j:
              for k in items_list:
                if i != k and j != k:
                  Pmat[k, j] = np.maximum(Pmat[k, j], np.minimum(Pmat[i, j], Pmat[k, i]))

    wins_candidate_has_over_others = np.sum(Pmat, axis=0)
    zip_scores_items = zip(wins_candidate_has_over_others, items_list)
    sorted_pairs = sorted(zip_scores_items, reverse=False)
    consensus = [element for _, element in sorted_pairs]
    consensus_group_ids = np.asarray([group_ids[np.argwhere(item_ids == i)[0][0]] for i in consensus])
    current_ranking = np.asarray(consensus)
    current_group_ids = consensus_group_ids
    return consensus, consensus_group_ids, current_ranking, current_group_ids

def calc_consensus_accuracy(base_ranks, consensus):
    agree_count = 0
    n_voters, n_items = np.shape(base_ranks)
    precedence_mat = precedence_matrix_agreement(base_ranks)
    positions = len(consensus)
    for pos in range(positions):
        won = consensus[pos]
        lost = consensus[pos + 1: positions]
        for x in lost:
            agree_count += precedence_mat[won, x]

    print("agree count", agree_count)
    print("sum precedence_mat", np.sum(precedence_mat))
    result = agree_count/np.sum(precedence_mat)
    return result

def RAPF(base_ranks, item_ids, group_ids, seed):
    np.random.seed(seed)  # for reproducibility
    num_voters, num_items = np.shape(base_ranks) #pick a perm
    rand = random.randint(0,num_voters-1)
    rank = base_ranks[rand,:]
    group = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in rank]

    numberOfItem = len(rank)

    rankGrp = {}
    for i in range(0, len(rank)):
        rankGrp[rank[i]] = group[i]

    grpCount = {}
    for i in group:
        grpCount[i] = 0

    rankGrpPos = {}
    for i in rank:
        grpCount[rankGrp[i]] = grpCount[rankGrp[i]] + 1
        rankGrpPos[i] = grpCount[rankGrp[i]]

    rankRange = {}
    for item in rank:
        i = rankGrpPos[item]
        n = numberOfItem
        fp = grpCount[rankGrp[item]]
        r1 = math.floor((i - 1) * n / fp) + 1
        r2 = math.ceil(i * n / fp)
        if r2 > numberOfItem:
            r2 = numberOfItem
        rankRange[item] = (r1, r2)

    B = nx.Graph()
    top_nodes = []
    bottom_nodes = []

    for i in rank:
        top_nodes.append(i)
        bottom_nodes.append(str(i))
    B.add_nodes_from(top_nodes, bipartite=0)
    B.add_nodes_from(bottom_nodes, bipartite=1)

    for i in rank:
        r1, r2 = rankRange[i]
        # print(r1,r2)
        for j in range(1, numberOfItem + 1):
            if j >= r1 and j <= r2:
                # print(i,j)
                B.add_edge(i, str(j), weight=abs(i - j))
            else:
                B.add_edge(i, str(j), weight=100000000000)
                # print(i,j)

    my_matching = nx.algorithms.bipartite.minimum_weight_full_matching(B, top_nodes, "weight")

    print(my_matching)

    vy = list(my_matching.keys())  # v @ position y, where y is zero-indexed
    v = vy[0:num_items]
    y = vy[num_items:num_items * 2]

    ranking = np.zeros(num_items, dtype=int)
    for ind_i in range(0, num_items):
        ranking[int(y[ind_i]) - 1] = v[ind_i]

    ranking_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in ranking]
    return ranking, np.asarray(ranking_group_ids)