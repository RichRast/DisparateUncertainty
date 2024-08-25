import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc
import numpy as np

def getFSMetrics(PRP_rel: np.ndarray,
        PRP_group_ids: np.ndarray,
        PRP_ranking: np.ndarray,
        alpha=0.1,
        p=0.5) -> np.ndarray:
    """
    params PRP_rel: relevance array of n items (n, 1) in sorted order
    params PRP_group_ids: group membership of n items (n,) in sorted order
    params PRP_ranking: candidate ids of n items in sorted order

    
    return 
    ranking : (n,)
    """
    num_items = PRP_rel.shape[0]

    fair = fsc.Fair(num_items, p, alpha)
    
    #PRP_ranking, PRP_group_ids, PRP_rel
    unfair_ranking=[]
    for i, p in enumerate(PRP_ranking):
        unfair_ranking.append(FairScoreDoc(p, PRP_rel[i], bool(PRP_group_ids[i])))
    re_ranked = fair.re_rank(unfair_ranking)
    ranking = np.full((num_items,),0, dtype=int)
    for i,r in enumerate(re_ranked):
        ranking[i]=r.id
        
    return ranking