from hypothesis import settings
from hypothesis.strategies import composite, floats, integers, lists, permutations


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


# Checks
# Whether the draws of relevances means converge to the mean of distribution it is drawn from. It checks whether the sampling of relevances is done currently
# Check the EO_constraint calculation for a given ranking
# check the majority, moniority and total cost for a given ranking and a given d number of draws of relevances
# check the utility (DCG) given a particular ranking
# check that for uniform ranking - given the variance of cost, the majority and minority cost are somewhat equal
# check that for EOR - given the variance of cost, the majority and minority cost are somewhat equal
# check that PRP always has the highest utility for a given dataset

@composite
def getK(draw, num_docs):
    return draw(integers(min_value=1, max_value=num_docs))

small_K = getK(num_docs=8)

