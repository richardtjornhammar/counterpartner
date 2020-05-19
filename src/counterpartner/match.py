"""
Copyright 2020 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd

def unpack( seq ) :
    if isinstance(seq, (list, tuple, set)):
        yield from (x for y in seq for x in unpack(y))
    elif isinstance(seq, dict):
        yield from (x for item in seq.items() for y in item for x in unpack(y))
    else:
        yield seq

def match_dataframe ( cohort_df , standardize = False ,
                      include_only = None , exclude = None ) :
    import scipy.spatial.distance as sc_
    import scipy.special as ss_
    #
    cohort_df_ = cohort_df.copy()
    #
    inc_pair_df,exc_pair_df,inc_pair_types,exc_pair_types = None,None,None,None
    if 'dict' in str( type( include_only ) ) :
        if ('label' in include_only and 'pair' in include_only) :
            inc_label = include_only['label']     # str
            inc_pair_types = include_only['pair']  # list
            inc_pair_df = cohort_df.loc[:,[inc_label]].copy()
            cohort_df = cohort_df.loc[ : , \
                [ c for c in cohort_df.columns if not str(c) in inc_label] ].copy()
    #
    if 'dict' in str( type( exclude ) ) :
        if ('label' in exclude and 'pair' in exclude) :
            exc_label = exclude['label']     # str
            exc_pair_types = exclude['pair']  # list
            exc_pair_df = cohort_df.loc[:,[exc_label]].copy()
            cohort_df = cohort_df.loc[ : , \
                [ c for c in cohort_df.columns if not str(c) in exc_label] ].copy()
    #
    if standardize :
        cohort_df = cohort_df.apply(lambda X: (X-np.mean(X))/np.std(X) )
    pdvals = sc_.pdist ( cohort_df.values ) # THE VALUES
    N = len(cohort_df)
    #
    # LOOKUP
    lookup = {}
    pidx2lidx = lambda n,i,j :int(ss_.binom(n,2)-ss_.binom(n-i,2)+(j-i-1))
    for i in range(N) :
        for j in range(i+1,N) :
            lookup[pidx2lidx(N,i,j)] = (i,j)
    ip_ = [ lookup[i_] for i_ in range(len(pdvals)) ] # THE ADDRESS
    #
    min2max_pairs  = [ s[1] for s in sorted ( [ (v,pidx) for (v,pidx) in zip ( pdvals,ip_ ) ] ) ]
    min2max_values = [ s[0] for s in sorted ( [ (v,pidx) for (v,pidx) in zip ( pdvals,ip_ ) ] ) ]
    #
    allowed , disallowed = { tuple(p) for p in min2max_pairs } , {}
    if not inc_pair_types is None :
        allowed    = { tuple(list(p)):1 for p in min2max_pairs if set([v[0] for v in inc_pair_df.iloc[list(p)].values])==set(inc_pair_types) }
    if not exc_pair_types is None :
        disallowed = { tuple(p):1 for p in min2max_pairs if set([v[0] for v in exc_pair_df.iloc[list(p)].values])==set(exc_pair_types) }
    #
    min2max_pairs = [ p for p in min2max_pairs if tuple(p) in allowed and not tuple(p) in disallowed ]
    matched = [ p for p in unpack([ min2max_pairs[0] ]) ] # FIRST IS SIMPLE
    matched_pairs = [ min2max_pairs[0] ]
    for pair in min2max_pairs :
        smp = set(matched)
        if pair[0] in smp or pair[1] in smp :
            continue
        [ matched.append( p ) for p in unpack(pair) ]
        matched_pairs.append(pair)
    #
    matched_df = None
    for m, im in zip(matched_pairs,range(len(matched_pairs))) :
        tdf = cohort_df_.iloc[[m[0],m[1]],:]
        tdf .loc[:,'MP.ID'] = im
        if matched_df is None :
            matched_df = tdf
        else :
            matched_df = pd.concat( [matched_df,tdf] )
    return ( matched_df )


def create_matched_selection ( use_information, exc , inc , sdf=None , flagged = set ( [''] ) ) :
    desc_ = """
        use_information = [ 'Gender' , 'Type' , 'Age' , 'BMI [kg/m2]' ]
        exc_dict = { 'label':'Gender' , 'pair':['Male','Female'] }
        inc_dict = { 'label':'Type'   , 'pair':['NGT','T2D'] }
        print ( create_matched_selection ( use_information , exc=gdict , inc=ddict ) )
    """
    if sdf is None :
        print ( 'FAILED' )
        return ( None )
    p_df = match_dataframe ( sdf.loc[[idx for idx in sdf.index if not idx in flagged],use_information ],
            standardize  = True  , include_only = ddict , exclude = gdict )
    p_df .loc[:,'name'] = p_df.index.values
    add_cols = [c for c in sdf.columns if c not in p_df]
    for c in add_cols :
        p_df.loc[:,c] = sdf.loc[p_df.index.values,c].values
    matchlist_pdf = p_df .loc[:,['name','MP.ID']].groupby('MP.ID').apply(lambda x: '.'.join([str(y[0]) for y in x.values])).values
    return ( p_df , matchlist_pdf )


def run_example() :
    #
    create_cohort_group = lambda n , s_ , v_: \
            np.random.rand(n*len(v_)).reshape(n,len(s_))*s_+v_
    #
    N , NG = 30 , 4
    cg1 = create_cohort_group( N,[2,5],[27,60] )  # M H
    cg2 = create_cohort_group( N,[5,7],[27,63] )  # M S
    cg3 = create_cohort_group( N,[2,5],[24,60] )  # F H
    cg4 = create_cohort_group( N,[5,7],[26,64] )  # F s
    cohort_df = pd.DataFrame( [ *cg1 , *cg2 , *cg3 , *cg4 ] )
    cohort_df.index = ['Name' + str(i_) for i_ in range(len(cohort_df)) ]

    print ( 'Here' )
    print ( match_dataframe( cohort_df ) )
    print ( 'Here' )
    print ( match_dataframe( cohort_df, standardize=True ) )
    print ( 'Here' )

    cohort_df.loc[:,'D'] = [c for c in 'H'*N + 'S'*N + 'H'*N + 'S'*N ]
    cohort_df.loc[:,'G'] = [c for c in 'M'*N*2 + 'F'*N*2 ]

    print ( cohort_df )
    print ( match_dataframe( cohort_df , standardize=True ,
                include_only = {'label':'D','pair':['H','S']} ,
                exclude = {'label':'G' , 'pair':['M','F']} ) )


if __name__ == '__main__' :
    run_example()
