import os
import sys
import argparse
import time
import random
import itertools
import threading
import numpy as np
import pandas as pd
from multiprocessing import  Pool
import json
import awkward
import uproot
import concurrent.futures
import ROOT
import glob
import h5py
import random
import errno
import pickle
import logging
#import dask.dataframe as dd
#import dask
#from multiprocessing.pool import ThreadPool

from utilities import python_mkdir

ROOT.gROOT.SetBatch()

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert')
parser.add_argument('rootDir', type=str, 
                    help='Directory of input root files')
parser.add_argument('convertDir', type=str,
                    help='Output directory')

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

NTHREADS = 16
parallel = True
vectorInput = False

inDir = args.rootDir
outDir = args.convertDir
if os.path.exists(outDir):
    print(outDir, 'already exists')
    sys.exit(0)

python_mkdir(outDir)

fnames = glob.glob('{}/*.root'.format(inDir))
treename = 'muonTree/MuonTree'
# must create these branches, they are what is output
out_truth = ['pion','muon']
# weight bins
weight_bins = [
    # muon_innerTrack_p
    np.array(
        [1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,30,40,50,100,1000],
        dtype=float
    ),
    # muon_innerTrack_abseta (need to create)
    np.array(
        [0.0,0.5,0.9,1.2,1.566,2.322,2.5], # motivated by segmentation changes in endcap
        dtype=float
    ),
]
weight_bin_labels = ['muon_innerTrack_p','muon_innerTrack_abseta']
weight_bin_axis_labels = [r'Track $p_{T}$', r'Track $|\eta|$']
weight_branches = ['muon_innerTrack_p','muon_innerTrack_eta']
reference = 'muon'
# these are helper branches (perhaps truth info) that are not output
#other_branches = ['muon_gen_matches_pion','muon_gen_matches_muon','muon_gen_deltaR']
other_branches = ['muon_gen_sim_pdgId','muon_pt']
# these are the branches that are output
branches = [
    #'muon_innerTrack_pt',
    'muon_innerTrack_p',
    'muon_innerTrack_eta',
    'muon_innerTrack_phi',
    'muon_innerTrack_qoverp',
    'muon_innerTrack_qoverpError',
    'muon_innerTrack_validFraction',
    'muon_innerTrack_highPurity',
    'muon_innerTrack_hitPattern_trackerLayersWithMeasurement',
    'muon_innerTrack_hitPattern_pixelLayersWithMeasurement',
    'muon_isolationR03_nTracks',
    'muon_isolationR03_sumPt',
    #'muon_caloCompatibility',
    #'muon_calEnergy_ecal_time',
    'muon_calEnergy_em',
    'muon_calEnergy_emMax',
    'muon_calEnergy_emS25',
    'muon_calEnergy_emS9',
    'muon_calEnergy_had',
    'muon_calEnergy_hadMax',
    'muon_calEnergy_hadS9',
    #'muon_calEnergy_hcal_time',
    'muon_calEnergy_ho',
    'muon_calEnergy_hoS9',
    #'muon_calEnergy_tower',
    #'muon_calEnergy_towerS9',
    'muon_calEnergy_hcal_ieta',
    'muon_calEnergy_hcal_iphi',
    'muon_calEnergy_crossedHadRecHits_ieta',
    'muon_calEnergy_crossedHadRecHits_iphi',
    'muon_calEnergy_crossedHadRecHits_depth',
    'muon_calEnergy_crossedHadRecHits_energy',
    'muon_calEnergy_crossedHadRecHits_time',
    'muon_calEnergy_crossedHadRecHits_chi2',
]
branch_groupings = [
    [b for b in branches if 'muon_calEnergy_crossedHadRecHits' not in b],
    [b for b in branches if 'muon_calEnergy_crossedHadRecHits' in b],
]
branch_lengths = {b: 15 for b in branches if 'muon_calEnergy_crossedHadRecHits' in b}
linear_branches = {
    #'muon_innerTrack_p': [1.5,1000],
}

# get weights
print('Calculating weights')
distributions = {}
for fname in fnames:
    print(fname)
    for df in uproot.iterate(fname,treename,other_branches+weight_branches,entrysteps=1000000000,outputtype=pd.DataFrame):
        if vectorInput: df = pd.DataFrame({k: list(itertools.chain.from_iterable(df[k])) for k in df.columns})
        df['muon_innerTrack_abseta'] = df['muon_innerTrack_eta'].abs()
        df = df[(df['muon_gen_sim_pdgId'].abs()==13) | (df['muon_gen_sim_pdgId'].abs()==211)].copy()
        df['muon'] = df['muon_gen_sim_pdgId'].abs()==13
        df['pion'] = df['muon_gen_sim_pdgId'].abs()==211

        for truth in out_truth:
            hist, xedges, yedges = np.histogram2d(
                df[df[truth]==1][weight_bin_labels[0]],
                df[df[truth]==1][weight_bin_labels[1]],
                weight_bins
            )
            if truth in distributions:
                distributions[truth] = distributions[truth]+hist
            else:
                distributions[truth] = hist

def divide_distributions(a,b):
    out = np.array(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i][j] = a[i][j]/b[i][j] if b[i][j] else 1
    return out

# normalize
for truth in out_truth:
    distributions[truth] = distributions[truth]/distributions[truth].sum()

weight_distributions = {}
for truth in out_truth:
    weight_distributions[truth] = divide_distributions(distributions[reference],distributions[truth])

def plot_hist(hist,outname):
    H=hist.T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, H)
    fig.colorbar(im, ax=ax)
    ax.set_xscale("log", nonposx='clip')
    plt.xlabel(weight_bin_axis_labels[0])
    plt.ylabel(weight_bin_axis_labels[1])
    fig.savefig(outname)
    plt.close()
for truth in out_truth:
    plot_hist(weight_distributions[truth],'{}/weight_{}.png'.format(outDir,truth))

# get means
print('Calculating means')
means_sum = {key:[] for key in branches}
varis_sum = {key:[] for key in branches}
for arrays in uproot.iterate(fnames[0],treename,branches+other_branches,entrysteps=100000000):
    means = {}
    varis = {}
    for key in arrays:
        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])
    for key in arrays:
        if key.decode('utf-8') not in branches: continue
        # normalize sumpt
        if key.decode('utf-8')=='muon_isolationR03_sumPt':
            arrays[key] = arrays[key]/arrays[b'muon_pt']
        # shift ieta, iphi to be centered at 0
        if key.decode('utf-8')=='muon_calEnergy_crossedHadRecHits_ieta':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_ieta']
        if key.decode('utf-8')=='muon_calEnergy_crossedHadRecHits_iphi':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_iphi']
        # normalize hcal digi energy to total hcal energy
        if key.decode('utf-8')=='muon_calEnergy_crossedHadRecHits_energy':
            arrays[key] = arrays[key]/arrays[key].sum()
        # broken for some reason
        #means[key] = arrays[key][~np.isnan(arrays[key])].flatten().mean()
        #varis[key] = arrays[key][~np.isnan(arrays[key])].flatten().var()
        a = arrays[key]
        while isinstance(a,awkward.JaggedArray): a = a.flatten()
        # change default no time/chi2 value to closer to real values
        if key.decode('utf-8')=='muon_calEnergy_crossedHadRecHits_time':
            a[a<-20] = -20
        if key.decode('utf-8')=='muon_calEnergy_crossedHadRecHits_chi2':
            a[a<-5] = -5
        means[key] = a[~np.isnan(a)].mean()
        varis[key] = a[~np.isnan(a)].var()
        means_sum[key.decode('utf-8')] += [means[key]]
        varis_sum[key.decode('utf-8')] += [varis[key]]
means = {key: np.array(means_sum[key]).mean() for key in branches}
varis = {key: np.array(varis_sum[key]).mean() for key in branches}
stds  = {key: np.sqrt(np.array(varis_sum[key]).mean()) for key in branches}

for key in sorted(means):
    print(key,means[key],stds[key])

result = {
    'means':{key:float(item) for key,item in means.items()},
    'stds':{key:float(item) for key,item in stds.items()},
    'linear': linear_branches,
}
with open('{}/means.json'.format(outDir),'w') as f:
    json.dump(result,f)


def parallel_apply(df,func,n=NTHREADS):
    df_split = np.array_split(df, n)
    pool = Pool(n)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_bin(value,bins):
    for i,b in enumerate(bins):
        if value<b: return i-1
    return bins.size-2
def get_weight(row):
    bX = get_bin(row[weight_bin_labels[0]],xedges)
    bY = get_bin(row[weight_bin_labels[1]],yedges)
    for truth in out_truth:
        if row[truth]: 
            return weight_distributions[truth][bX][bY]
    return 1
def weighting(df):
    # create abseta
    df['muon_innerTrack_abseta'] = df['muon_innerTrack_eta'].abs()
    df['weight'] = df.apply(lambda row: get_weight(row), axis=1)
    return df

def normalize(df):
    for key in branches:
        # normalize sumpt
        if key=='muon_isolationR03_sumPt':
            df[key] = df[key]/df['muon_pt']
        # shift ieta, iphi to be centered at 0
        if key=='muon_calEnergy_crossedHadRecHits_ieta':
            df[key] = df.apply(lambda row: [x-row['muon_calEnergy_hcal_ieta'] for x in row[key]], axis=1)
        if key=='muon_calEnergy_crossedHadRecHits_iphi':
            df[key] = df.apply(lambda row: [x-row['muon_calEnergy_hcal_iphi'] for x in row[key]], axis=1)
        # normalize hcal digi energy to total hcal energy
        if key=='muon_calEnergy_crossedHadRecHits_energy':
            df[key] = df.apply(lambda row: [x/x.sum() if x.sum() else 0 for x in row[key]], axis=1)
        # prevent very different default values
        if key=='muon_calEnergy_crossedHadRecHits_time':
            df[key] = df[key].apply(lambda x: [-20 if ix<-20 else ix for ix in x])
        if key=='muon_calEnergy_crossedHadRecHits_chi2':
            df[key] = df[key].apply(lambda x: [-5 if ix<-5 else ix for ix in x])

    for key in branches:
        if df[key].dtype==object: # needed in pandas 0.25.3
            df[key] = df[key].apply(lambda row: np.array(row))
        if key in linear_branches:
            df[key] = (df[key].clip(*linear_branches[key])-linear_branches[key][0])/(linear_branches[key][1]-linear_branches[key][0])
        else:
            df[key] = df[key]-means[key]
            df[key] = df[key]/stds[key]
    return df

def padtruncate(df):
    for b,l in branch_lengths.items():
        if b not in df.columns: continue
        df[b] = df[b].apply(lambda x: np.pad(np.array(x), (0,l), 'constant')[:l])
    return df

def convert_fname(fname,i):
    print('Processing',fname)
    for df in uproot.iterate(fname,treename,other_branches+branches,entrysteps=100000000,outputtype=pd.DataFrame):
        isval = i%10==1
        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        print('Converting',i)
        for key in other_branches+branches:
            if isinstance(df[key],awkward.ObjectArray): df[key] = awkward.fromiter(df[key])

        # flatten the dataframe
        print('Flattening',i)
        if vectorInput: df = pd.DataFrame({k: list(itertools.chain.from_iterable(df[k])) for k in df.columns})

        # selections
        print('Reducing',i)
        #if not isval: df = df[df['muon_gen_deltaR']<0.1]
        #df['muon'] = df['muon_gen_matches_muon']
        #df['pion'] = df['muon_gen_matches_pion']
        if not isval: df = df[(df['muon_gen_sim_pdgId'].abs()==13) | (df['muon_gen_sim_pdgId'].abs()==211)].copy()
        df['muon'] = df['muon_gen_sim_pdgId'].abs()==13
        df['pion'] = df['muon_gen_sim_pdgId'].abs()==211

        # throw out 80% pions
        if not isval: df = df.drop(df[df['pion']==1].sample(frac=0.8).index)

        # throw out barrel
        if not isval: df = df[df['muon_innerTrack_eta'].abs()>1.566] # HE start full depth
    
        # calculate weight
        print('Weighting',i)
        if parallel:
            df = parallel_apply(df,weighting)
        else:
            df = weighting(df)


        # normalize
        print('Normalizing',i)
        if parallel:
            df = parallel_apply(df,normalize)
        else:
            df = normalize(df)

        # zero pad and truncate
        print('Padding and truncating',i)
        if parallel:
            df = parallel_apply(df,padtruncate)
        else:
            df = padtruncate(df)
    
    
        # convert to numpy
        if isval:
            print('Preparing',i)
            W = df['weight']
            # note: this stacks the list of arrays that happens if a branch is an array
            X = [np.stack([np.stack(a) for a in df[groupb].to_numpy()]) for groupb in branch_groupings]
            Y = df[out_truth]
    
            print('Saving',i)
            name = 'output_validation'
            np.save('{}/{}_{}.w.npy'.format(outDir,name,i),W)
            for j,x in enumerate(X):
                np.save('{}/{}_{}.x{}.npy'.format(outDir,name,i,j),x)
            np.save('{}/{}_{}.y.npy'.format(outDir,name,i),Y)
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)
        else:
            print('Preparing',i)
            W = {truth: df[df[truth]==1]['weight'] for truth in out_truth}
            # note: this stacks the list of arrays that happens if a branch is an array
            X = {truth: [np.stack([np.stack(a) for a in df[df[truth]==1][groupb].to_numpy()]) for groupb in branch_groupings] for truth in out_truth}
            Y = {truth: df[df[truth]==1][out_truth] for truth in out_truth}
    
            print('Saving',i)
            name = 'output'
            for truth in out_truth:
                np.save('{}/{}_{}_{}.w.npy'.format(outDir,name,truth,i),W[truth])
                for j,x in enumerate(X[truth]):
                    np.save('{}/{}_{}_{}.x{}.npy'.format(outDir,name,truth,i,j),x)
                np.save('{}/{}_{}_{}.y.npy'.format(outDir,name,truth,i),Y[truth])
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)

for i,fname in enumerate(fnames):
    convert_fname(fname,i)
#with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
#    inds = range(len(fnames))
#    results = executor.map(convert_fname,fnames,inds)
#    for r in results: # just so it throws an error
#        if r: print(r)
#    executor.shutdown()
