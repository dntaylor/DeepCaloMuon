import os
import sys
import argparse
import time
import random
import itertools
import threading
import numpy as np
from multiprocessing import  Pool
import json
import awkward
import uproot
import concurrent.futures
import glob
import h5py
import random
import errno
import logging

from utilities import python_mkdir

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
parallel = True # TODO: reimplement
doElectron = False

inDir = args.rootDir
outDir = args.convertDir
if os.path.exists(outDir):
    print(outDir, 'already exists')
    sys.exit(0)

python_mkdir(outDir)

fnames = glob.glob('{}/*.root'.format(inDir))
treename = 'muonTree/MuonTree'
# must create these branches, they are what is output
if doElectron:
    out_truth = [b'pion',b'muon',b'electron']
else:
    out_truth = [b'pion',b'muon']
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
weight_bin_labels = [b'muon_innerTrack_p',b'muon_innerTrack_abseta']
weight_bin_axis_labels = [r'Track $p_{T}$', r'Track $|\eta|$']
weight_branches = [b'muon_innerTrack_p',b'muon_innerTrack_eta']
reference = b'muon'
# these are helper branches (perhaps truth info) that are not output
#other_branches = [b'muon_gen_matches_pion',b'muon_gen_matches_muon',b'muon_gen_deltaR']
other_branches = [b'muon_gen_sim_pdgId',b'muon_pt']
# these are the branches that are output
branches = [
    #b'muon_innerTrack_pt',
    b'muon_innerTrack_p',
    b'muon_innerTrack_eta',
    b'muon_innerTrack_phi',
    b'muon_innerTrack_qoverp',
    b'muon_innerTrack_qoverpError',
    b'muon_innerTrack_validFraction',
    b'muon_innerTrack_highPurity',
    b'muon_innerTrack_hitPattern_trackerLayersWithMeasurement',
    b'muon_innerTrack_hitPattern_pixelLayersWithMeasurement',
    b'muon_isolationR03_nTracks',
    b'muon_isolationR03_sumPt',
    #b'muon_caloCompatibility',
    #b'muon_calEnergy_ecal_time',
    b'muon_calEnergy_em',
    b'muon_calEnergy_emMax',
    b'muon_calEnergy_emS25',
    b'muon_calEnergy_emS9',
    b'muon_calEnergy_had',
    b'muon_calEnergy_hadMax',
    b'muon_calEnergy_hadS9',
    #b'muon_calEnergy_hcal_time',
    b'muon_calEnergy_ho',
    b'muon_calEnergy_hoS9',
    #b'muon_calEnergy_tower',
    #b'muon_calEnergy_towerS9',
    b'muon_calEnergy_hcal_ieta',
    b'muon_calEnergy_hcal_iphi',
    b'muon_calEnergy_crossedHadRecHits_ieta',
    b'muon_calEnergy_crossedHadRecHits_iphi',
    b'muon_calEnergy_crossedHadRecHits_depth',
    b'muon_calEnergy_crossedHadRecHits_energy',
    b'muon_calEnergy_crossedHadRecHits_time',
    b'muon_calEnergy_crossedHadRecHits_chi2',
]
branch_groupings = [
    [b for b in branches if b'muon_calEnergy_crossedHadRecHits' not in b],
    [b for b in branches if b'muon_calEnergy_crossedHadRecHits' in b],
]
branch_lengths = {b: 15 for b in branches if b'muon_calEnergy_crossedHadRecHits' in b}
linear_branches = {
    #b'muon_innerTrack_p': [1.5,1000],
    #b'muon_innerTrack_eta': [-3.0,3.0],
}

# get weights
print('Calculating weights')
distributions = {}
for fname in fnames:
    print(fname)
    for arrays in uproot.iterate(fname,treename,other_branches+weight_branches,entrysteps=1000000000):
        arrays[b'muon_innerTrack_abseta'] = abs(arrays[b'muon_innerTrack_eta'])
        if doElectron:
            keep =  ((abs(arrays[b'muon_gen_sim_pdgId'])==13) 
                | (abs(arrays[b'muon_gen_sim_pdgId'])==11)
                | (abs(arrays[b'muon_gen_sim_pdgId'])==211))
        else:
            keep =  ((abs(arrays[b'muon_gen_sim_pdgId'])==13) 
                | (abs(arrays[b'muon_gen_sim_pdgId'])==211))
        arrays[b'muon'] = (abs(arrays[b'muon_gen_sim_pdgId'])==13)
        arrays[b'pion'] = (abs(arrays[b'muon_gen_sim_pdgId'])==211)
        arrays[b'electron'] = (abs(arrays[b'muon_gen_sim_pdgId'])==11)

        for truth in out_truth:
            hist, xedges, yedges = np.histogram2d(
                arrays[weight_bin_labels[0]][arrays[truth]],
                arrays[weight_bin_labels[1]][arrays[truth]],
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
    plot_hist(weight_distributions[truth],'{}/weight_{}.png'.format(outDir,truth.decode('utf-8')))

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
        if key not in branches: continue
        # normalize sumpt
        if key==b'muon_isolationR03_sumPt':
            arrays[key] = arrays[key]/arrays[b'muon_pt']
        # shift ieta, iphi to be centered at 0
        if key==b'muon_calEnergy_crossedHadRecHits_ieta':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_ieta']
        if key==b'muon_calEnergy_crossedHadRecHits_iphi':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_iphi']
        # normalize hcal digi energy to total hcal energy
        if key==b'muon_calEnergy_crossedHadRecHits_energy':
            arrays[key] = arrays[key]/arrays[key].sum()
        # broken for some reason
        #means[key] = arrays[key][~np.isnan(arrays[key])].flatten().mean()
        #varis[key] = arrays[key][~np.isnan(arrays[key])].flatten().var()
        a = arrays[key]
        while isinstance(a,awkward.JaggedArray): a = a.flatten()
        # change default no time/chi2 value to closer to real values
        if key==b'muon_calEnergy_crossedHadRecHits_time':
            a[a<-20] = -20
        if key==b'muon_calEnergy_crossedHadRecHits_chi2':
            a[a<-5] = -5
        means[key] = a[~np.isnan(a)].mean()
        varis[key] = a[~np.isnan(a)].var()
        means_sum[key] += [means[key]]
        varis_sum[key] += [varis[key]]
means = {key: np.array(means_sum[key]).mean() for key in branches}
varis = {key: np.array(varis_sum[key]).mean() for key in branches}
stds  = {key: np.sqrt(np.array(varis_sum[key]).mean()) for key in branches}

for key in sorted(means):
    print(key,means[key],stds[key])

result = {
    'means':{key.decode('utf-8'):float(item) for key,item in means.items()},
    'stds':{key.decode('utf-8'):float(item) for key,item in stds.items()},
    'linear': {key.decode('utf-8'):item for key,item in linear_branches.items()},
}
with open('{}/means.json'.format(outDir),'w') as f:
    json.dump(result,f)


def weighting(arrays):
    # create abseta
    arrays[b'muon_innerTrack_abseta'] = abs(arrays[b'muon_innerTrack_eta'])
    arrays[b'weight'] = np.zeros(arrays[b'muon_innerTrack_abseta'].shape)
    for truth in out_truth:
        for xi in range(len(xedges)-1):
            for yi in range(len(yedges)-1):
                mask = ((arrays[truth]) 
                    & (arrays[weight_bin_labels[0]]>xedges[xi]) 
                    & (arrays[weight_bin_labels[0]]<xedges[xi+1])
                    & (arrays[weight_bin_labels[1]]>yedges[yi]) 
                    & (arrays[weight_bin_labels[1]]<yedges[yi+1]))
                arrays[b'weight'][mask] = weight_distributions[truth][xi][yi]
    return arrays

def normalize(arrays):
    for key in branches:
        # normalize sumpt
        if key==b'muon_isolationR03_sumPt':
            arrays[key] = arrays[key]/arrays[b'muon_pt']
        # shift ieta, iphi to be centered at 0
        if key==b'muon_calEnergy_crossedHadRecHits_ieta':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_ieta']
        if key==b'muon_calEnergy_crossedHadRecHits_iphi':
            arrays[key] = arrays[key]-arrays[b'muon_calEnergy_hcal_iphi']
        # normalize hcal digi energy to total hcal energy
        if key==b'muon_calEnergy_crossedHadRecHits_energy':
            arrays[key] = arrays[key]/arrays[key].sum()
        # prevent very different default values
        if key==b'muon_calEnergy_crossedHadRecHits_time':
            arrays[key][(arrays[key]<-20)] = -20
        if key==b'muon_calEnergy_crossedHadRecHits_chi2':
            arrays[key][(arrays[key]<-5)] = -5

    for key in branches:
        if key in linear_branches:
            arrays[key] = (arrays[key].clip(*linear_branches[key])-linear_branches[key][0])/(linear_branches[key][1]-linear_branches[key][0])
        else:
            arrays[key] = arrays[key]-means[key]
            arrays[key] = arrays[key]/stds[key]
    return arrays

def padtruncate(arrays):
    for b,l in branch_lengths.items():
        if b not in arrays: continue
        arrays[b] = arrays[b].pad(l,clip=True).fillna(0).regular()
    return arrays

def convert_fname(fname,i):
    print('Processing',fname)
    for arrays in uproot.iterate(fname,treename,other_branches+branches,entrysteps=100000000):
        isval = i%10==1
        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        print('Converting',i)
        for key in other_branches+branches:
            if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])

        # selections
        print('Reducing',i)
        if not isval: 
            if doElectron:
                keep = ((abs(arrays[b'muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays[b'muon_gen_sim_pdgId'])==11)
                    | (abs(arrays[b'muon_gen_sim_pdgId'])==211))
            else:
                keep = ((abs(arrays[b'muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays[b'muon_gen_sim_pdgId'])==211))
        else:
            keep = (np.zeros(arrays[b'muon_gen_sim_pdgId'].shape)==0)
        for key in arrays:
            arrays[key] = arrays[key][keep]
        arrays[b'muon'] = (abs(arrays[b'muon_gen_sim_pdgId'])==13)
        arrays[b'pion'] = (abs(arrays[b'muon_gen_sim_pdgId'])==211)
        arrays[b'electron'] = (abs(arrays[b'muon_gen_sim_pdgId'])==11)
        for t in out_truth:
            print(t,arrays[t].sum())

        # calculate weight
        print('Weighting',i)
        arrays = weighting(arrays)


        # normalize
        print('Normalizing',i)
        arrays = normalize(arrays)

        # zero pad and truncate
        print('Padding and truncating',i)
        arrays = padtruncate(arrays)
    
    
        # convert to numpy
        if isval:
            print('Preparing',i)
            W = arrays[b'weight']
            # note: this stacks the list of arrays that happens if a branch is an array
            X = [np.swapaxes(np.stack([arrays[ab] for ab in groupb]),0,1) for groupb in branch_groupings]
            Y = np.swapaxes(np.stack([arrays[ot] for ot in out_truth]),0,1)
    
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
            W = {truth: arrays[b'weight'][arrays[truth]] for truth in out_truth}
            X = {truth: [np.swapaxes(np.stack([arrays[ab][arrays[truth]] for ab in groupb]),0,1) for groupb in branch_groupings] for truth in out_truth}
            Y = {truth: np.swapaxes(np.stack([arrays[ot][arrays[truth]] for ot in out_truth]),0,1) for truth in out_truth}

            print('Saving',i)
            name = 'output'
            for truth in out_truth:
                np.save('{}/{}_{}_{}.w.npy'.format(outDir,name,truth.decode('utf-8'),i),W[truth])
                for j,x in enumerate(X[truth]):
                    np.save('{}/{}_{}_{}.x{}.npy'.format(outDir,name,truth.decode('utf-8'),i,j),x)
                np.save('{}/{}_{}_{}.y.npy'.format(outDir,name,truth.decode('utf-8'),i),Y[truth])
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
