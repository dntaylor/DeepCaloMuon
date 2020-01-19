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
from tqdm.auto import tqdm

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
doElectron = True

inDir = args.rootDir
outDir = args.convertDir
if os.path.exists(outDir):
    logging.info(f'{outDir} already exists')
    sys.exit(0)

os.makedirs(outDir, exist_ok=True)

# get all root files in the directory
fnames = []
for r, d, f in os.walk(inDir):
    for fname in f:
        if fname.endswith('.root'):
            fnames += [os.path.join(r,fname)]

fnames = [fname for fname in fnames if 'JpsiToMuMu_JpsiPt8_TuneCP5_14TeV-pythia8' not in fname]
logging.info('Will convert {} files'.format(len(fnames)))

treename = 'deepMuonTree/DeepMuonTree'
# must create these branches, they are what is output
if doElectron:
    out_truth = ['pion','muon','electron']
else:
    out_truth = ['pion','muon']
# weight bins
weight_bins = [
    # muon_innerTrack_p
    np.array(
        [1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,15,20,30,40,50,100,1000,7000],
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
other_branches = [
    'muon_gen_sim_pdgId',
    'muon_gen_sim_tpAssoQuality',
    'muon_pt',
    'muon_energy',
]
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
with open('{}/branches.txt'.format(outDir),'w') as f:
    for b in branches:
        f.write(b+'\n')
# energy branches to normalize against muon energy
energy_branches = [
    'muon_calEnergy_em',
    'muon_calEnergy_emMax',
    'muon_calEnergy_emS25',
    'muon_calEnergy_emS9',
    'muon_calEnergy_had',
    'muon_calEnergy_hadMax',
    'muon_calEnergy_hadS9',
    'muon_calEnergy_ho',
    'muon_calEnergy_hoS9',
    'muon_calEnergy_tower',
    'muon_calEnergy_towerS9',
]

branch_groupings = [
    [b for b in branches if 'muon_calEnergy_crossedHadRecHits' not in b],
    [b for b in branches if 'muon_calEnergy_crossedHadRecHits' in b],
]
branch_lengths = {b: 15 for b in branches if 'muon_calEnergy_crossedHadRecHits' in b}
linear_branches = {
    #'muon_innerTrack_p': [1.5,7000.],
    'muon_innerTrack_eta': [-3.0,3.0],
    'muon_innerTrack_phi': [-np.pi,np.pi],
    'muon_innerTrack_hitPattern_trackerLayersWithMeasurement': [0.,20.],
    'muon_innerTrack_hitPattern_pixelLayersWithMeasurement':[0.,5.],
    'muon_isolationR03_nTracks':[0.,15.],
    'muon_calEnergy_hcal_ieta':[-30.,30.],
    'muon_calEnergy_hcal_iphi':[0.,72.],
    #'muon_calEnergy_crossedHadRecHits_depth':[0,8],
}
loglinear_branches = {
    'muon_innerTrack_p': [1.5,7000.],
}

# get weights
distributions = {}
with tqdm(unit='files', total=len(fnames), desc='Calculating weights') as pbar:
    for fname in fnames:
        for arrays in uproot.iterate(fname,treename,other_branches+weight_branches,namedecode="utf-8",entrysteps=1000000000):
            arrays['muon_innerTrack_abseta'] = abs(arrays['muon_innerTrack_eta'])
            if doElectron:
                keep =  ((abs(arrays['muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays['muon_gen_sim_pdgId'])==11)
                    | (abs(arrays['muon_gen_sim_pdgId'])==211))
            else:
                keep =  ((abs(arrays['muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays['muon_gen_sim_pdgId'])==211))
            arrays['muon'] = (abs(arrays['muon_gen_sim_pdgId'])==13)
            arrays['pion'] = (abs(arrays['muon_gen_sim_pdgId'])==211)
            arrays['electron'] = (abs(arrays['muon_gen_sim_pdgId'])==11)
    
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
        pbar.update(1)

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

def transform(arrays):
    for key in branches:
        if key not in arrays: continue
        # normalize sumpt
        if key=='muon_isolationR03_sumPt':
            arrays[key] = arrays[key]/arrays['muon_pt']
        # shift ieta, iphi to be centered at 0
        if key=='muon_calEnergy_crossedHadRecHits_ieta':
            arrays[key] = arrays[key]-arrays['muon_calEnergy_hcal_ieta']
        if key=='muon_calEnergy_crossedHadRecHits_iphi':
            arrays[key] = arrays[key]-arrays['muon_calEnergy_hcal_iphi']
        # normalize hcal digi energy to total hcal energy
        if key=='muon_calEnergy_crossedHadRecHits_energy':
            arrays[key] = arrays[key]/arrays[key].sum()
        # normalize energies relative to the muon energy
        if key in energy_branches:
            arrays[key] = arrays[key]/arrays['muon_energy']
        # prevent very different default values
        if key=='muon_calEnergy_crossedHadRecHits_time':
            arrays[key][(arrays[key]<-20)] = -20
        if key=='muon_calEnergy_crossedHadRecHits_chi2':
            arrays[key][(arrays[key]<-5)] = -5
    return arrays

# get means
means_sum = {key:[] for key in branches}
varis_sum = {key:[] for key in branches}
with tqdm(unit='files', total=len(fnames), desc='Calculating means') as pbar:
    for fname in fnames:
        for arrays in uproot.iterate(fname,treename,branches+other_branches,namedecode="utf-8",entrysteps=100000000):
            means = {}
            varis = {}
            for key in arrays:
                # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
                if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])
            arrays = transform(arrays)
            for key in arrays:
                if key not in branches: continue
                a = arrays[key]
                while isinstance(a,awkward.JaggedArray): a = a.flatten()
                means[key] = a[~np.isnan(a)].mean()
                varis[key] = a[~np.isnan(a)].var()
                means_sum[key] += [means[key]]
                varis_sum[key] += [varis[key]]
        pbar.update(1)
means = {key: np.array(means_sum[key]).mean() for key in branches}
varis = {key: np.array(varis_sum[key]).mean() for key in branches}
stds  = {key: np.sqrt(np.array(varis_sum[key]).mean()) for key in branches}

for key in sorted(means):
    logging.info(f'{key} {means[key]} +/- {stds[key]}')

result = {
    'means':{key:float(item) for key,item in means.items()},
    'stds':{key:float(item) for key,item in stds.items()},
    'linear': {key:item for key,item in linear_branches.items()},
    'loglinear': {key:item for key,item in loglinear_branches.items()},
}
with open('{}/means.json'.format(outDir),'w') as f:
    json.dump(result,f)


def weighting(arrays):
    # create abseta
    arrays['muon_innerTrack_abseta'] = abs(arrays['muon_innerTrack_eta'])
    arrays['weight'] = np.zeros(arrays['muon_innerTrack_abseta'].shape)
    for truth in out_truth:
        for xi in range(len(xedges)-1):
            for yi in range(len(yedges)-1):
                mask = ((arrays[truth]) 
                    & (arrays[weight_bin_labels[0]]>xedges[xi]) 
                    & (arrays[weight_bin_labels[0]]<xedges[xi+1])
                    & (arrays[weight_bin_labels[1]]>yedges[yi]) 
                    & (arrays[weight_bin_labels[1]]<yedges[yi+1]))
                arrays['weight'][mask] = weight_distributions[truth][xi][yi]
    return arrays

def normalize(arrays):
    for key in branches:
        if key in linear_branches:
            arrays[key] = (arrays[key].clip(*linear_branches[key])-linear_branches[key][0])/(linear_branches[key][1]-linear_branches[key][0])
        elif key in loglinear_branches:
            arrays[key] = (np.log(arrays[key].clip(*loglinear_branches[key]))-np.log(loglinear_branches[key][0]))/(np.log(loglinear_branches[key][1])-np.log(loglinear_branches[key][0]))
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
    for arrays in uproot.iterate(fname,treename,other_branches+branches,namedecode="utf-8",entrysteps=100000000):
        isval = i%10==1
        # convert vector<vector<T>> (ObjectArray by default) into nested JaggedArray
        for key in other_branches+branches:
            if isinstance(arrays[key],awkward.ObjectArray): arrays[key] = awkward.fromiter(arrays[key])

        # selections
        if not isval: 
            if doElectron:
                keep = ((abs(arrays['muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays['muon_gen_sim_pdgId'])==11)
                    | (abs(arrays['muon_gen_sim_pdgId'])==211))
            else:
                keep = ((abs(arrays['muon_gen_sim_pdgId'])==13) 
                    | (abs(arrays['muon_gen_sim_pdgId'])==211))
            keep = (keep & (arrays['muon_gen_sim_tpAssoQuality']>0.5)) # try only keeping high quality matches
        else:
            keep = (np.zeros(arrays['muon_gen_sim_pdgId'].shape)==0)
        for key in arrays:
            arrays[key] = arrays[key][keep]
        arrays['muon'] = (abs(arrays['muon_gen_sim_pdgId'])==13)
        arrays['pion'] = (abs(arrays['muon_gen_sim_pdgId'])==211)
        arrays['electron'] = (abs(arrays['muon_gen_sim_pdgId'])==11)

        # calculate weight
        arrays = weighting(arrays)

        # transform
        arrays = transform(arrays)

        # zero pad and truncate
        arrays = padtruncate(arrays)
    
        # normalize
        arrays = normalize(arrays)

        # convert to numpy
        def get_output(arrays,out_truth,selection=None):
            if selection is None:
                selection = np.ones_like(arrays['weight'], dtype=bool)
            W = arrays['weight'][selection]
            # note: this stacks the list of arrays that happens if a branch is an array
            X = [np.swapaxes(np.stack([arrays[ab][selection] for ab in groupb]),0,1) for groupb in branch_groupings]
            Y = np.swapaxes(np.stack([arrays[ot][selection] for ot in out_truth]),0,1)
            return W, X, Y

        if isval:
            W, X, Y = get_output(arrays,out_truth)

            name = 'output_validation'
            np.save('{}/{}_{}.w.npy'.format(outDir,name,i),W)
            for j,x in enumerate(X):
                np.save('{}/{}_{}.x{}.npy'.format(outDir,name,i,j),x)
            np.save('{}/{}_{}.y.npy'.format(outDir,name,i),Y)
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)
        else:
            W, X, Y = {}, {}, {}
            for truth in out_truth:
                W[truth], X[truth], Y[truth] = get_output(arrays,out_truth,arrays[truth])

            name = 'output'
            for truth in out_truth:
                np.save('{}/{}_{}_{}.w.npy'.format(outDir,name,truth,i),W[truth])
                for j,x in enumerate(X[truth]):
                    np.save('{}/{}_{}_{}.x{}.npy'.format(outDir,name,truth,i,j),x)
                np.save('{}/{}_{}_{}.y.npy'.format(outDir,name,truth,i),Y[truth])
            with open('{}/{}_{}.input'.format(outDir,name,i),'w') as f:
                f.write(fname)
    

def _futures_handler(futures_set, status=True, unit='items', desc='Processing'):
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                while finished:
                    res = finished.pop().result()
                    pbar.update(1)
                time.sleep(0.5)
    except KeyboardInterrupt:
        for job in futures_set:
            job.cancel()
        if status:
            logging.info("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.", file=sys.stderr)
            logging.info("Running jobs:", sum(1 for j in futures_set if j.running()), file=sys.stderr)
    except Exception:
        for job in futures_set:
            job.cancel()
        raise

#for i,fname in enumerate(fnames):
#    convert_fname(fname,i)

# TODO: probably not optimal
# currently break apart by file
# would like to break apart by chunks
with concurrent.futures.ThreadPoolExecutor(NTHREADS) as executor:
    futures = set(executor.submit(convert_fname, fname, i) for i, fname in enumerate(fnames))
    _futures_handler(futures, status=True, unit='items', desc='Processing')
