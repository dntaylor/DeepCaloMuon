import os
import sys
import glob
import json
import argparse
import numpy as np
import awkward
import uproot
import logging

import ROOT
ROOT.gROOT.SetBatch()

import tdrstyle
tdrstyle.setTDRStyle()

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

inDir = args.rootDir
outDir = args.convertDir
plotDir = outDir+'/features'
doElectron = False
if os.path.exists(plotDir):
    print(plotDir, 'already exists')
    sys.exit(0)


python_mkdir(plotDir)

fnames = glob.glob('{}/*.root'.format(inDir))
treename = 'muonTree/MuonTree'

with open('{}/means.json'.format(outDir)) as f:
    means_result = json.load(f)

with open('{}/branches.txt'.format(outDir)) as f:
    branches = [line.strip() for line in f.readlines()]

means = means_result['means']
stds = means_result['stds']
linear = means_result['linear']
loglinear = means_result['loglinear']

gen_branches = [
    'muon_gen_sim_pdgId',
]

truths = ['muon','pion','electron']
colors = {
    'muon': ROOT.kRed,
    'pion': ROOT.kBlack,
    'electron': ROOT.kBlue,
}

def plot_input(savename,arrays,**kwargs):
    binning = kwargs.pop('binning',[])

    if not binning:
        full = np.concatenate(list(arrays.values()))
        xmin = full.min()
        xmax = full.max()
        nbins = 100
        binning = [100,xmin,xmax]


    hists = {}

    canvas = ROOT.TCanvas('c','c',800,800)
    canvas.SetTopMargin(0.12)

    legend = ROOT.TLegend(0.2,0.88,0.8,0.98)
    legend.SetNColumns(3)
    legend.SetTextFont(42)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)

    for t,truth in enumerate(truths):
        array = arrays[truth]
        hists[truth] = ROOT.TH1F('{}_{}'.format(savename.replace('/','_'),truth),'',*binning)
        nphist, edges = np.histogram(array,binning[0],binning[1:])
        for i in range(binning[0]):
            hists[truth].SetBinContent(i+1,nphist[i])
        hists[truth].Scale(1./hists[truth].Integral())
        hists[truth].SetLineColor(colors[truth])
        hists[truth].SetLineWidth(2)
        if t==0:
            hists[truth].Draw('hist')
            hists[truth].GetXaxis().SetTitle(savename.split('/')[-1])
            hists[truth].GetYaxis().SetTitle('Unit normalized')
        else:
            hists[truth].Draw('hist same')
        legend.AddEntry(hists[truth],truth,'l')

    legend.Draw()

    canvas.Print(savename+'.png')


# first plot the inputs
for arrays in uproot.iterate(fnames,treename,branches+gen_branches,namedecode="utf-8",entrysteps=1000000):
    print(arrays.keys())

    muon = (abs(arrays['muon_gen_sim_pdgId'])==13)
    pion = (abs(arrays['muon_gen_sim_pdgId'])==211)
    electron = (abs(arrays['muon_gen_sim_pdgId'])==11)

    for key in arrays:
        if key in gen_branches: continue
        toPlot = {}
        toPlot['muon'] = arrays[key][muon]
        toPlot['pion'] = arrays[key][pion]
        toPlot['electron'] = arrays[key][electron]
        for truth in toPlot:
            if isinstance(toPlot[truth],awkward.JaggedArray):
                toPlot[truth] = toPlot[truth].flatten()
        savename = '{}/input_{}'.format(plotDir,key)
        plot_input(savename,toPlot)

    break

# now plot the outputs
nx = 2
if doElectron:
    truths = ['pion','muon','electron']
else:
    truths = ['pion','muon']
fnames = {truth: sorted([f for f in glob.glob('{}/output_{}*.x0.npy'.format(outDir,truth)) if 'validation' not in f]) for truth in truths}
X = {}
Y = {}
W = {}
for truth in truths:
    Xs = [[np.load(fname.replace('.x0.npy','.x{}.npy'.format(i))) for i in range(nx)] for fname in fnames[truth]]
    Ys = [np.load(fname.replace('.x0.npy','.y.npy')) for fname in fnames[truth]]
    Ws = [np.load(fname.replace('.x0.npy','.w.npy')) for fname in fnames[truth]]

    Ws = [np.reshape(w,(w.shape[0],1)) for w in Ws]

    X[truth] = [np.vstack([Xs[j][i] for j in range(len(Xs))]) for i in range(nx)]
    Y[truth] = np.vstack(Ys)
    W[truth] = np.vstack(Ws)

def plot_output(savename,arrays,**kwargs):
    binning = kwargs.pop('binning',[])
    weights = kwargs.pop('weights',{})

    if not binning:
        full = np.concatenate(list(arrays.values()))
        xmin = full.min()
        xmax = full.max()
        nbins = 100
        binning = [100,xmin,xmax]


    hists = {}

    canvas = ROOT.TCanvas('c','c',800,800)
    canvas.SetTopMargin(0.12)

    legend = ROOT.TLegend(0.2,0.88,0.8,0.98)
    legend.SetNColumns(3)
    legend.SetTextFont(42)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)

    for t,truth in enumerate(truths):
        array = arrays[truth]
        hists[truth] = ROOT.TH1F('{}_{}'.format(savename.replace('/','_'),truth),'',*binning)
        if weights:
            nphist, edges = np.histogram(array,binning[0],binning[1:],weights=np.squeeze(weights[truth]))
        else:
            nphist, edges = np.histogram(array,binning[0],binning[1:])
        for i in range(binning[0]):
            hists[truth].SetBinContent(i+1,nphist[i])
        hists[truth].Scale(1./hists[truth].Integral())
        hists[truth].SetLineColor(colors[truth])
        hists[truth].SetLineWidth(2)
        if t==0:
            hists[truth].Draw('hist')
            hists[truth].GetXaxis().SetTitle(savename.split('/')[-1])
            hists[truth].GetYaxis().SetTitle('Unit normalized')
        else:
            hists[truth].Draw('hist same')
        legend.AddEntry(hists[truth],truth,'l')

    legend.Draw()

    canvas.Print(savename+'.png')


for i,branch in enumerate(branches):
    toPlot = {}
    weight = {}
    for t,truth in enumerate(truths):
        shapes = [X[truth][j].shape for j in range(nx)]
        toPlot[truth] = X[truth][0][:,i] if i<shapes[0][1] else X[truth][1][:,i-shapes[0][1],:]
        weight[truth] = W[truth]
        if i>=shapes[0][1]:
            newweight = np.ones_like(toPlot[truth]) * weight[truth]
            weight[truth] = newweight.flatten()
            toPlot[truth] = toPlot[truth].flatten()

    savename = '{}/output_{}'.format(plotDir,branch)
    plot_output(savename,toPlot)
    savename = '{}/output_weighted_{}'.format(plotDir,branch)
    plot_output(savename,toPlot,weights=weight)














