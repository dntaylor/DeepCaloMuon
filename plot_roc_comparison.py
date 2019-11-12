import os
import sys
import argparse
from array import array
import json

import ROOT
ROOT.gROOT.SetBatch(True)

import tdrstyle
tdrstyle.setTDRStyle()

parser = argparse.ArgumentParser(description='Plot comparison')
parser.add_argument('jsons', type=str, nargs='+',
                    help='JSON files for ROC')
parser.add_argument('--labels', type=str, nargs='+',
                    help='Labels for plot')

args = parser.parse_args()

tag = 'comparison'
jsonFiles = args.jsons
jsonLabels = args.labels
print(jsonFiles)
print(jsonLabels)

#jsonFiles = [
#    'roc_caloCompatibility.json',
#    #'/storage/local/data1/gpuscratch/dntaylor/caloMuons_hcalDigis_trackerQuality_simHit/train_noHcalDigis_v1/roc_muon.json',
#    '/storage/local/data1/gpuscratch/dntaylor/caloMuons_hcalDigis_trackerQuality_simHit_v2/train_v5/roc_muon.json',
#]
#
#jsonLabels = [
#    'caloCompatibility',
#    #'ML + Track',
#    'ML + HCAL Digis + Track',
#]


colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen+2, ROOT.kRed, ROOT.kMagenta, ROOT.kCyan]
styles = [1,4,2,6,10]
widths = [3,1,7,1,9]
markers = [21,22,23,33,34]
idnames = ['isPFMuon','isTrackerMuon']

pbins = [2,5,10,50]#,1000]
npb = len(pbins)-1
etabins = [0,1.2,1.566,2.322,2.5]
neb = len(etabins)-1
plabels = []
etalabels = []
for pb in range(npb):
    plabels += ['{} < p < {} GeV'.format(pbins[pb],pbins[pb+1])]
for eb in range(neb):
    etalabels += ['{} < |#eta| < {}'.format(etabins[eb],etabins[eb+1])]

results = {}
for fname in jsonFiles:
    with open(fname) as f:
        results[fname] = json.load(f)


def plot_hist(savename,allxvals,allyvals,labels=[],additional='',working=[]):
    canvas = ROOT.TCanvas('c','c',800,800)
    canvas.SetTopMargin(0.2)
    canvas.SetLogy()
    graphs = {}
    wps = {}
    mg = ROOT.TMultiGraph()
    wpmg = ROOT.TMultiGraph()
    legend1 = ROOT.TLegend(0.2,0.8,0.5,0.98,'','NDC')
    legend1.SetTextFont(42)
    legend1.SetBorderSize(0)
    legend1.SetFillColor(0)
    #legend1.SetNColumns(len(jsonLabels))
    legend2 = ROOT.TLegend(0.55,0.8,0.85,0.98,'','NDC')
    legend2.SetTextFont(42)
    legend2.SetBorderSize(0)
    legend2.SetFillColor(0)
    #legend2.SetNColumns(len(jsonLabels))
    legendWorking = ROOT.TLegend(0.6,0.15,0.9,0.3,'','NDC')
    legendWorking.SetTextFont(42)
    legendWorking.SetBorderSize(0)
    legendWorking.SetFillColor(0)
    for k in range(len(jsonLabels)):
        for j,(xvals,yvals) in enumerate(zip(allxvals[k],allyvals[k])):
            if j>=len(labels): continue
            graphs[(k,j,)] = ROOT.TGraph(len(xvals),array('d',xvals),array('d',yvals))
            graphs[(k,j,)].SetLineColor(colors[j])
            graphs[(k,j,)].SetLineWidth(widths[k])
            graphs[(k,j,)].SetLineStyle(styles[k])
            mg.Add(graphs[(k,j,)])
            if j==0: legend1.AddEntry(graphs[(k,j,)],jsonLabels[k],'l')
            if labels and k==0: legend2.AddEntry(graphs[(k,j,)],labels[j],'l')
            if k==0:
                for i, idname in enumerate(idnames):
                    wps['{}{}'.format(idname,j)] = ROOT.TGraph(1,array('d',[working[j][i][0]]),array('d',[working[j][i][1]]))
                    wps['{}{}'.format(idname,j)].SetMarkerColor(colors[j])
                    wps['{}{}'.format(idname,j)].SetMarkerStyle(markers[i])
                    wpmg.Add(wps['{}{}'.format(idname,j)])
                    if j==0: legendWorking.AddEntry(wps['{}{}'.format(idname,j)],idname,'p')

    mg.Draw('AL')
    mg.GetXaxis().SetTitle('True Muon')
    mg.GetYaxis().SetTitle('Fake Pion')
    mg.GetHistogram().SetMinimum(1e-5)
    mg.GetHistogram().SetMaximum(1)
    legend1.Draw()
    legend2.Draw()
    wpmg.Draw('P')
    if working: legendWorking.Draw()
    if additional:
        text = ROOT.TPaveText(0.7,0.2,0.95,0.4,'NB NDC')
        text.SetTextFont(42)
        text.SetBorderSize(0)
        text.SetFillColor(0)
        text.SetTextAlign(11)
        text.SetTextSize(0.05)
        text.AddText(additional)
        text.Draw()
    canvas.Print('{}.png'.format(savename))

for pb in range(npb):
    with open('roc_caloCompatibility_pBin{}.json'.format(pb)) as f:
        result = json.load(f)
    plot_hist('roc_{}_pBin{}'.format(tag,pb),
        [results[jsonFile]['tpr']['pbin{}'.format(pb)] for jsonFile in jsonFiles],
        [results[jsonFile]['fpr']['pbin{}'.format(pb)] for jsonFile in jsonFiles],
        labels = etalabels,
        additional = plabels[pb],
        working=result)

for eb in range(neb):
    with open('roc_caloCompatibility_etaBin{}.json'.format(eb)) as f:
        result = json.load(f)
    plot_hist('roc_{}_etaBin{}'.format(tag,eb),
        [results[jsonFile]['tpr']['etabin{}'.format(eb)] for jsonFile in jsonFiles],
        [results[jsonFile]['fpr']['etabin{}'.format(eb)] for jsonFile in jsonFiles],
        labels = plabels,
        additional = etalabels[eb],
        working=result)

with open('roc_caloCompatibility_pBins.json') as f:
    result = json.load(f)
plot_hist('roc_{}_pBins'.format(tag),
    [results[jsonFile]['tpr']['pbins'] for jsonFile in jsonFiles],
    [results[jsonFile]['fpr']['pbins'] for jsonFile in jsonFiles],
    labels = plabels,
    working=result)

with open('roc_caloCompatibility_etaBins.json') as f:
    result = json.load(f)
plot_hist('roc_{}_etaBins'.format(tag),
    [results[jsonFile]['tpr']['etabins'] for jsonFile in jsonFiles],
    [results[jsonFile]['fpr']['etabins'] for jsonFile in jsonFiles],
    labels = etalabels,
    working=result)

with open('roc_caloCompatibility.json') as f:
    result = json.load(f)
plot_hist('roc_{}'.format(tag),
    [results[jsonFile]['tpr']['all'] for jsonFile in jsonFiles],
    [results[jsonFile]['fpr']['all'] for jsonFile in jsonFiles],
    working=result)

