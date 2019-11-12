from array import array
import json
import glob

import ROOT
ROOT.gROOT.SetBatch(True)

import tdrstyle
tdrstyle.setTDRStyle()

##fname = 'genMuons_jpsi_mergedMuons.root'
##fname = 'genMuons_jpsi.root'
#fname = '/storage/local/data1/gpuscratch/dntaylor/caloMuons/muonTree_2.root'
#tfile = ROOT.TFile.Open(fname)
##tree = tfile.Get('AnalysisTree')
#tree = tfile.Get('muonTree/MuonTree')

tree = ROOT.TChain('muonTree/MuonTree')
for fname in glob.glob('/storage/local/data1/gpuscratch/dntaylor/caloMuons/*.root'):
    tree.Add(fname)

discvar = 'muon_caloCompatibility'
nd = 100
discrange = [x*.01 for x in range(0,101,1)]
discbinning = [nd,0,1]
pvar = 'muon_p'
etavar = 'abs(muon_eta)'
pbins = [2,5,10,50,1000]
npb = len(pbins)-1
etabins = [0,1.2,1.566,2.322,2.5]
neb = len(etabins)-1
plabels = []
etalabels = []
for pb in range(npb):
    plabels += ['{} < p < {} GeV'.format(pbins[pb],pbins[pb+1])]
for eb in range(neb):
    etalabels += ['{} < |#eta| < {}'.format(etabins[eb],etabins[eb+1])]
markers = [21,22,23,33,34]
idnames = ['isPFMuon','isTrackerMuon']

counts = {}
for pb in range(npb):
    counts[pb] = {}
    for eb in range(neb):
        print('Getting',pb,eb)
        counts[pb][eb] = {}
        bincut = '{}>={} && {}<{} && {}>={} && {}<{}'.format(
            pvar, pbins[pb], pvar, pbins[pb+1],
            etavar, etabins[eb], etavar, etabins[eb+1],
        )
        #tprcut = '{} && muon_gen_matches_muon>0.5 && muon_gen_deltaR<0.1'.format(bincut)
        #fprcut = '{} && muon_gen_matches_pion>0.5 && muon_gen_deltaR<0.1'.format(bincut)
        tprcut = '{} && abs(muon_gen_sim_pdgId)==13'.format(bincut)
        fprcut = '{} && abs(muon_gen_sim_pdgId)==211'.format(bincut)
        muonname = 'h_muon_{}_{}'.format(pb,eb)
        pionname = 'h_pion_{}_{}'.format(pb,eb)
        tree.Draw('{}>>{}({})'.format(discvar,muonname,', '.join([str(x) for x in discbinning])),'{}*({})'.format(1,tprcut),'goff')
        if ROOT.gDirectory.Get(muonname):
            counts[pb][eb]['tpr'] = ROOT.gDirectory.Get(muonname)
        tree.Draw('{}>>{}({})'.format(discvar,pionname,', '.join([str(x) for x in discbinning])),'{}*({})'.format(1,fprcut),'goff')
        if ROOT.gDirectory.Get(pionname):
            counts[pb][eb]['fpr'] = ROOT.gDirectory.Get(pionname)

        # muon IDs
        for idname in idnames:
            muonname = 'h_muon_{}_{}_{}'.format(idname,pb,eb)
            pionname = 'h_pion_{}_{}_{}'.format(idname,pb,eb)
            tree.Draw('{}>>{}({})'.format('muon_'+idname,muonname,'2,-0.5,1.5'),'{}*({})'.format(1,tprcut),'goff')
            if ROOT.gDirectory.Get(muonname):
                counts[pb][eb]['tpr'+idname] = ROOT.gDirectory.Get(muonname)
            tree.Draw('{}>>{}({})'.format('muon_'+idname,pionname,'2,-0.5,1.5'),'{}*({})'.format(1,fprcut),'goff')
            if ROOT.gDirectory.Get(pionname):
                counts[pb][eb]['fpr'+idname] = ROOT.gDirectory.Get(pionname)
            print(idname,[float(counts[pb][eb][x+idname].GetBinContent(2))/counts[pb][eb][x+idname].Integral() for x in ['tpr','fpr']])



def plot_hist(savename,xcounts,ycounts,labels=[],additional='',working=[]):
    allxvals = []
    allyvals = []
    canvas = ROOT.TCanvas('c','c',800,600)
    canvas.SetLogy()
    graphs = {}
    wps = {}
    mg = ROOT.TMultiGraph()
    wpmg = ROOT.TMultiGraph()
    colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kGreen+2, ROOT.kRed, ROOT.kMagenta, ROOT.kCyan]
    legend = ROOT.TLegend(0.2,0.80,0.5,0.94,'','NDC')
    legend.SetTextFont(42)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legendWorking = ROOT.TLegend(0.6,0.25,0.9,0.4,'','NDC')
    legendWorking.SetTextFont(42)
    legendWorking.SetBorderSize(0)
    legendWorking.SetFillColor(0)
    with open(savename+'.json','w') as f:
        json.dump(working,f)
    print(savename)
    for j,(xcount,ycount) in enumerate(zip(xcounts,ycounts)):
        xvals = [float(sum(xcount[i:]))/sum(xcount) for i,x in enumerate(xcount)]
        yvals = [float(sum(ycount[i:]))/sum(ycount) for i,y in enumerate(ycount)]

        graphs[j] = ROOT.TGraph(len(xvals),array('d',xvals),array('d',yvals))
        graphs[j].SetLineColor(colors[j])
        graphs[j].SetLineWidth(2)
        mg.Add(graphs[j])
        if labels: legend.AddEntry(graphs[j],labels[j],'l')
        print(j,labels[j] if labels else '')

        # working point 0.6
        for i,d in enumerate(discrange):
            if d>=0.6:
                xval = xvals[i]
                yval = yvals[i]
                break
        wps[j] = ROOT.TGraph(1,array('d',[xval]),array('d',[yval]))
        wps[j].SetMarkerColor(colors[j])
        wpmg.Add(wps[j])
        if j==0: legendWorking.AddEntry(wps[j],'isCaloMuon','p')

        for i, idname in enumerate(idnames):
            wps['{}{}'.format(idname,j)] = ROOT.TGraph(1,array('d',[working[j][i][0]]),array('d',[working[j][i][1]]))
            wps['{}{}'.format(idname,j)].SetMarkerColor(colors[j])
            wps['{}{}'.format(idname,j)].SetMarkerStyle(markers[i])
            wpmg.Add(wps['{}{}'.format(idname,j)])
            if j==0: legendWorking.AddEntry(wps['{}{}'.format(idname,j)],idname,'p')
            print(idname,working[j][i])

        allxvals += [xvals]
        allyvals += [yvals]

    mg.Draw('AL')
    mg.GetXaxis().SetTitle('True Muon')
    mg.GetYaxis().SetTitle('Fake Pion')
    mg.GetHistogram().SetMinimum(1e-4)
    wpmg.Draw('P')
    if labels: legend.Draw()
    if working: legendWorking.Draw()
    if additional:
        text = ROOT.TPaveText(0.7,0.15,0.95,0.25,'NB NDC')
        text.SetTextFont(42)
        text.SetBorderSize(0)
        text.SetFillColor(0)
        text.SetTextAlign(11)
        text.SetTextSize(0.05)
        text.AddText(additional)
        text.Draw()
    canvas.Print('{}.png'.format(savename))

    return allxvals, allyvals

counts_all     = {'tpr':[0]*nd,'fpr':[0]*nd}
counts_pbins   = {x:{'tpr':[0]*nd,'fpr':[0]*nd} for x in range(npb)}
counts_etabins = {x:{'tpr':[0]*nd,'fpr':[0]*nd} for x in range(neb)}
counts_p_eta_bins = {pb:{eb:{'tpr':[0]*nd,'fpr':[0]*nd} for eb in range(neb)} for pb in range(npb)}
for idname in idnames:
    counts_all['tpr'+idname] = 0
    counts_all['fpr'+idname] = 0
    counts_all['tr'+idname] = 0
    counts_all['fr'+idname] = 0
    for pb in range(npb):
        counts_pbins[pb]['tpr'+idname] = 0
        counts_pbins[pb]['fpr'+idname] = 0
        counts_pbins[pb]['tr'+idname] = 0
        counts_pbins[pb]['fr'+idname] = 0
    for eb in range(neb):
        counts_etabins[eb]['tpr'+idname] = 0
        counts_etabins[eb]['fpr'+idname] = 0
        counts_etabins[eb]['tr'+idname] = 0
        counts_etabins[eb]['fr'+idname] = 0
    for pb in range(npb):
        for eb in range(neb):
            counts_p_eta_bins[pb][eb]['tpr'+idname] = 0
            counts_p_eta_bins[pb][eb]['fpr'+idname] = 0
            counts_p_eta_bins[pb][eb]['tr'+idname] = 0
            counts_p_eta_bins[pb][eb]['fr'+idname] = 0
for pb in range(npb):
    for eb in range(neb):
        for i in range(counts[pb][eb]['tpr'].GetNbinsX()):
            ntpr = counts[pb][eb]['tpr'].GetBinContent(i+1)
            nfpr = counts[pb][eb]['fpr'].GetBinContent(i+1)
            counts_all['tpr'][i] += ntpr
            counts_all['fpr'][i] += nfpr
            counts_pbins[pb]['tpr'][i] += ntpr
            counts_pbins[pb]['fpr'][i] += nfpr
            counts_etabins[eb]['tpr'][i] += ntpr
            counts_etabins[eb]['fpr'][i] += nfpr
            counts_p_eta_bins[pb][eb]['tpr'][i] += ntpr
            counts_p_eta_bins[pb][eb]['fpr'][i] += nfpr

        # working points
        for idname in idnames:
            ntpr = counts[pb][eb]['tpr'+idname].GetBinContent(2)
            nfpr = counts[pb][eb]['fpr'+idname].GetBinContent(2)
            ntfr = counts[pb][eb]['tpr'+idname].GetBinContent(1)
            nffr = counts[pb][eb]['fpr'+idname].GetBinContent(1)
            counts_all['tpr'+idname] += ntpr
            counts_all['fpr'+idname] += nfpr
            counts_all['tr'+idname] += ntpr+ntfr
            counts_all['fr'+idname] += nfpr+nffr
            counts_pbins[pb]['tpr'+idname] += ntpr
            counts_pbins[pb]['fpr'+idname] += nfpr
            counts_pbins[pb]['tr'+idname] += ntpr+ntfr
            counts_pbins[pb]['fr'+idname] += nfpr+nffr
            counts_etabins[eb]['tpr'+idname] += ntpr
            counts_etabins[eb]['fpr'+idname] += nfpr
            counts_etabins[eb]['tr'+idname] += ntpr+ntfr
            counts_etabins[eb]['fr'+idname] += nfpr+nffr
            counts_p_eta_bins[pb][eb]['tpr'+idname] += ntpr
            counts_p_eta_bins[pb][eb]['fpr'+idname] += nfpr
            counts_p_eta_bins[pb][eb]['tr'+idname] += ntpr+ntfr
            counts_p_eta_bins[pb][eb]['fr'+idname] += nfpr+nffr


xvals = {}
yvals = {}
for pb in range(npb):
    xv, yv = plot_hist('roc_caloCompatibility_pBin{}'.format(pb),
                       [counts_p_eta_bins[pb][eb]['tpr'] for eb in range(neb)],
                       [counts_p_eta_bins[pb][eb]['fpr'] for eb in range(neb)],
                       labels=etalabels,
                       additional=plabels[pb],
                       working=[[
                            [float(counts_p_eta_bins[pb][eb]['tpr'+idname])/counts_p_eta_bins[pb][eb]['tr'+idname],
                             float(counts_p_eta_bins[pb][eb]['fpr'+idname])/counts_p_eta_bins[pb][eb]['fr'+idname]]
                            for idname in idnames] for eb in range(neb)])
    xvals['pbin{}'.format(pb)] = xv
    yvals['pbin{}'.format(pb)] = yv
for eb in range(neb):
    xv, yv = plot_hist('roc_caloCompatibility_etaBin{}'.format(eb),
                       [counts_p_eta_bins[pb][eb]['tpr'] for pb in range(npb)],
                       [counts_p_eta_bins[pb][eb]['fpr'] for pb in range(npb)],
                       labels=plabels,
                       additional=etalabels[eb],
                       working=[[
                            [float(counts_p_eta_bins[pb][eb]['tpr'+idname])/counts_p_eta_bins[pb][eb]['tr'+idname],
                             float(counts_p_eta_bins[pb][eb]['fpr'+idname])/counts_p_eta_bins[pb][eb]['fr'+idname]] 
                            for idname in idnames] for pb in range(npb)])
    xvals['etabin{}'.format(eb)] = xv
    yvals['etabin{}'.format(eb)] = yv

xv, yv = plot_hist('roc_caloCompatibility_pBins',
                   [counts_pbins[pb]['tpr'] for pb in range(npb)],
                   [counts_pbins[pb]['fpr'] for pb in range(npb)],
                   labels=plabels,
                   working=[[
                        [float(counts_pbins[pb]['tpr'+idname])/counts_pbins[pb]['tr'+idname],
                         float(counts_pbins[pb]['fpr'+idname])/counts_pbins[pb]['fr'+idname]] 
                        for idname in idnames] for pb in range(npb)])
xvals['pbins'] = xv
yvals['pbins'] = yv
xv, yv = plot_hist('roc_caloCompatibility_etaBins',
                   [counts_etabins[eb]['tpr'] for eb in range(neb)],
                   [counts_etabins[eb]['fpr'] for eb in range(neb)],
                   labels=etalabels,
                   working=[[
                        [float(counts_etabins[eb]['tpr'+idname])/counts_etabins[eb]['tr'+idname],
                         float(counts_etabins[eb]['fpr'+idname])/counts_etabins[eb]['fr'+idname]] 
                        for idname in idnames] for eb in range(neb)])
xvals['etabins'] = xv
yvals['etabins'] = yv


xv, yv = plot_hist('roc_caloCompatibility',[counts_all['tpr']],[counts_all['fpr']],
                   working=[[
                        [float(counts_all['tpr'+idname])/counts_all['tr'+idname],
                         float(counts_all['fpr'+idname])/counts_all['fr'+idname]] 
                        for idname in idnames]])
xvals['all'] = xv
yvals['all'] = yv
result = {'tpr': xvals, 'fpr': yvals}
with open('roc_caloCompatibility.json','w') as f:
    json.dump(result,f)


