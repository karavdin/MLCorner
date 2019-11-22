# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, kOrange, gStyle

signalmasses = ['All', '0.7 TeV', '1.5 TeV']
signal_linestyles = ['-', '--', ':', '-.']
signal_identifiers = ['TstarTstarToTgluonTgluon_All', 'TstarTstarToTgluonTgluon_M-700', 'TstarTstarToTgluonTgluon_M-1500']

colorstr = ['C0', 'C3', 'C1', 'C2', 'C4']
rootcolors = {'C3': kRed, 'C0': kBlue+1, 'C2': kGreen+1, 'C4': kMagenta, 'C1': kOrange+1}
