import os, glob, sys, itertools
import datetime as dt
from optparse import OptionParser
import optparse
import textwrap
from multiprocessing import Pool, cpu_count
import numpy as np
from ctypes import pointer, c_int, c_double
from lgmpy import Lgm_CTrans, Lgm_Vector, magcoords
from lgmpy.Lgm_Wrap import _SgpInfo, _SgpTLE, LgmSgp_ReadTlesFromStrings, Lgm_JD, LGM_TIME_SYS_UTC, Lgm_init_ctrans, LgmSgp_SGP4_Init, Lgm_Convert_Coords, Lgm_Set_Coord_Transforms, LgmSgp_SGP4, WGS84_A, TEME_TO_GSE, TEME_TO_GEO

import dateutil.parser as dup
import spacepy.time as spt
import spacepy.toolbox as tb
import FindTLEforGivenTime as fTLE
 
def findNamed(path, name):
        pp = os.path.split(path)
        if pp[-1] == '':
            return None
        if pp[-1] != name:
            path = findNamed(pp[0], name)
        return path

basepath = findNamed(os.path.dirname(os.path.abspath(__file__)), 'dream')
sys.path.insert(-1, os.path.join(basepath, 'bin'))
from MagEphem import IndentedHelpFormatterWithNL, defaults
 
defaults = {'Delta': 5,
            'outpath': os.path.join(basepath, 'Spacecraft'),
            'TLEpath': os.path.join(os.path.sep, 'n', 'space_data', 'TLE_DATABASE')
            }

WGS72_A = 6378.135

#the TLEs used here are assumed to have the (3-line) format:
#Line0 = 'NAVSTAR 49 (USA 154)'
#Line1 = '1 26605U 00071A   07067.92696543 -.00000083  00000-0  10000-3 0  5386'
#Line2 = '2 26605 056.6140 012.5765 0031970 244.1062 115.5621 02.00556447 46349'
 
def parserSetup():
    # Define a command-line option parser and add the options we need
    parser = OptionParser(  usage="%prog [options]",\
                            formatter=IndentedHelpFormatterWithNL(),\
                            version="%prog Version 1.1 (June 6, 2013)"  )
 
    parser.add_option("-s", "--Start",      dest="Start_ISO",
                            help="Start date in ISO format [required]",
                            metavar="YYYY-MM-DD")

    parser.add_option("-e", "--End",      dest="End_ISO",
                            help="End date in ISO format [required]",
                            metavar="YYYY-MM-DD")

    parser.add_option("-b", "--Bird",      dest="Bird",
                            help="Satellite Name [required for header]")

    parser.add_option("-n", "--SatNum",    dest="SatNum",
                            help="NORAD tracking number of object [required]")

    parser.add_option("-d", "--Delta",    dest="Delta",
                            help="Time increment [minutes] for ephemeris data [default 5 minutes]")

    parser.add_option("-t", "--TLEpath",      dest="TLEpath",
                            help="Path to TLEs for given Bird [default /n/space_data/TLE_DATABASE/XXXXX_XXXXX/SatNum]")

    parser.add_option("-o", "--outpath",      dest="outpath",
                            help="Path for file output (/Ephem/YEAR is auto-appended) [default .]")

    parser.add_option("-m", "--Mission",      dest="Mission",
                            help="Mission Name (e.g. ns41's mission is GPS) [optional, defaults to same as Bird]")

    return parser

def getLatLonRadfromTLE(epochs, TLEpath, options):
    ##now do Mike's setup for getting coords from TLE using SGP4
    pos_in = [0,0,0]
    s = _SgpInfo()
    TLEs = _SgpTLE()

    #loop over all times
    testlat = np.asarray(epochs).copy()
    testlat.fill(0)
    testlon = testlat.copy()
    testrad = testlat.copy()
    testtdiff = testlat.copy()
    print('Fetching TLEs & converting for range {0} to {1}'.format(epochs[0].isoformat(), epochs[-1].isoformat()))
    for idx, c_date in enumerate(epochs):
        #print('Doing {0}'.format(c_date))
        #put into JD as SGP4 needs serial time
        c = Lgm_init_ctrans(0)

        #now do Mike's setup for getting coords from TLE using SGP4
        dstr = int(c_date.strftime('%Y%j')) + c_date.hour/24.0 + c_date.minute/1440.0 + c_date.second/86400.0
        globstat = os.path.join(TLEpath, '*.txt')
        TLEfiles = glob.glob(globstat)
        if not TLEfiles:
            raise IOError('No TLE files found in {0}. Aborting...'.format(TLEpath))
        Line0, Line1, Line2 = fTLE.findTLEinfiles(TLEfiles,
                ParseMethod='UseSatelliteNumber', TargetEpoch=dstr, SatelliteNumber=options.SatNum,
                Verbose=False, PurgeDuplicates=True)
        #print("{0}\n{1}\n{2}\n\n".format(Line0,Line1,Line2))
        nTLEs = c_int(0)
        LgmSgp_ReadTlesFromStrings( Line0, Line1, Line2, pointer(nTLEs), pointer(TLEs), 1)
        LgmSgp_SGP4_Init( pointer(s), pointer(TLEs) )
        date = Lgm_CTrans.dateToDateLong(c_date)
        utc = Lgm_CTrans.dateToFPHours(c_date)
        JD = Lgm_JD(c_date.year, c_date.month, c_date.day, utc, LGM_TIME_SYS_UTC, c)

        #Set up the trans matrices
        Lgm_Set_Coord_Transforms( date, utc, c )
        #get SGP4 output, needs minutes-since-TLE-epoch
        tsince = (JD - TLEs.JD)*1440.0
        LgmSgp_SGP4(tsince, pointer(s))

        pos_in[0] = s.X
        pos_in[1] = s.Y
        pos_in[2] = s.Z

        Pin = Lgm_Vector.Lgm_Vector(*pos_in)
        Pout = Lgm_Vector.Lgm_Vector()
        Lgm_Convert_Coords( pointer(Pin), pointer(Pout), TEME_TO_GEO, c)
        PoutPy = Pout.tolist()
        PoutPy[0] /= WGS84_A
        PoutPy[1] /= WGS84_A
        PoutPy[2] /= WGS84_A
        nlat, nlon, nrad = Lgm_Vector.CartToSph(*PoutPy)
        testlat[idx] = nlat
        testlon[idx] = nlon
        testrad[idx] = nrad
        testtdiff[idx] = tsince/1440.0

    return testlat, testlon, testrad