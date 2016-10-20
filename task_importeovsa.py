import os
import sys
import gc
import jdutil
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import time
import glob
import aipy
import chan_util_bc as cu
import read_idb as ri
from util import Time


def bl_list2(nant=16):
    ''' Returns a two-dimensional array bl2ord that will translate
        a pair of antenna indexes (antenna number - 1) to the ordinal
        number of the baseline in the 'x' key.  Note bl2ord(i,j) = bl2ord(j,i),
        and bl2ord(i,i) = -1.
    '''
    bl2ord = np.ones((nant, nant), dtype='int') * (-1)
    k = 0
    for i in range(nant):
        for j in range(i, nant):
            bl2ord[i, j] = k
            # bl2ord[j,i] = k
            k += 1
    return bl2ord


def get_band_edge(nband=34):
    # Input the frequencies from UV, returen the indices frequency edges of all bands
    idx_start_freq = [0]
    ntmp = 0
    for i in range(1, nband + 1):
        ntmp += len(cu.start_freq(i))
        idx_start_freq.append(ntmp)
    return np.asarray(idx_start_freq)


def idb2ms(vis,
           doavg,
           timebin,
           width,
           outpath,
           nocreatms,
           modelms,
           doconcat):

    nowritems = False
    if type(vis) == Time:
        filelist = ri.get_trange_files(vis)
    else:
        # If input type is not Time, assume that it is the list of files to read
        filelist = vis

    if not modelms:
        modelms = '/home/user/sjyu/20160531/ms/sun/SUN/SUN_20160531T142234-10m.1s.ms'
    try:
        for f in filelist:
            os.path.exists(f)
    except ValueError:
        print("Some files in filelist are invalid. Aborting...")
    if not outpath:
        # use current directory
        outpath = './'

    for filename in filelist:
        uv = aipy.miriad.UV(filename)
        if uv['source'].lower() == 'sun':
            outpath = outpath + 'sun/'
            if not os.path.exists(outpath):
                os.mkdir(outpath)
        else:
            outpath = outpath + 'calibrator/'
            if not os.path.exists(outpath):
                os.mkdir(outpath)
        uv.rewind()

        start_time = 0  # The start and stop times are referenced to ref_time_jd in second
        end_time = 600
        delta_time = 1
        time_steps = (end_time - start_time) / delta_time
        time0 = time.time()

        if 'antlist' in uv.vartable:
            ants = uv['antlist']
            antlist = map(int, ants.split())
        else:
            antlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        good_idx = np.arange(len(uv['sfreq']))

        ref_time_jd = uv['time']
        ref_time_mjd = jdutil.jd_to_mjd(ref_time_jd) * 24. * 3600. + 0.5 * delta_time
        nf = len(good_idx)
        freq = uv['sfreq'][good_idx]
        npol = uv['npol']
        nants = uv['nants']
        sdf = uv['sdf']
        project = uv['proj']
        source_id = uv['source']
        ra, dec = uv['ra'], uv['dec']
        scan_id = uv['scanid']
        nbl = nants * (nants - 1) / 2
        bl2ord = bl_list2(nants)
        npairs = nbl + nants
        flag = np.ones((npol, nf, time_steps, npairs), dtype=bool)
        out = np.zeros((npol, nf, time_steps, npairs), dtype=np.complex64)  # Cross-correlations
        uvwarray = np.zeros((3, time_steps, npairs), dtype=np.float)
        bandedge = get_band_edge(freq)
        nband = len(bandedge) - 1

        uv.rewind()
        l = -1
        for preamble, data in uv.all():
            uvw, t, (i0, j0) = preamble
            i = antlist.index(i0 + 1)
            j = antlist.index(j0 + 1)
            if i > j:
                # Reverse order of indices
                j = antlist.index(i0 + 1)
                i = antlist.index(j0 + 1)
            # Assumes uv['pol'] is one of -5, -6, -7, -8
            k = -5 - uv['pol']
            l += 1
            out[k, :, l / (npairs * npol), bl2ord[i0, j0]] = data.data
            flag[k, :, l / (npairs * npol), bl2ord[i0, j0]] = data.mask
            if i != j:
                if k == 3:
                    uvwarray[:, l / (npairs * npol), bl2ord[i0, j0]] = -uvw * constants.speed_of_light / 1e9

        nrows = time_steps * npairs
        out = out.reshape(npol, nf, nrows)
        flag = flag.reshape(npol, nf, nrows)
        uvwarray = uvwarray.reshape(3, nrows)
        uvwarray = np.tile(uvwarray, (1, nband))
        sigma = np.ones((4, nrows), dtype=np.float) + 1
        sigma = np.tile(sigma, (1, nband))

        print 'IDB File {0} is readed in --- {1:10.2f} seconds ---'.format(filename, (time.time() - time0))

        msname = list(filename.split('/')[-1])
        msname.insert(11, 'T')
        msname = outpath + source_id.upper() + '_' + ''.join(msname[3:]) + '-10m.1s.ms'

        if not nocreatms:
            if os.path.exists(msname):
                os.system("rm -fr %s" % msname)
            """ Creates an empty measurement set using CASA simulate (sm) tool. """
            sm.open(msname)

            enu = np.reshape(uv['antpos'], (16, 3)) * constants.speed_of_light / 1e9
            refpos_wgs84 = me.position('wgs84',
                                       '-118.286952892965deg',
                                       '37.2331698901026deg',
                                       '1207.1339m')
            lon, lat, rad = [me.measure(refpos_wgs84, 'itrf')[x]['value'] for x in 'm0', 'm1', 'm2']
            # 3x3 transform matrix. Each row is a normal vector, i.e. the rows are (dE,dN,dU)
            # ----------- local xyz ------------
            xform = np.array([
                [0, -np.sin(lat), np.cos(lat)],
                [1, 0, 0],
                [0, np.cos(lat), np.sin(lat)]])
            xyz = enu.dot(xform)  # + xyz0[np.newaxis,:]

            # ----------- global xyz ------------
            # xyz0 = rad*np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
            # # 3x3 transform matrix. Each row is a normal vector, i.e. the rows are (dE,dN,dU)
            # xform = np.array([
            #     [-np.sin(lon),np.cos(lon),0],
            #     [-np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)],
            #     [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
            # ])
            # xyz = xyz0[np.newaxis,:] + enu.dot(xform)

            dishdiam = np.full(uv['nants'], 2.1)
            dishdiam[-3:-1] = 27
            dishdiam[-1] = 2.1
            station = uv['telescop']
            mount = ['ALT-AZ'] * uv['nants']
            for l in [8, 9, 10, 12, 13, 14]:
                mount[l] = 'EQUATORIAL'
            sm.setconfig(telescopename=station,
                         x=np.asarray(xyz)[:, 0],
                         y=np.asarray(xyz)[:, 1],
                         z=np.asarray(xyz)[:, 2],
                         dishdiameter=dishdiam,
                         mount=mount,
                         antname=['eo' + "{0:02d}".format(l) for l in antlist],
                         padname=station,
                         coordsystem='local', referencelocation=refpos_wgs84)

            sm.setfield(sourcename=source_id,
                        sourcedirection=me.direction('J2000',
                                                     '{:22.19f}'.format(uv['obsra']) + 'rad',
                                                     '{:22.19f}'.format(uv['obsdec']) + 'rad'))
            sm.setfeed(mode='perfect X Y')

            ref_time = me.epoch('tai',
                                '{:20.13f}'.format(jdutil.jd_to_mjd(ref_time_jd)) + 'd')

            sm.settimes(integrationtime='1s',
                        usehourangle=False,
                        referencetime=ref_time)

            for l, bdedge in enumerate(bandedge[:-1]):
                nchannels = (bandedge[l + 1] - bandedge[l])
                stokes = 'XX YY XY YX'
                df = sdf[bandedge[l]]
                st_freq = freq[bandedge[l]]

                sm.setspwindow(spwname='band%02d' % (l + 1),
                               freq='{:22.19f}'.format(st_freq) + 'GHz',
                               deltafreq='{:22.19f}'.format(df) + 'GHz',
                               freqresolution='{:22.19f}'.format(df) + 'GHz',
                               nchannels=nchannels,
                               stokes=stokes)

            nband = len(bandedge) - 1
            for bdid in range(nband):
                sm.observe(source_id, 'band%02d' % (bdid + 1),
                           starttime=start_time, stoptime=end_time,
                           project=project,
                           state_obs_mode='')

            if sm.done():
                print 'Empty MS {0} created in --- {1:10.2f} seconds ---'.format(msname, (time.time() - time0))
            else:
                raise RuntimeError('Failed to create MS. Look at the log file. '
                                   'Double check you settings.')
        else:
            print '----------------------------------------'
            print 'copying standard MS to {0}'.format(msname, (time.time() - time0))
            print '----------------------------------------'
            os.system("rm -fr %s" % msname)
            os.system("cp -r " + " %s" % modelms + " %s" % msname)
            print 'Standard MS is copied to {0} in --- {1:10.2f} seconds ---'.format(msname, (time.time() - time0))

        if not nowritems:
            tb.open(msname, nomodify=False)
            print '----------------------------------------'
            print "Updating the main table of" '%s' % msname
            print '----------------------------------------'
            for l, bdedge in enumerate(bandedge[:-1]):
                time1 = time.time()
                nchannels = (bandedge[l + 1] - bandedge[l])
                for row in range(nrows):
                    tb.putcell('DATA', (row + l * nrows), out[:, bandedge[l]:bandedge[l + 1], row])
                    tb.putcell('FLAG', (row + l * nrows), flag[:, bandedge[l]:bandedge[l + 1], row])
                print '---spw {0:02d} is updated in --- {1:10.2f} seconds ---'.format((l + 1), time.time() - time1)
            tb.putcol('UVW', uvwarray)
            tb.putcol('SIGMA', sigma)
            tb.putcol('WEIGHT', 1.0 / sigma ** 2)
            timearr = np.arange((time_steps), dtype=np.float)
            timearr = timearr.reshape(1, time_steps, 1)
            timearr = np.tile(timearr, (nband, 1, npairs))
            timearr = timearr.reshape(nband * npairs * time_steps) + ref_time_mjd
            tb.putcol('TIME', timearr)
            tb.putcol('TIME_CENTROID', timearr)
            colnames = tb.colnames()
            cols2rm = ["MODEL_DATA", "CORRECTED_DATA"]
            for l in range(len(cols2rm)):
                if cols2rm[l] in colnames:
                    tb.removecols(cols2rm[l])
            tb.close()

            print '----------------------------------------'
            print "Updating the OBSERVATION table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/OBSERVATION', nomodify=False)
            tb.putcol('TIME_RANGE',
                      np.asarray([ref_time_mjd - 0.5 * delta_time, ref_time_mjd + end_time - 0.5 * delta_time]).reshape(
                          2, 1))
            tb.putcol('OBSERVER', ['EOVSA team'])
            tb.close()

            print '----------------------------------------'
            print "Updating the POINTING table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/POINTING', nomodify=False)
            timearr = np.arange((time_steps), dtype=np.float).reshape(1, time_steps, 1)
            timearr = np.tile(timearr, (nband, 1, nants))
            timearr = timearr.reshape(nband * time_steps * nants) + ref_time_mjd
            tb.putcol('TIME', timearr)
            tb.putcol('TIME_ORIGIN', timearr - 0.5 * delta_time)
            direction = tb.getcol('DIRECTION')
            direction[0, 0, :] = ra
            direction[1, 0, :] = dec
            tb.putcol('DIRECTION', direction)
            target = tb.getcol('TARGET')
            target[0, 0, :] = ra
            target[1, 0, :] = dec
            tb.putcol('TARGET', target)
            tb.close()

            print '----------------------------------------'
            print "Updating the SOURCE table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/SOURCE', nomodify=False)
            radec = tb.getcol('DIRECTION')
            radec[0], radec[1] = ra, dec
            tb.putcol('DIRECTION', radec)
            name = np.array([source_id], dtype='|S{0}'.format(len(source_id) + 1))
            tb.putcol('NAME', name)
            tb.close()

            print '----------------------------------------'
            print "Updating the DATA_DESCRIPTION table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/DATA_DESCRIPTION/', nomodify=False)
            pol_id = tb.getcol('POLARIZATION_ID')
            pol_id *= 0
            tb.putcol('POLARIZATION_ID', pol_id)
            spw_id = tb.getcol('SPECTRAL_WINDOW_ID')
            spw_id *= 0
            tb.putcol('SPECTRAL_WINDOW_ID', spw_id)
            tb.close()

            print '----------------------------------------'
            print "Updating the POLARIZATION table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/POLARIZATION/', nomodify=False)
            tb.removerows(rownrs=np.arange(1, nband, dtype=int))
            tb.close()

            print '----------------------------------------'
            print "Updating the FIELD table of" '%s' % msname
            print '----------------------------------------'
            tb.open(msname + '/FIELD/', nomodify=False)
            delay_dir = tb.getcol('DELAY_DIR')
            delay_dir[0], delay_dir[1] = ra, dec
            tb.putcol('DELAY_DIR', delay_dir)
            phase_dir = tb.getcol('PHASE_DIR')
            phase_dir[0], phase_dir[1] = ra, dec
            tb.putcol('PHASE_DIR', phase_dir)
            reference_dir = tb.getcol('REFERENCE_DIR')
            reference_dir[0], reference_dir[1] = ra, dec
            tb.putcol('REFERENCE_DIR', reference_dir)
            name = np.array([source_id], dtype='|S{0}'.format(len(source_id) + 1))
            tb.putcol('NAME', name)
            tb.close()

            # FIELD: DELAY_DIR, PHASE_DIR, REFERENCE_DIR, NAME


            del out, flag, uvwarray, uv, timearr, sigma
            gc.collect()  #

        print("finished in --- %s seconds ---" % (time.time() - time0))
        return True



