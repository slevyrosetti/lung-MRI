#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program creates a GIF animation for a 4D volume.


Created on Mon Oct  14 19:21:50 2019

@author: slevy
"""

import nibabel as nib
import argparse
import imageio
import numpy as np
import time
import os
import shutil
# import dsc_utils
import matplotlib.pyplot as plt
import scipy.misc

clims = [0, 2000]

def main(iFname, oFname, duration, cropX, cropY, cropT, iFname2, maskFname):

    # load data
    data4d = nib.load(iFname).get_fdata()

    # define cropping boundaries
    if not cropX: cropX = [0, data4d.shape[0]-1]
    if not cropY: cropY = [0, data4d.shape[1]-1]
    if not cropT: cropT = [0, data4d.shape[3]-1]

    if not iFname2:
        # directly create GIF (without generating PNG)
        with imageio.get_writer(oFname+'.gif', mode='I', duration=duration) as writer:
            for i_vol in range(cropT[0], cropT[1]+1):
                vol_3slices_view = (np.rot90(np.squeeze(data4d)[cropX[0]:cropX[1], cropY[0]:cropY[1], i_vol]))
                writer.append_data(vol_3slices_view)

    else:

        # load data
        data4d_2 = nib.load(iFname2).get_fdata()

        # if not maskFname:
        #     # save all frames to PNG in a temporary directory
        #     tmpDirPath = oFname+"_%s" % time.strftime("%y%m%d%H%M%S")
        #     os.makedirs(tmpDirPath)
        #     for i_vol in range(cropT[0], cropT[1]+1):
        #         vol_3slices_view = np.concatenate((np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 0, i_vol]),
        #                                            np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 1, i_vol]),
        #                                            np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 2, i_vol])), axis=0)
        #         imageio.imwrite(tmpDirPath + '/frame' + str(i_vol) + '.jpeg', vol_3slices_view, cmin=0.0, cmax=500.0)
        #
        # else:

        # save all frames to PNG in a temporary directory
        tmpDirPath = oFname+"_%s" % time.strftime("%y%m%d%H%M%S")
        os.makedirs(tmpDirPath)

        for i_vol in range(cropT[0], cropT[1] + 1):

            fig_i_vol, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 10))
            plt.subplots_adjust(wspace=-0.1, left=0, right=1.00, hspace=0.0, bottom=-0.1, top=1.0)

            ax1.imshow(np.rot90(np.squeeze(data4d)[cropX[0]:cropX[1], cropY[0]:cropY[1], i_vol]), cmap='gray', clim=clims)
            # ax1.imshow(mask_3slices_view, cmap='Blues', alpha=0.5, clim=(.5, 1))
            # ax1.set_axis_off()
            ax1.set_xlabel('Rep #'+str(i_vol))
            ax1.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax1.patch.set_visible(False)
            ax1.text(17, -5, "(A)", size=50, ha="center", color="w", weight='bold')

            ax2.imshow(np.rot90(np.squeeze(data4d_2)[cropX[0]:cropX[1], cropY[0]:cropY[1], i_vol]), cmap='gray', clim=clims)
            ax2.set_xlabel('Rep #'+str(i_vol))
            ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            ax2.patch.set_visible(False)
            ax2.text(17, -5, "(B)", size=50, ha="center", color="w", weight='bold')

            fig_i_vol.patch.set_facecolor('black')

            fig_i_vol.savefig(tmpDirPath + '/frame' + str(i_vol) + '.jpeg', transparent=False, quality=40, dpi=25)
            plt.close(fig_i_vol)
            print('Save to tmp dir: ' + tmpDirPath + '/frame' + str(i_vol) + '.jpeg')

        # prepare and run the imageMagick command with variable effective TR
        shellCmd = 'convert '
        for i_vol in range(cropT[0], cropT[1] + 1):
            shellCmd += '-delay '+str(duration/100)+' '+ tmpDirPath + '/frame' + str(i_vol) + '.jpeg '
        shellCmd += '-loop 0 '+oFname+'.gif'
        print('Run command: '+shellCmd)
        os.system(shellCmd)
        # time.sleep(5)
        shutil.rmtree(tmpDirPath)

    print('\nSaved to: '+oFname+'.gif')


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program creates a GIF animation from a 4D volume.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to 4D MRI data.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help='Filename for output GIF (do not include the extension .gif).', type=str, required=True)

    optionalArgs.add_argument('-i2', dest='iFname2', help='Path to a second 4D MRI data to plot aside of the first one.', type=str, required=False, default='')
    optionalArgs.add_argument('-d', dest='duration', help='Duration between each frame (in seconds).', type=float, required=False, default=0.5)
    optionalArgs.add_argument('-cx', dest='cropX', help='Cropping boundaries along X as x1,x2.', type=str, required=False, default='')
    optionalArgs.add_argument('-cy', dest='cropY', help='Cropping boundaries along Y as y1,y2.', type=str, required=False, default='')
    optionalArgs.add_argument('-ct', dest='cropT', help='Cropping boundaries along T as t1,t2.', type=str, required=False, default='')
    optionalArgs.add_argument('-m', dest='maskFname', help='Path to mask for the 4D MRI data in input.', type=str, required=False, default='')

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    cropX, cropY, cropT = None, None, None
    if args.cropX: cropX = list(map(int, args.cropX.split(',')))
    if args.cropY: cropY = list(map(int, args.cropY.split(',')))
    if args.cropT: cropT = list(map(int, args.cropT.split(',')))

    # run main
    main(iFname=args.iFname, oFname=args.oFname, duration=args.duration, cropX=cropX,
         cropY=cropY, cropT=cropT, iFname2=args.iFname2, maskFname=args.maskFname)

