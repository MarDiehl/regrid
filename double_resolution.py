#!/usr/bin/env python3

import tempfile
import shutil
import os

import numpy as np
import h5py

import damask

load = 'tensionX'
grid = '20grains16x16x16'
mat = 'material'
grid2 = grid+'-2'

cwd = os.getcwd()
print(wd := tempfile.mkdtemp())


damask.util.run(f'DAMASK_grid -g {cwd}/{grid}.vti -l {cwd}/{load}.yaml -m {cwd}/{mat}.yaml -w {wd}')
r = damask.Result(f'{wd}/{grid}_{load}_{mat}.hdf5')
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd,mode='cell')
r.export_VTK(target_dir=cwd,mode='point')
mapping = damask.grid_filters.regrid(r.size,np.broadcast_to(np.eye(3),tuple(r.cells)+(3,3)),r.cells*2)
r.export_DADF5(f'{wd}/double.hdf5',mapping=mapping)
r = damask.Result(f'{wd}/double.hdf5')
r.export_VTK(target_dir=cwd,mode='cell')
r.export_VTK(target_dir=cwd,mode='point')
