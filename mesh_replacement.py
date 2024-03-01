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


def new_cells(F_avg,cells):
    return (F_avg@cells * np.max(cells/(F_avg@cells))).astype(int)


def regrid_restart(fname_in,fname_out,mapping_flat):
    with (h5py.File(fname_in) as f_in, h5py.File(fname_out,'w') as f_out):
        f_in.copy('homogenization',f_out)

        f_out.create_group('phase')
        for label in f_in['phase']:
            f_out['phase'].create_group(label)
            F_e0 = np.matmul(f_in['phase'][label]['F'][()],np.linalg.inv(f_in['phase'][label]['F_p'][()]))
            R_e0, V_e0 = damask.mechanics._polar_decomposition(F_e0, ['R','V'])
            f_out['phase'][label].create_dataset('F',data=np.broadcast_to(np.eye(3),(len(mapping_flat),3,3,)))
            f_out['phase'][label].create_dataset('F_e',data=R_e0[mapping_flat])
            f_out['phase'][label].create_dataset('F_p',data=damask.tensor.transpose(R_e0)[mapping_flat])
            f_out['phase'][label].create_dataset('S',data=np.zeros((len(mapping_flat),3,3)))
            for d in f_in['phase'][label]:
                if d in f_out[f'phase/{label}']: continue
                f_out['phase'][label].create_dataset(d,data=f_in['phase'][label][d][()][mapping_flat])

        f_out.create_group('solver')
        for d in ['F','F_lastInc']:
            f_out['solver'].create_dataset(d,data=np.broadcast_to(np.eye(3),np.append(cells_new.prod(),(3,3))))
        for d in ['F_aim', 'F_aim_lastInc']:
            f_out['solver'].create_dataset(d,data=np.eye(3))
        f_out['solver'].create_dataset('F_aimDot',data=np.zeros((3,3)))
        for d in f_in['solver']:
            if d not in f_out['solver']: f_in['solver'].copy(d,f_out['solver'])

damask.util.run(f'DAMASK_grid -g {cwd}/{grid}.vti -l {cwd}/{load}.yaml -m {cwd}/{mat}.yaml -w {wd}')
r = damask.Result(f'{wd}/{grid}_{load}_{mat}.hdf5')
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)
F_avg = np.average(r.view(increments=-1).get('F'),axis=0)

cells_new = new_cells(F_avg,r.cells)
mapping = damask.grid_filters.regrid(r.size,r.view(increments=-1).get('F').reshape(tuple(r.cells)+(3,3)),cells_new)
mapping_flat = mapping.reshape(-1,order='F')

g = damask.GeomGrid.load(f'{grid}.vti')
g.size = F_avg@g.size
g.assemble(mapping).save(f'{wd}/{grid2}.vti')

regrid_restart(f'{wd}/{grid}_{load}_{mat}_restart.hdf5',f'{wd}/{grid2}_{load}_{mat}_restart.hdf5',mapping_flat)

r.view(increments=0).export_DADF5(f'{wd}/{grid2}_{load}_{mat}.hdf5',mapping=mapping)

with h5py.File(f'{wd}/{grid2}_{load}_{mat}.hdf5','a') as f:
    f['geometry'].attrs['size'] = g.size

shutil.copyfile(f'{cwd}/{load}-2.yaml',f'{wd}/{load}.yaml')
shutil.copyfile(f'{wd}/{grid}_{load}_{mat}.sta',f'{wd}/{grid2}_{load}_{mat}.sta')
damask.util.run(f'DAMASK_grid -g {grid2}.vti -l {load}.yaml -m {cwd}/{mat}.yaml -r 140 -w {wd}')
r = damask.Result(f'{wd}/{grid2}_{load}_{mat}.hdf5').view_less(increments=0)
r.add_IPF_color([0,0,1])
r.export_VTK(target_dir=cwd)
