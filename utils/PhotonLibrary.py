import h5py  as h5
import numpy as np
import os

class PhotonLibrary:

    def __init__(self,fname=None):
        if fname is None:
            fname = 'plib_compressed.h5'
            if not os.path.isfile(fname):
                print('Downloading photon library file...')
                os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib_compressed.h5 ./')
            if not os.path.isfile(fname):
                print('Error: failed to download the photon library file...')
                raise Exception
        f = h5.File(fname,'r')
        self._plib = np.array(f['plib'])
        self._xmin = float(f['min'][0])
        self._ymin = float(f['min'][1])
        self._zmin = float(f['min'][2])
        self._xmax = float(f['max'][0])
        self._ymax = float(f['max'][1])
        self._zmax = float(f['max'][2])
        self._nx   = int(f['numvox'][0])
        self._ny   = int(f['numvox'][1])
        self._nz   = int(f['numvox'][2])
        

    def GetVoxelID(self, x, y, z):

        xid = int((x-self._xmin) / (self._xmax-self._xmin) * self._nx)
        yid = int((y-self._ymin) / (self._ymax-self._ymin) * self._ny)
        zid = int((z-self._zmin) / (self._zmax-self._zmin) * self._nz)

        if xid<0 or self._nx<=xid or yid<0 or self._ny<=yid or zid<0 or self._nz<=zid:
            return -1

        return xid + yid * self._ny + zid * (self._nx * self._ny)


    def GetVisibility(self, x, y, z, OpChannel):
        return self._plib[OpChannel][self.GetVoxelID(x,y,z)]


    def XRange(self): return (self._xmin,self._xmax)

    
    def YRange(self): return (self._ymin,self._ymax)

    
    def ZRange(self): return (self._zmin,self._zmax)

    
    def VisibilityYZ(self, x_frac, ny, nz):
        x  = self._xmin + x_frac * (self._xmax - self._xmin)
        ys = self._ymin + np.arange(ny) * (self._ymax - self._ymin)
        zs = self._zmin + np.arange(nz) * (self._zmax - self._zmin)
        result = np.zeros(shape=[ny,nz],dtype=np.float64)
        for iy in range(ny):
            for iz in range(nz):
                voxel_id = self.GetVoxelID(x,ys[iy],zs[iz])
                for ch in range(len(self._plib)):
                    result[iy][iz] += self._plib[ch][voxel_id]                
        return result
    
    
    def VisibilityXY(self, z_frac, nx, ny):
        xs = self._xmin + np.arange(nx) * (self._xmax - self._xmin)
        ys = self._ymin + np.arange(ny) * (self._ymax - self._ymin)
        z  = self._zmin + z_frac * (self._zmax - self._zmin)
        result = np.zeros(shape=[nx,ny],dtype=np.float64)
        for ix in range(nx):
            for iy in range(ny):
                voxel_id = self.GetVoxelID(xs[ix],ys[iy],z)
                for ch in range(len(self._plib)):
                    result[ix][iy] += self._plib[ch][voxel_id]                
        return result
