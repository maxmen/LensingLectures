import numpy as np
import numpy.fft as fftengine
import astropy.io.fits as pyfits 


class deflector:
    
    # initialize the deflector using a surface density (covergence) map
    # the boolean variable pad indicates whether zero-padding is used or not
    def __init__(self,filekappa,pad=False):
        kappa,header=pyfits.getdata(filekappa,header=True)
        self.kappa=2.0*kappa
        self.nx=kappa.shape[0]
        self.ny=kappa.shape[1]
        self.pad=pad
        if (pad):
            self.kpad()
        self.kx,self.ky=self.kernel()
        self.pot=self.potential()
        if (pad):
            self.pot=self.mapCrop(self.pot)
        self.a2,self.a1=np.gradient(self.pot)
        self.a12,self.a11=np.gradient(self.a1)
        self.a22,self.a21=np.gradient(self.a2)
        self.pixel=header['CDELT2']
        
    # implement the kernel function K
    def kernel(self):
        x=np.linspace(-0.5,0.5,self.kappa.shape[0])
        y=np.linspace(-0.5,0.5,self.kappa.shape[1])
        kx,ky=np.meshgrid(x,y)
        norm=(kx**2+ky**2+1e-12)
        kx=kx/norm/np.pi
        ky=ky/norm/np.pi
        return(kx,ky)
    
    # compute the deflection angle maps by convolving
    # the surface density with the kernel function
    def angles(self):
        # FFT of the surface density and of the two components of the kernel
        kappa_ft = fftengine.fftn(self.kappa,axes=(0,1))
        kernelx_ft = fftengine.fftn(self.kx,axes=(0,1),
                                     s=self.kappa.shape)
        kernely_ft = fftengine.fftn(self.ky,axes=(0,1),
                                     s=self.kappa.shape)
        # perform the convolution in Fourier space and transform the result
        # back in real space. Note that a shift needs to be applied using 
        # fftshift
        alphax = 2.0/(self.kappa.shape[0])/(np.pi)**2*\
                fftengine.fftshift(fftengine.ifftn(2.0*\
                np.pi*kappa_ft*kernelx_ft))
        alphay = 2.0/(self.kappa.shape[0])/(np.pi)**2*\
                fftengine.fftshift(fftengine.ifftn(2.0*\
                np.pi*kappa_ft*kernely_ft))
        return(alphax.real,alphay.real)
    
    # returns the surface-density (convergence) of the deflector
    def kmap(self):
        return(self.kappa)
    
    # performs zero-padding
    def kpad(self):
        # add zeros around the original array
        def padwithzeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        # use the pad method from numpy.lib to add zeros (padwithzeros) in a
        # frame with thickness self.kappa.shape[0]
        self.kappa=np.lib.pad(self.kappa, self.kappa.shape[0]*2, 
                              padwithzeros)
    
    # crop the maps to remove zero-padded areas and get back to the original 
    # region.
    def mapCrop(self,mappa):
        xmin=self.kappa.shape[0]/2-self.nx/2
        ymin=self.kappa.shape[1]/2-self.ny/2
        xmax=xmin+self.nx
        ymax=ymin+self.ny
        mappa=mappa[xmin:xmax,ymin:ymax]
        return(mappa)
    
    # alternative using astropy.convolve, which can also include zero-padding
    # and many other features:
    # http://docs.astropy.org/en/stable/api/astropy.convolution.convolve_fft.html
    def angles_alternative(self):
        from astropy.convolution import convolve, convolve_fft
        angx = 2.0*np.pi*convolve_fft(self.kappa, self.kx, fft_pad=True)
        angy = 2.0*np.pi*convolve_fft(self.kappa, self.ky, fft_pad=True)
        return(angx,angy)


    def potential(self):
        # define an array of wavenumbers (two components k1,k2)
        k = np.array(np.meshgrid(fftengine.fftfreq(self.kappa.shape[0])\
                                 ,fftengine.fftfreq(self.kappa.shape[1])))
        pix=1 # pixel scale (now using pixel units)
        #Compute Laplace operator in Fourier space = -4*pi*l*l
        kk = k[0]**2 + k[1]**2
        kk[0,0] = 1.0
        #FFT of the convergence
        kappa_ft = fftengine.fftn(self.kappa)
        #compute the FT of the potential
        kappa_ft *= - (pix)**2 / (kk * (2.0*np.pi**2))
        kappa_ft[0,0] = 0.0
        potential=fftengine.ifftn(kappa_ft)
        return potential.real

    def convergence_map(self):
        return(0.5*(self.a11+self.a22))

    def shear_map(self):
        gamma1=0.5*(self.a11-self.a22)
        gamma2=self.a12
        return(gamma1,gamma2)

    def angle_map(self):
        return(self.a1,self.a2)

    def getPotential(self):
    	return(self.pot)

    def getPixel(self):
    	return(self.pixel)


