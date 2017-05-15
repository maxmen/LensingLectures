import numpy as np
from astropy import constants as const
from astropy import units as u

# this class deals with the source in the microlensing event
class point_source(object):
    
    def __init__(self,flux=1.0,ds=10.0,vel=200.):
        self.ds=ds
        self.flux=flux
        self.vel=vel
        
# this class deals with the lens. It requires a point source to be 
# provided in order to build the point lens
class point_lens(object):

    # the constructor of the microlens
    def __init__(self,ps,mass=1.0,dl=5.0,ds=8.0,t0=0.0,y0=0.1):
        self.M=mass
        self.dl=dl
        self.ps=ps
        self.y0=y0
        self.t0=t0
        self.tE=self.EinsteinCrossTime()
    
    # a function returning the Einstein radius
    def EinsteinRadius(self):
        mass=self.M*const.M_sun
        G=const.G
        c=c=const.c
        aconv=180.0*3600.0/np.pi*u.arcsecond
        return((np.sqrt(4.0*(G*mass/c/c).to('kpc')*(self.ps.ds-self.dl)
                        /self.dl/self.ps.ds/u.kpc))*aconv)
    
    # a function retruning the Einstein radius crossing time
    def EinsteinCrossTime(self):
        theta_e=self.EinsteinRadius()
        return(((theta_e.to('radian').value*self.dl*u.kpc).to('km')
                /self.ps.vel/u.km*u.s).to('day'))
    
    # a function returning the coordinates of the unlensed source 
    # at time t
    def y(self,t):
        y1=(t-self.t0)/self.tE.value
        y2=np.ones(len(t))*self.y0
        return(y1,y2)
    
    # a function returning the coordinates of the x_+ image at time t
    def xp(self,t):
        y1, y2  = self.y(t)
        Q = np.sqrt(y1**2 + y2**2 +4)/(np.sqrt(y1**2 + y2**2))
        xp1= 0.5 *(1 + Q)* y1
        xp2= 0.5 *(1 + Q)* y2
        return(xp1, xp2)
    
    # a function retruning the coordinates of the x_- image at time t
    def xm(self,t):
        y1, y2  = self.y(t)
        Q = np.sqrt(y1**2 + y2**2 +4)/(np.sqrt(y1**2 + y2**2))
        xm1= 0.5 *(1 - Q)* y1
        xm2= 0.5 *(1 - Q)* y2
        return(xm1, xm2)
    
    # the magnification of the x_+ image
    def mup(self,t):
        y1, y2  = self.y(t)
        yy=np.sqrt(y1**2+y2**2)
        mup=0.5*(1+(yy**2+2)/yy/np.sqrt(yy**2+4))
        return (mup)
    
    # the magnification of the x_- image
    def mum(self,t):
        y1, y2  = self.y(t)
        yy=np.sqrt(y1**2+y2**2)
        mum=0.5*(1-(yy**2+2)/yy/np.sqrt(yy**2+4))
        return (mum)
    
    # a function retruning the coordinate of the light centroid
    def xc(self,t):
        xp=self.xp(t)
        xm=self.xm(t)
        xc=(xp*np.abs(self.mup(t))+xm*np.abs(self.mum(t)))/(np.abs(self.mup(t))+np.abs(self.mum(t)))
        return (xc)
    
    ################################################################################################
    def xp_ext_source(self,t,r):
        phi=np.linspace(0.0,2*np.pi,360)
        dy1=r*np.cos(phi)
        dy2=r*np.sin(phi)
        y1,y2=self.y(t)
        yy1=y1+dy1
        yy2=y2+dy2
        Q=np.sqrt(yy1**2+yy2**2+4.0)/np.sqrt(yy1**2+yy2**2)
        xp1=0.5*(1+Q)*yy1
        xp2=0.5*(1+Q)*yy2
        return(xp1,xp2)   
    
    def xm_ext_source(self,t,r):
        phi=np.linspace(0.0,2*np.pi,360)
        dy1=r*np.cos(phi)
        dy2=r*np.sin(phi)
        y1,y2=self.y(t)
        yy1=y1+dy1
        yy2=y2+dy2
        Q=np.sqrt(yy1**2+yy2**2+4.0)/np.sqrt(yy1**2+yy2**2)
        xm1=0.5*(1-Q)*yy1
        xm2=0.5*(1-Q)*yy2
        return(xm1,xm2)
    
    def deltaxc(self,t):
        y1,y2=self.y(t)
        yy=(y1**2+y2**2)
        return(y1/(yy+2),y2/(yy+2))


class binary_lens(object):
    """
    The object binary_lens will be built using the mass of the first lens, the mass ratio
    and the distance between the lenses in units of the equivalent Einstein radius.
    By convention, we will place the two lenses on the real axis and will put the origin of 
    the reference frame in the midpoint between the two masses.
    """
    def __init__(self,ps,dl=5.0,m1=1.0,q=1.0,d=2.0,t0=0.0,y0=0.1,theta=np.pi/4):
        self.z1=complex(d/2.0,0.0)
        self.q=q
        self.dl=dl
        self.ds=ps.ds
        m2=m1/q
        self.mtot=m1+m2
        self.m1=m1/self.mtot
        self.m2=m2/self.mtot
        pl = point_lens(ps=ps, mass=m1+m2, dl=dl)
        self.pl=pl
        self.thetaE=pl.EinsteinRadius()
        self.tE=pl.EinsteinCrossTime()
        self.t0=t0
        self.y0=y0
        self.theta=theta

    """
    This function finds the lens critical lines and caustics
    """
    def CritCau(self,ncpt=10000):
        # set the phase vector
        phi_=np.linspace(0,2.*np.pi,ncpt)
        
        x=[]
        y=[]
        xs=[]
        ys=[]

        # we need to find the roots of our fourth order polynomial for each value of phi
        for i in range(phi_.size):
            phi=phi_[i]
            # the coefficients of the complex polynomial
            coefficients = [1.0,0.0,-2*np.conj(self.z1)**2-np.exp(1j*phi),
                            -np.conj(self.z1)*2*(self.m1-self.m2)*np.exp(1j*phi),
                            np.conj(self.z1)**2*(np.conj(self.z1)**2-np.exp(1j*phi))]
            # use the numpy function roots to find the roots of the polynomial
            z=np.roots(coefficients) # these are the critical points!
    
            # use the lens equation (complex form) to map the critical points on the source plane 
            zs=z-self.m1/(np.conj(z)-np.conj(self.z1))-self.m2/((np.conj(z)-np.conj(-self.z1))) # these are the caustics!
    
            # append critical and caustic points
            x.append(z.real)
            y.append(z.imag)
            xs.append(zs.real)
            ys.append(zs.imag)
        
        return(np.array(x),np.array(y),np.array(xs),np.array(ys))
    
    """
    This function finds the images of a source at a given position with respect to the lens
    """
    def Images(self,ys1,ys2):
        zs=complex(ys1,ys2)
        m=0.5*(self.m1+self.m2)
        Dm=(self.m2-self.m1)/2.0

        c5=self.z1**2-np.conj(zs)**2
        c4=-2*m*np.conj(zs)+zs*np.conj(zs)**2-2*Dm*self.z1-zs*self.z1**2
        c3=4.0*m*zs*np.conj(zs)+4.0*Dm*np.conj(zs)*self.z1+2.0*np.conj(zs)**2*self.z1**2-2.0*self.z1**4
        c2=4.0*m**2*zs+4.0*m*Dm*self.z1-4.0*Dm*zs*np.conj(zs)*self.z1-2.0*zs*np.conj(zs)**2\
                *self.z1**2+4.0*Dm*self.z1**3+2.0*zs*self.z1**4
        c1=-8.0*m*Dm*zs*self.z1-4.0*Dm**2*self.z1**2-4.0*m**2*self.z1**2-4.0*m*zs*np.conj(zs)*self.z1**2\
                -4.0*Dm*np.conj(zs)*self.z1**3-np.conj(zs)**2*self.z1**4+self.z1**6
        c0=self.z1**2*(4.0*Dm**2*zs+4.0*m*Dm*self.z1+4.0*Dm*zs*np.conj(zs)*self.z1+\
                   2.0*m*np.conj(zs)*self.z1**2+zs*np.conj(zs)**2*self.z1**2-2*Dm*self.z1**3-zs*self.z1**4)

        coefficients=[c5,c4,c3,c2,c1,c0]

        images=np.roots(coefficients)
        #print images
        # now, we need to drop the spurious solutions. This can be done by checking which solutions 
        # satisfy the lens equation
        z2=-self.z1
        deltazs=zs-(images-self.m1/(np.conj(images)-np.conj(self.z1))-self.m2/(np.conj(images)-np.conj(z2)))
        #print np.abs(deltazs)
        return (np.array([images.real[np.abs(deltazs)<1e-3]]),np.array([images.imag[np.abs(deltazs)<1e-3]]))
    
    def SourcePos(self,t):
        p=(t-self.t0)/self.tE.value
        zreal=np.cos(self.theta)*p+np.sin(self.theta)*self.y0
        zimag=-np.sin(self.theta)*p+np.cos(self.theta)*self.y0
        return(zreal,zimag)
    
    def detA(self,z):
        z2=-self.z1
        deta=1-np.abs(self.m1/(np.conj(z)-np.conj(self.z1))**2+self.m2/(np.conj(z)-np.conj(z2))**2)
        return(deta)
    
    def Magnification(self,t):
        ys1,ys2=self.SourcePos(t)
        xi1,xi2=self.Images(ys1,ys2)
        images=xi1+1j*xi2
        mu=1.0/self.detA(images)
        return(np.abs(mu).sum())
    
    def LightCurve(self,times):
        p=(times-self.t0)/self.tE.value
        mu=[]
        for t in times:
            mu.append(self.Magnification(t))
        #mu=np.array(mu)
        return(p,mu)
        
    
    """
    Some utilities
    """
    def getPos(self):
        return(self.z1)
        
    def gettE(self):
        return(self.tE)
    
    def getThetaE(self):
        return(self.thetaE)
    
    def WideIntTrans(self):
        dwi=((self.m1)**(1./3.)+(self.m2)**(1./3.))**(3./2.)
        return(dwi)
    
    def IntCloseTrans(self):
        dic=((self.m1)**(1./3.)+(self.m2)**(1./3.))**(-3./4.)
        return(dic)