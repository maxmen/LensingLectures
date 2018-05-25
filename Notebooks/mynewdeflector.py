font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 25}
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rc('font', **font)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import map_coordinates
import pylab
import astropy.io.fits as pyfits
from scipy.ndimage import map_coordinates
import numpy.fft as fftengine


# the parent class
class gen_lens(object):
    def __init__(self):
        self.pot_exists=False

    
    # geometrical time delay
    def t_geom_surf(self, beta=None):
        x = np.arange(0, self.npix, 1, float)*self.pixel
        y = x[:,np.newaxis]
        if beta is None:
            x0 = y0 = self.npix / 2*self.pixel
        else:
            x0 = beta[0]+self.npix/2*self.pixel
            y0 = beta[1]+self.npix/2*self.pixel
        
        return 0.5*((x-x0)*(x-x0)+(y-y0)*(y-y0))
    
    # gravitational time delay
    def t_grav_surf(self):
        return -self.pot
    
    # total time delay
    def t_delay_surf(self,beta=None):
        t_grav=self.t_grav_surf()
        t_geom=self.t_geom_surf(beta)
        td=(t_grav+t_geom)
        return(t_grav+t_geom)
    
    # convergence 
    def convergence(self):
        if (self.pot_exists):
            kappa=0.5*(self.a11+self.a22)
        else:
            print ("The lens potential is not initialized yet")
            
        return(kappa)
    
    #shear
    def shear(self):
        if (self.pot_exists):
            g1=0.5*(self.a11-self.a22)
            g2=self.a12
        else:
            print ("The lens potential is not initialized yet")
        return(g1,g2)
    
    # determinant of the Jacobian matrix
    def detA(self):
        if (self.pot_exists):
            deta=(1.0-self.a11)*(1.0-self.a22)-self.a12*self.a21
        else:
            print ("The lens potential is not initialized yet")
        return(deta)
    
    # critical lines overlaid to the map of detA, returns a set of contour objects
    def crit_lines(self,ax=None,show=True):
        if (ax==None): 
            print ("specify the axes to display the critical lines")
        else:
            deta=self.detA()
            #ax.imshow(deta,origin='lower')
            cs=ax.contour(deta,levels=[0.0],colors='white',alpha=0.0)
            if show==False:
                ax.clear()
        return(cs)
    
    # plot of the critical lines
    def clines(self,ax=None,color='red',alpha=1.0,lt='-',fontsize=15):
        cs=self.crit_lines(ax=ax,show=False)
        contour=cs.collections[0]
        p=contour.get_paths()
        sizevs=np.empty(len(p),dtype=int)
        
        no=self.pixel
        # if we found any contour, then we proceed
        if (sizevs.size > 0):
            for j in range(len(p)):
                # for each path, we create two vectors containing 
                #the x1 and x2 coordinates of the vertices
                vs = contour.get_paths()[j].vertices 
                sizevs[j]=len(vs)
                x1=[]
                x2=[]
                for i in range(len(vs)):
                    xx1,xx2=vs[i]
                    x1.append(float(xx1))
                    x2.append(float(xx2))
        
                # plot the results!
                ax.plot((np.array(x1)-self.npix/2.)*no,
                        (np.array(x2)-self.npix/2.)*no,lt,color=color,alpha=alpha)
        ax.set_xlabel(r'$\theta_1$',fontsize=fontsize)
        ax.set_ylabel(r'$\theta_2$',fontsize=fontsize)
        return(p)
    
    # plot of the caustics
    def caustics(self,ax=None,alpha=1.0,color='red',lt='-',fontsize=15):
        cs=self.crit_lines(ax=ax,show=True)
        contour=cs.collections[0]
        p=contour.get_paths() # p contains the paths of each individual 
                              # critical line
        sizevs=np.empty(len(p),dtype=int)
        
        # if we found any contour, then we proceed
        if (sizevs.size > 0):
            for j in range(len(p)):
                # for each path, we create two vectors containing 
                # the x1 and x2 coordinates of the vertices
                vs = contour.get_paths()[j].vertices 
                sizevs[j]=len(vs)
                x1=[]
                x2=[]
                for i in range(len(vs)):
                    xx1,xx2=vs[i]
                    x1.append(float(xx1))
                    x2.append(float(xx2))
                # these are the points we want to map back on the source plane. 
                # To do that we need to evaluate the deflection angle at their positions
                # using scipy.ndimage.interpolate.map_coordinates we perform a bi-linear interpolation
                a_1=map_coordinates(self.a1, [[x2],[x1]],order=1)
                a_2=map_coordinates(self.a2, [[x2],[x1]],order=1)
        
                # now we can make the mapping using the lens equation:
                no=self.pixel
                y1=(x1-a_1[0]-self.npix/2.)*no
                y2=(x2-a_2[0]-self.npix/2.)*no
        
                # plot the results!
                #ax.plot((np.array(x1)-npix/2.)*no*f,(np.array(x2)-npix/2.)*no*f,'-')
                ax.plot(y1,y2,lt,color=color,alpha=alpha)
            ax.set_xlabel(r'$\beta_1$',fontsize=fontsize)
            ax.set_ylabel(r'$\beta_2$',fontsize=fontsize)
                
    # display the time delay surface
    def show_surface(self,surf0,ax=None,minx=-25,miny=-25,vmax=2,rstride=1,
                     cstride=1,cmap=plt.get_cmap('Paired'),
                     linewidth=0, antialiased=False,alpha=0.2,fontsize=20,offz=0.0):
        
        surf=surf0+offz
        if ax==None:
            print ("specify the axes with 3d projection to display the surface")
        else:
            xa=np.arange(-self.npix/2, self.npix/2, 1)
            ya=np.arange(-self.npix/2, self.npix/2, 1)
        # I will show the contours levels projected in the x-y plane
            levels=np.linspace(np.amin(surf),np.amax(surf),40)

            minx=minx
            maxx=-minx

            miny=miny
            maxy=-miny

            pixel_size=self.size/(self.npix-1)
            X, Y = np.meshgrid(xa*pixel_size, ya*pixel_size)
            ax.plot_surface(X,Y,surf,vmax=vmax,rstride=rstride, cstride=cstride, cmap=cmap,
                       linewidth=linewidth, antialiased=antialiased,alpha=alpha)

            cset = ax.contour(X, Y, surf, zdir='z', 
                               offset=np.amin(surf)-20.0, cmap=cmap,levels=levels)
            deta=self.detA()
            cset = ax.contour(X, Y, deta, zdir='z', 
                               offset=np.amin(surf)-20.0, colors='black',levels=[0])            
            cset = ax.contour(X, Y, surf, zdir='x', offset=minx, cmap=cmap,levels=[0])
            cset = ax.contour(X, Y, surf, zdir='y', offset=maxy, cmap=cmap,levels=[0])
            ax.set_xlim3d(minx, maxx)
            ax.set_ylim3d(miny, maxy)
            ax.set_zlim3d(np.amin(surf)-20.0, 10)
            ax.set_xlabel(r'$\theta_1$',fontsize=fontsize)
            ax.set_ylabel(r'$\theta_2$',fontsize=fontsize)
            ax.set_aspect('equal')            

    # display the time delay contours
    def show_contours(self,surf0,ax=None,minx=-25,miny=-25,cmap=plt.get_cmap('Paired'),
                     linewidth=1,fontsize=20,nlevels=40,levmax=100,offz=0.0):
        if ax==None:
            print ("specify the axes to display the contours")
        else:
            minx=minx
            maxx=-minx
            miny=miny
            maxy=-miny
            surf=surf0-np.min(surf0)
            levels=np.linspace(np.min(surf),levmax,nlevels)
            ax.contour(surf, cmap=cmap,levels=levels,
                       linewidth=linewidth,
                       extent=[-self.size/2,self.size/2,-self.size/2,self.size/2])
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_xlabel(r'$\theta_1$',fontsize=fontsize)
            ax.set_ylabel(r'$\theta_2$',fontsize=fontsize)
            ax.set_aspect('equal')    

    '''
    Find the images of a source at (ys1,ys2) by mapping triangles on the lens plane into
    triangles in the source plane. Then search for the triangles which contain the source. 
    The image position is then computed by weighing with the distance from the vertices of the 
    triangle on the lens plane
    '''
    def find_images(self,y1,y2):
    

        pc=np.linspace(-self.size/2.0,self.size/2.0,self.npix)
        xk,yk=pc,pc#np.meshgrid(pc,pc)
        y1s=y1/2.0/np.amax(xk)*xk.size+xk.size/2.0
        y2s=y2/2.0/np.amax(xk)*xk.size+xk.size/2.0
        # ray tracing from the lens to the source plane
        xray=(xk-self.a1*self.pixel) # y1 coordinates on the source plane
        yray=np.transpose(yk-np.transpose(self.a2*self.pixel)) # y2 coordinates on the source plane
        xray=np.array(xray)/2.0/np.amax(xk)*xk.size+xk.size/2.0
        yray=np.array(yray)/2.0/np.amax(xk)*xk.size+xk.size/2.0
    
        # shift the maps by one pixel
        xray1=np.roll(xray,1,axis=1)
        xray2=np.roll(xray1,1,axis=0)
        xray3=np.roll(xray2,-1,axis=1)
        yray1=np.roll(yray,1,axis=1)
        yray2=np.roll(yray1,1,axis=0)
        yray3=np.roll(yray2,-1,axis=1)
    
        """
        for each pixel on the lens plane, build two triangle. By means of ray-tracing 
        these are mapped onto the source plane into other two triangles. Compute the 
        distances of the vertices of the triangles on the source plane from the source 
        and check using cross-products if the source is inside one of the two triangles
        """
        x1=y1s-xray
        y1=y2s-yray
    
        x2=y1s-xray1
        y2=y2s-yray1
    
        x3=y1s-xray2
        y3=y2s-yray2
        
        x4=y1s-xray3
        y4=y2s-yray3   
        
        prod12=x1*y2-x2*y1
        prod23=x2*y3-x3*y2
        prod31=x3*y1-x1*y3
        prod13=-prod31
        prod34=x3*y4-x4*y3
        prod41=x4*y1-x1*y4
    
        image=np.zeros(xray.shape)
        image[((np.sign(prod12) == np.sign(prod23)) & (np.sign(prod23) == np.sign(prod31)))]=1
        image[((np.sign(prod13) == np.sign(prod34)) & (np.sign(prod34) == np.sign(prod41)))]=2

        # first kind of images (first triangle)
        images1=np.argwhere(image==1)
        xi_images_=images1[:,1]
        yi_images_=images1[:,0]
        xi_images=xi_images_[(xi_images_>0) & (yi_images_>0)]
        yi_images=yi_images_[(xi_images_>0) & (yi_images_>0)]
    
        # compute the weights
        w=np.array([1./np.sqrt(x1[xi_images,yi_images]**2+y1[xi_images,yi_images]**2),
            1./np.sqrt(x2[xi_images,yi_images]**2+y2[xi_images,yi_images]**2),
            1./np.sqrt(x3[xi_images,yi_images]**2+y3[xi_images,yi_images]**2)])
        xif1,yif1=xi_images,yi_images#self.refineImagePositions(xi_images,yi_images,w,1)
    
        # second kind of images
        images1=np.argwhere(image==2)
        xi_images_=images1[:,1]
        yi_images_=images1[:,0]
        xi_images=xi_images_[(xi_images_>0) & (yi_images_>0)]
        yi_images=yi_images_[(xi_images_>0) & (yi_images_>0)]
    
        # compute the weights
        w=np.array([1./np.sqrt(x1[xi_images,yi_images]**2+y1[xi_images,yi_images]**2),
            1./np.sqrt(x3[xi_images,yi_images]**2+y3[xi_images,yi_images]**2),
            1./np.sqrt(x4[xi_images,yi_images]**2+y4[xi_images,yi_images]**2)])
        xif2,yif2=xi_images,yi_images#self.refineImagePositions(xi_images,yi_images,w,1) # 1 or 2 here?        
    
        xi=np.concatenate([xif1,xif2])
        yi=np.concatenate([yif1,yif2])
        xi=(xi-xk.size/2.0)*self.pixel
        yi=(yi-xk.size/2.0)*self.pixel
        return(xi,yi)

    def refineImagePositions(self,x,y,w,typ):
        if (typ==1):
            xp=np.array([x,x+1,x+1])
            yp=np.array([y,y,y+1])
        else:
            xp=np.array([x,x+1,x])
            yp=np.array([y,y+1,y+1])
        xi=np.zeros(x.size)
        yi=np.zeros(y.size)
        for i in range(x.size):
            xi[i]=(xp[:,i]/w[:,i]).sum()/(1./w[:,i]).sum()
            yi[i]=(yp[:,i]/w[:,i]).sum()/(1./w[:,i]).sum()
        return(xi,yi)

    def imageMagnification(self,xi1,xi2):
        if (self.computed_deta==False):
            self.deta=self.detA()
            self.computed_deta==True
        x1pix=(xi1+self.size/2.0)/self.pixel
        x2pix=(xi2+self.size/2.0)/self.pixel
        deta_ima = map_coordinates(self.deta,[x2pix,x1pix],order=5)
        return(deta_ima)

    def imageTimeDelay(self,xi1,xi2,y1,y2):
        td=self.t_delay_surf([y1,y2])
        x1pix=(xi1+self.size/2.0)/self.pixel
        x2pix=(xi2+self.size/2.0)/self.pixel
        td_ima = map_coordinates(td,[x2pix,x1pix],order=5)
        return(td_ima)


    def map_back(self,xi1,xi2):
        px=self.pixel#size/(self.df.npix-1)
        x1pix=(xi1+self.size/2.0)/px
        x2pix=(xi2+self.size/2.0)/px

        a1 = map_coordinates(self.a1,
                             [x2pix,x1pix],order=2)*px
        a2 = map_coordinates(self.a2,
                             [x2pix,x1pix],order=2)*px
        
        y1=(xi1-a1) # y1 coordinates on the source plane
        y2=(xi2-a2) # y2 coordinates on the source plane
        return(y1,y2)


# child class PSIE
class psie(gen_lens):
    def __init__(self,x0=0.0,y0=0.0,size=100.0,npix=200,**kwargs):
        
        if ('theta_c' in kwargs):
            self.theta_c=kwargs['theta_c']
        else:
            self.theta_c=0.0
            
        if ('ell' in kwargs):
            self.ell=kwargs['ell']
        else:
            self.ell=0.0
            
        if ('norm' in kwargs):
            self.norm=kwargs['norm']
        else: 
            self.norm=1.0

        if ('pa' in kwargs):
            self.pa=kwargs['pa']
        else: 
            self.pa=np.pi/2.0
 
         
        self.size=size
        self.npix=npix
        self.pixel=float(self.size)/float((self.npix-1))
        self.x0=x0/self.pixel+(self.npix-1) / 2
        self.y0=y0/self.pixel+(self.npix-1) / 2
        self.potential()
        self.computed_deta=False
        
        
    def potential(self):
        f=1.0-self.ell
        x = np.arange(0, self.npix, 1, float)
        y = x[:,np.newaxis]

        xx = np.cos(self.pa)*(x-self.x0)+np.sin(self.pa)*(y-self.y0)
        yy = -np.sin(self.pa)*(x-self.x0)+np.cos(self.pa)*(y-self.y0)
        r = np.sqrt((xx)**2+(yy/f)**2)       
        #x0 = y0 = (self.npix-1) / 2
        no=self.pixel**2
        self.pot_exists=True
        #pot=np.sqrt(((x-x0)*self.pixel)**2/(1-self.ell)
        #                 +((y-y0)*self.pixel)**2*(1-self.ell)
        #                 +self.theta_c**2)*self.norm
        #self.pot=pot#/no
        self.pot=np.sqrt(r*r*self.pixel**2+self.theta_c**2)*self.norm
        self.a2,self.a1=np.gradient(self.pot/self.pixel**2)
        self.a12,self.a11=np.gradient(self.a1)
        self.a22,self.a21=np.gradient(self.a2)
        



# child class deflector  
class deflector(gen_lens):

    
    # initialize the deflector using a surface density (covergence) map
    # the boolean variable pad indicates whether zero-padding is used or not
    
    '''   def __init__(self,filekappa,pad=False,npix=200,size=100):
        kappa,header=pyfits.getdata(filekappa,header=True)
        self.pixel_scale=header['CDELT2']*3600.0
        self.kappa=kappa
        self.nx=kappa.shape[0]
        self.ny=kappa.shape[1]
        self.pad=pad
        self.npix=npix
        self.size=size
        self.pixel=float(self.size)/float(self.npix-1)
        if (pad):
            self.kpad()
        self.potential()
    '''

    # initialize the deflector using a surface density (covergence) map
    # the boolean variable pad indicates whether zero-padding is used or not
    def __init__(self,kappa,pad,pixel,npix,size):
        self.kappa=kappa
        self.nx=kappa.shape[0]
        self.ny=kappa.shape[1]
        self.pad=pad
        self.pixel_scale=pixel
        self.npix=npix
        self.size=size
        self.pixel=float(self.size)/float(self.npix-1)
        if (self.pad):
            self.kpad()
        #self.kx,self.ky=self.kernel()
        self.potential()
        self.computed_deta=False
        #if (self.pad):
        #    self.pot=self.mapCrop(self.pot)
        #self.a2,self.a1=np.gradient(self.pot)
        #self.a12,self.a11=np.gradient(self.a1)
        #self.a22,self.a21=np.gradient(self.a2)
        
    # performs zero-padding
    def kpad(self):
        # add zeros around the original array
        def padwithzeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        # use the pad method from numpy.lib to add zeros (padwithzeros) in a
        # frame with thickness self.kappa.shape[0]
        self.kappa=np.lib.pad(self.kappa, self.kappa.shape[0], 
                              padwithzeros)
        
    # calculate the potential by solving the poisson equation
    def potential_from_kappa(self):
        # define an array of wavenumbers (two components k1,k2)
        k = np.array(np.meshgrid(fftengine.fftfreq(self.kappa.shape[0])\
                                 ,fftengine.fftfreq(self.kappa.shape[1])))
        pix=1 # pixel scale (now using pixel units)
        #Compute Laplace operator in Fourier space = -4*pi*k^2
        kk = k[0]**2 + k[1]**2
        kk[0,0] = 1.0
        #FFT of the convergence
        kappa_ft = fftengine.fftn(self.kappa)
        #compute the FT of the potential
        kappa_ft *= - pix**2 / (kk * (2.0*np.pi**2))
        kappa_ft[0,0] = 0.0
        potential=fftengine.ifftn(kappa_ft) #units should be rad**2
        if self.pad:
            pot=self.mapCrop(potential.real)
        return pot
    
    # returns the map of the gravitational time delay
    def potential(self):
        no=self.pixel
        x_ = np.linspace(0,self.npix-1,self.npix)
        y_ = np.linspace(0,self.npix-1,self.npix)
        x,y=np.meshgrid(x_,y_)
        potential=self.potential_from_kappa()
        x0 = y0 = potential.shape[0] / 2*self.pixel_scale-self.size/2.0
        x=(x0+x*no)/self.pixel_scale
        y=(y0+y*no)/self.pixel_scale
        self.pot_exists=True
        pot=map_coordinates(potential,[[y],[x]],order=5).reshape(int(self.npix),int(self.npix))
        self.pot=pot*self.pixel_scale**2/no/no
        self.a2,self.a1=np.gradient(self.pot)
        self.a12,self.a11=np.gradient(self.a1)
        self.a22,self.a21=np.gradient(self.a2)
        self.pot=pot*self.pixel_scale**2
        

    # crop the maps to remove zero-padded areas and get back to the original 
    # region.
    def mapCrop(self,mappa):
        xmin=int(self.kappa.shape[0]/2-self.nx/2)
        ymin=int(self.kappa.shape[1]/2-self.ny/2)
        print (xmin,ymin,type(mappa))
        xmax=int(xmin+self.nx)
        ymax=int(ymin+self.ny)
        mappa=mappa[xmin:xmax,ymin:ymax]
        return(mappa)


class deflector_from_file(deflector):

    # initialize the deflector using a surface density (covergence) map read from file
    # the boolean variable pad indicates whether zero-padding is used or not
    def __init__(self,filekappa,pad=False,npix=200,size=100):
        kappa,header=pyfits.getdata(filekappa,header=True)
        pixel=header['CDELT2']
        self.computed_deta=False
        deflector.__init__(self,kappa,pad,pixel,npix,size)


class deflector_from_map(deflector):

    # initialize the deflector using a surface density (covergence) map 
    # the boolean variable pad indicates whether zero-padding is used or not
    def __init__(self,kappa,pixel,pad=False,npix=200,size=100):
        self.computed_deta=False
        deflector.__init__(self,kappa,pad,pixel,npix,size)

class deflector_from_potential(deflector):

    # initialize the deflector using  map of the potential 
    def __init__(self,pot,npix,size):
        self.computed_deta=False
        self.pixel=size/(npix-1)
        self.pot=pot/self.pixel**2
        self.pot_exists=True 
        self.nx=pot.shape[0]
        self.ny=pot.shape[1]
        self.pixel_scale=self.pixel
        self.npix=npix
        self.size=size
        self.pixel=float(self.size)/float(self.npix-1)
        self.a2,self.a1=np.gradient(self.pot)
        self.a12,self.a11=np.gradient(self.a1)
        self.a22,self.a21=np.gradient(self.a2)
        self.kappa=0.5*(self.a11+self.a22)
        self.pot=pot*self.pixel**2



class sersic(object):
    
    def __init__(self,size,N,gl=None,**kwargs):
        
        if ('n' in kwargs):
            self.n=kwargs['n']
        else:
            self.n=4
            
        if ('re' in kwargs):
            self.re=kwargs['re']
        else:
            self.re=5.0
            
        if ('q' in kwargs):
            self.q=kwargs['q']
        else:
            self.q=1.0
            
        if ('pa' in kwargs):
            self.pa=kwargs['pa']
        else:
            self.pa=0.0

        if ('ys1' in kwargs):
            self.ys1=kwargs['ys1']
        else:
            self.ys1=0.0
            
        if ('ys2' in kwargs):
            self.ys2=kwargs['ys2']
        else:
            self.ys2=0.0
            
        self.N=N
        self.size=float(size)
        self.df=gl
        
        # define the pixel coordinates 
        pc=np.linspace(-self.size/2.0,self.size/2.0,self.N)
        self.x1, self.x2 = np.meshgrid(pc,pc)
        if self.df != None:
            y1,y2 = self.ray_trace()
        else:
            y1,y2 = self.x1,self.x2
                    
        self.image=self.brightness(y1,y2)

    def ray_trace(self):
        px=self.df.pixel#size/(self.df.npix-1)
        x1pix=(self.x1+self.df.size/2.0)/px
        x2pix=(self.x2+self.df.size/2.0)/px

        a1 = map_coordinates(self.df.a1,
                             [x2pix,x1pix],order=2)*px
        a2 = map_coordinates(self.df.a2,
                             [x2pix,x1pix],order=2)*px
        
        y1=(self.x1-a1) # y1 coordinates on the source plane
        y2=(self.x2-a2) # y2 coordinates on the source plane
        return(y1,y2)
        
        
    def brightness(self,y1,y2):
        x = np.cos(self.pa)*(y1-self.ys1)+np.sin(self.pa)*(y2-self.ys2)
        y = -np.sin(self.pa)*(y1-self.ys1)+np.cos(self.pa)*(y2-self.ys2)
        r = np.sqrt(((x)/self.q)**2+(y)**2)
        
        # profile
        bn = 1.992*self.n - 0.3271
        brightness = np.exp(-bn*((r/self.re)**(1.0/self.n)-1.0))
        return(brightness)


class postage_stamp(object):
    
    def __init__(self,image,image_size,size_raygrid,Nray,gl=None,**kwargs):
        
        if ('ys1' in kwargs):
            self.ys1=kwargs['ys1']
        else:
            self.ys1=0.0
            
        if ('ys2' in kwargs):
            self.ys2=kwargs['ys2']
        else:
            self.ys2=0.0

        if ('flux' in kwargs):
            self.flux=kwargs['flux']
        else:
            self.flux=0.0

        if ('pa' in kwargs):
            self.pa=kwargs['pa']
        else:
            self.pa=0.0
        
        self.image=image
        self.image_size=image_size
        self.pixel=image_size/(image.shape[0]-1)
        self.size=float(size_raygrid)
        self.N=Nray
        self.df=gl
        
        # define the pixel coordinates 
        pc=np.linspace(-self.size/2.0,self.size/2.0,self.N)
        self.x1, self.x2 = np.meshgrid(pc,pc)
        if self.df != None:
            y1,y2 = self.ray_trace()
        else:
            y1,y2 = self.x1,self.x2
                    
        self.image=self.brightness(y1,y2)


    def ray_trace(self):
        px=self.df.pixel#size/(self.df.npix-1)
        x1pix=(self.x1+self.df.size/2.0)/px
        x2pix=(self.x2+self.df.size/2.0)/px

        a1 = map_coordinates(self.df.a1,
                             [x2pix,x1pix],order=2)*px
        a2 = map_coordinates(self.df.a2,
                             [x2pix,x1pix],order=2)*px
        
        y1=(self.x1-a1) # y1 coordinates on the source plane
        y2=(self.x2-a2) # y2 coordinates on the source plane
        return(y1,y2)
        
        
    def brightness(self,y1,y2):
        x = np.cos(self.pa)*(y1-self.ys1)+np.sin(self.pa)*(y2-self.ys2)
        y = -np.sin(self.pa)*(y1-self.ys1)+np.cos(self.pa)*(y2-self.ys2)
        x1pix=(x+self.image_size/2.0)/self.pixel
        x2pix=(y+self.image_size/2.0)/self.pixel
        print (np.min(x1pix),np.max(x1pix))   
        brightness = map_coordinates(self.image,[x2pix,x1pix],order=5)
        return(brightness)
