/*****************************************************************************
Author       : Yongxu Lu
Last modified: 2018-12.29
Email        : Luyx@cumtb.cdu.cn
Organization : China University of Mining and Technology, Beijing
*****************************************************************************/

#include <math.h>
#include <fstream>
#include <vector>
#include <memory.h>
#include <string>
#include <sstream>
#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

#define MAX(a,b) (a>b)?a:b
#define MIN(a,b) (a<b)?a:b
#define PI 3.1415926
/*** round */
//#define NINT(a) (a)>0?((int)((a)+0.5)):((int)((a)-0.5))
#define NINT(a) (int)(a+0.5)
/*** 3D index */
#define  I3(iz,ix,iy)   ((iz)*nx*ny             + (ix)*ny       + (iy))
#define I33(iz,ix,iy)   ((iz)*nxpadded*nypadded + (ix)*nypadded + (iy))
#define NUM_THREADS 28


/*****************************************************************************
-----------------------------Global parameters--------------------------------
*****************************************************************************/
/*** set variables for 3D-FDTD ***/
unsigned long I33IZX=0;
unsigned long I33IZXN=0;
/* accuracy */
int torder = 2; /* order of temporal accuracy */
int sorder = 4; /* order of spacial accuracy */

int nx=150; /* number of nods in x/y/z direction */
int ny=150;
int nz=150;

float dx=0.01;          /* spacial interval in each direction */
float dy=0.01;
float dz=0.01; 

float dt=1e-11;         /* temporal interval(s) */
float lt=3e-9;         /* record time(s) */

float sx[]={0.75};       /* array of position of sources */
float sy[]={0.75};
float sz[]={0.75};       /* location of sources in x/y/z direction(m) */
int stype=1;            /* add source to: 1=Ex; 2=Ey; 3=Ez ; 4=Ex=Ey=Ez */
int wtype=2;            /* wavelet type(1=gaussian;2=derivative of gaussian;3=ricker) */
float favg=1e9;         /* source average frequency (HZ) */
float ts=2e-9;          /* source duration(s) */
int sr=3;               /* source radius for RSG, source center at [isz,isx,isy] */

int abct=1;             /* absorbing boundary condition type(0=none;1=cpml;) */
int abcw=30;            /* width(in nods) of absorbing boundary */
float R=0.0001;         /* reflectivity of P wave of PML */

float hrz=0.0;          /* z coordinate of horizontal line of seismogram(m) */
int snap=1;               /* save snapshots?
				          (0=no; 1=save E; 2=save H; 3=save all) */
float sntime[]={0.5e-9,1e-9,1.5e-9,2e-9,2.5e-9,3e-9,4e-9,5e-9}; /* array contain which time to save snapshots(s) */

string epsr_xx_fn="epsr_xx.model"; /* model file */
string epsr_xy_fn="epsr_xy.model";
string epsr_xz_fn="epsr_xz.model";
string epsr_yy_fn="epsr_yy.model";
string epsr_yz_fn="epsr_yz.model";
string epsr_zz_fn="epsr_zz.model";

string sig_xx_fn="sig_xx.model";
string sig_xy_fn="sig_xy.model";
string sig_xz_fn="sig_xz.model";
string sig_yy_fn="sig_yy.model";
string sig_yz_fn="sig_yz.model";
string sig_zz_fn="sig_zz.model";

const float u0=4*PI*1e-7;
const float eps0 = 8.854187817e-12;

bool verbose=1; 

/***  global variables for 3D-FDTD ***/
int nxpadded,nypadded,nzpadded; /* number of nods after padded */
float dtu0;  /* dt/u0 */
int iter;   /* iteration counter */
float t;    /* propagation time */
int nt;     /* number of time step */
int sohalf; /* half of order of spacial accuracy */

/* model parameters */
float *eps_xx=NULL,*eps_xy=NULL,*eps_xz=NULL; /* note that size of these arrays is nxpadded*nypadded*nzpadded */
float              *eps_yy=NULL,*eps_yz=NULL;
float                           *eps_zz=NULL;

float *sig_xx=NULL,*sig_xy=NULL,*sig_xz=NULL;
float              *sig_yy=NULL,*sig_yz=NULL;
float                           *sig_zz=NULL;

float *Pxx=NULL,*Pxy=NULL,*Pxz=NULL;
float           *Pyy=NULL,*Pyz=NULL;
float                     *Pzz=NULL;

float *Qxx=NULL,*Qxy=NULL,*Qxz=NULL;
float           *Qyy=NULL,*Qyz=NULL;
float                     *Qzz=NULL;

float epsr_max;

MatrixXf tmp_eps(3,3),tmp_sig(3,3),tmp_P(3,3),tmp_Q(3,3);
 
/* source */
int nsou;            /* number of sources */
int isou;
float *source;
int ns;              /* number of samples of source function */
int *isx,*isy,*isz;  /* index of grid points of each source */
string source_fn="source.data";    /* source file */
FILE *source_fp;

/* wave field */
float *Hx,*Hy,*Hz;
float *Ex,*Ey,*Ez; /* E at time n */
float *Ex2,*Ey2,*Ez2; /* E at time n+1 */

float *Ex_snap,*Ey_snap,*Ez_snap;

/* auxiliary variables */
float dHxy,dHxz,dHyx,dHyz,dHzx,dHzy;
float dExy,dExz,dEyx,dEyz,dEzx,dEzy;

float dHx1,dHx2,dHx3,dHx4;
float dHy1,dHy2,dHy3,dHy4;
float dHz1,dHz2,dHz3,dHz4;

float dEx1,dEx2,dEx3,dEx4;
float dEy1,dEy2,dEy3,dEy4;
float dEz1,dEz2,dEz3,dEz4;

/* horizontal seismic record */
string hEx_fn="hEx.data"; /* file name */
string hEy_fn="hEy.data";
string hEz_fn="hEz.data";
FILE *hEx_fp;
FILE *hEy_fp;
FILE *hEz_fp;
int ihrz; /* horizontal receiver line */

/* snap */
int nsnap;
int *isnap;
stringstream snap_fn_temp; /* file name */
string snap_fn;
FILE *snap_Ex_fp;
FILE *snap_Ey_fp;
FILE *snap_Ez_fp;

/* coefficient for spacial accuracy */
float *csg; /* for staggered grid */

/* boundary */
int *wbc; /* pad boundary width. 
			 In a 3D case, index 0,1,2,3,4,5 for top,bottom,left,right,back,front respectively */
int *cb;  /* calculation boundary index. same as wbc */
int *mb;  /* model boundary index. same as wbc */

/* CPML */
float npower,mpower;
float kappa_max;
float alpha_max;
float *df, *df_h;
float alpha, alpha_h;
float *kappa, *kappa_h;
float *b, *b_h;
float *a, *a_h;

float *psi_Exy,*psi_Exz;
float *psi_Eyx,*psi_Eyz;
float *psi_Ezx,*psi_Ezy;

float *psi_Hxy,*psi_Hxz;
float *psi_Hyx,*psi_Hyz;
float *psi_Hzx,*psi_Hzy;

int ipsi_cpml, iabk_cpml; /* index of psi and a,b,kappa, respectively */

/* other variables */
float energy;

/*****************************************************************************
-----------------------------Main codes--------------------------------
*****************************************************************************/

/* return length of an array */
template <class T>
int getArrayLen(T& array)
{
return (sizeof(array) / sizeof(array[0]));
}

template <typename TYPE>
TYPE pad_3d_array(int nz,int nx,int ny,int *pw,TYPE in_arr,TYPE out_arr)
/************************************************************
Inputs:
	nz*nx*ny		size of the 
	
	in_arr[nz][nx][ny]
	out_arr[nz+pw[0]+pw[1]][nx+pw[2]+pw[3]][ny+pw[4]+pw[5]]
************************************************************/
{
	int iz,ix,iy;
	int nzpadded,nxpadded,nypadded;
	
	nzpadded = nz+pw[0]+pw[1];
	nxpadded = nx+pw[2]+pw[3];
	nypadded = ny+pw[4]+pw[5];
	
	memset((void *) out_arr, 0, nzpadded*nxpadded*nypadded*sizeof(out_arr[0])); 

	/*** one body ***/
	
	/* middle body */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],ix-pw[2],iy-pw[4])];
	
	/*** six surfaces ***/
	
	/* upper surface */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,ix-pw[2],iy-pw[4])];
	
	/* lower surface */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,ix-pw[2],iy-pw[4])];	
	
	/* left surface */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],0,iy-pw[4])];
			
	/* right surface */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],nx-1,iy-pw[4])];

	/* back surface */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],ix-pw[2],0)];
	/* front surface */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],ix-pw[2],ny-1)];
	
	/*** twelve edges ***/
	
	/* upper left edge */
	for (iz=0;iz<pw[0];iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,0,iy-pw[4])];
			
	/* upper right edge */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,nx-1,iy-pw[4])];
	
	/* upper back edge */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,ix-pw[2],0)];
	
	/* upper front edge */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,ix-pw[2],ny-1)];
	
	/* left back edge */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],0,0)];
				
	/* left front edge */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],0,ny-1)];
				
	/* right back edge */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],nx-1,0)];
				
	/* right front edge */
	for (iz=pw[0];iz<pw[0]+nz;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(iz-pw[0],nx-1,ny-1)];
	
	/* lower left edge */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,0,iy-pw[4])];

	/* lower right edge */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4];iy<pw[4]+ny;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,nx-1,iy-pw[4])];
	
	/* lower back edge */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,ix-pw[2],0)];
	
	/* lower front edge */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2];ix<pw[2]+nx;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,ix-pw[2],ny-1)];
	
	/*** eight corners ***/	
	
	/* upper left back corner */
	for (iz=0;iz<pw[0];iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,0,0)];
	
	/* upper left front corner */
	for (iz=0;iz<pw[0];iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,0,ny-1)];
	
	/* upper right back corner */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,nx-1,0)];
				
	/* upper right front corner */
	for (iz=0;iz<pw[0];iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(0,nx-1,ny-1)];
	
	/* lower left back corner */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,0,0)];
	
	/* lower left front corner */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=0;ix<pw[2];ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,0,ny-1)];
	
	/* lower right back corner */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=0;iy<pw[4];iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,nx-1,0)];
	
	/* lower right front corner */
	for (iz=pw[0]+nz;iz<nzpadded;iz++)
		for (ix=pw[2]+nx;ix<nxpadded;ix++)
			for (iy=pw[4]+ny;iy<nypadded;iy++)
				out_arr[I33(iz,ix,iy)]=in_arr[I3(nz-1,nx-1,ny-1)];
				
	return(out_arr);
}

float * read_3d_single(float *b, string bn)
/****************************************************** 
read single model parameter file 
 a : model parameter matrix
 b : padded model parameter matrix
bn : name of b
******************************************************/
{
	FILE *afp;
	float *a;
	int ix,iy,iz;
	a = new float[nz*nx*ny]; /* allocate space */
	/* open model file */
	if((afp=fopen(bn.data(),"rb"))==NULL) 
		printf("Cannot open file : %s\n",bn.data());
	/* read model */
	for(iz=0;iz<nz;iz++)
		for(ix=0;ix<nx;ix++)
			for(iy=0;iy<ny;iy++) {
				if(fread(&a[I3(iz,ix,iy)],sizeof(float),1,afp)!=1)
					printf("Cannot read or write file : %s\n",bn.data());
			}
	fclose(afp);
	/* read model */
	/* pad model */
	b = new float[nzpadded*nxpadded*nypadded];
	b = pad_3d_array(nz,nx,ny,wbc,a,b);
	delete [] a;
	return(b);
}

void calc_df_rsg_cpml()
/******************************************************************************************

******************************************************************************************/
{
	/*** set some values */
	npower=4.0; /* can be set as 2,3,4 */
	mpower=1.0; /* can be set as 2 or 3 (Martin et.al., 2009, CMES) */
	kappa_max=5.0; /* Martin et al. 2008(2009) */
	alpha_max=0; 
	
	/*** alloc space */
	kappa   = new float[abcw];
	kappa_h = new float[abcw];
	
	b   = new float[abcw];
	b_h = new float[abcw];
	
	a   = new float[abcw];
	a_h = new float[abcw];
	
	/*** calculation */
	float tv=0.0;
	float dfmax=0.0,df=0.0,df_h=0.0;
	float pmlin=0.0,pmlin_h=0.0; /* normalized distance to PML boundary */
	int i;
	
	dfmax = (npower+1.0)/(150*PI*sqrt(epsr_max)*dx);

	for(i=0;i<abcw;i++) {
		
		pmlin   = (i+1.0)/abcw;
		pmlin_h = (i+0.5)/abcw;
		
		df   = dfmax*pow(pmlin,npower);
		df_h = dfmax*pow(pmlin_h,npower);
		
		kappa[i]   = 1+(kappa_max-1)*pow(pmlin,npower);
		kappa_h[i] = 1+(kappa_max-1)*pow(pmlin_h,npower);
		
		alpha   = alpha_max * (1-pow(pmlin,mpower))+0.1*alpha_max; /* add "+0.1*alpha_max" according to code “SEISMIC_CPML_1.2” */
		alpha_h = alpha_max * (1-pow(pmlin_h,mpower))+0.1*alpha_max;
		
		b[i]   = exp(-(df/kappa[i]+alpha)*dt/eps0);
		b_h[i] = exp(-(df_h/kappa_h[i]+alpha_h)*dt/eps0);
		
		a[i] = df*(b[i]-1)/(kappa[i]*(df+kappa[i]*alpha));
		a_h[i] = df_h*(b_h[i]-1)/(kappa_h[i]*(df_h+kappa_h[i]*alpha_h));
		
	}
}

int main(){

	int ix,iy,iz; /* loop counters */
	int in; 
	int i,j;
	
	/* initial value of the maximum number of threads in a team for OpenMP */
	omp_set_num_threads(NUM_THREADS); 
	/* get start time */
	double start_time,end_time;
	start_time = omp_get_wtime();
	
	/********************** set model width **********************/
	sohalf = sorder/2;
	
	wbc = new int[6]; 
	
	for (i=0;i<6;i++) 
		wbc[i]=abcw+sohalf;
		
	nzpadded = nz+wbc[0]+wbc[1];
	nxpadded = nx+wbc[2]+wbc[3];
	nypadded = ny+wbc[4]+wbc[5];
	
	cb = new int[6]; 
	cb[0] = cb[2] = cb[4] = sohalf;
	cb[1] = nzpadded-sohalf-1;
	cb[3] = nxpadded-sohalf-1;
	cb[5] = nypadded-sohalf-1;
			
	mb = new int[6];  
	mb[0] = wbc[0];
	mb[1] = wbc[0]+nz-1;
	mb[2] = wbc[2];
	mb[3] = wbc[2]+nx-1;
	mb[4] = wbc[4];
	mb[5] = wbc[4]+ny-1;
	
	/********************** read in model **********************/
	
	eps_xx=read_3d_single(eps_xx,epsr_xx_fn);
	eps_xy=read_3d_single(eps_xy,epsr_xy_fn);
	eps_xz=read_3d_single(eps_xz,epsr_xz_fn);
	eps_yy=read_3d_single(eps_yy,epsr_yy_fn);
	eps_yz=read_3d_single(eps_yz,epsr_yz_fn);
	eps_zz=read_3d_single(eps_zz,epsr_zz_fn);
	
	sig_xx=read_3d_single(sig_xx,sig_xx_fn);
	sig_xy=read_3d_single(sig_xy,sig_xy_fn);
	sig_xz=read_3d_single(sig_xz,sig_xz_fn);
	sig_yy=read_3d_single(sig_yy,sig_yy_fn);
	sig_yz=read_3d_single(sig_yz,sig_yz_fn);
	sig_zz=read_3d_single(sig_zz,sig_zz_fn);
	
	epsr_max = eps_xx[0];
	for(i=0;i<nxpadded*nypadded*nzpadded;i++){
		epsr_max = MAX(epsr_max,eps_xx[i]);
		epsr_max = MAX(epsr_max,eps_xy[i]);
		epsr_max = MAX(epsr_max,eps_xz[i]);
		epsr_max = MAX(epsr_max,eps_yy[i]);
		epsr_max = MAX(epsr_max,eps_yz[i]);
		epsr_max = MAX(epsr_max,eps_zz[i]);
	}
	
	for(i=0;i<nxpadded*nypadded*nzpadded;i++) {
	    /* relative to real dielectric permittivity */
		eps_xx[i]=eps_xx[i]*eps0;
		eps_xy[i]=eps_xy[i]*eps0;
		eps_xz[i]=eps_xz[i]*eps0;
		eps_yy[i]=eps_yy[i]*eps0;
		eps_yz[i]=eps_yz[i]*eps0;
		eps_zz[i]=eps_zz[i]*eps0;
	}
	
	dtu0 = dt/u0;
	
	/* eps, sig to P, Q */
	Pxx = new float[nzpadded*nxpadded*nypadded];
	Pxy = new float[nzpadded*nxpadded*nypadded];
	Pxz = new float[nzpadded*nxpadded*nypadded];
	Pyy = new float[nzpadded*nxpadded*nypadded];
	Pyz = new float[nzpadded*nxpadded*nypadded];
	Pzz = new float[nzpadded*nxpadded*nypadded];
	
	Qxx = new float[nzpadded*nxpadded*nypadded];
	Qxy = new float[nzpadded*nxpadded*nypadded];
	Qxz = new float[nzpadded*nxpadded*nypadded];
	Qyy = new float[nzpadded*nxpadded*nypadded];
	Qyz = new float[nzpadded*nxpadded*nypadded];
	Qzz = new float[nzpadded*nxpadded*nypadded];
	
	for(iz=0;iz<nzpadded;iz++) {
		for(ix=0;ix<nxpadded;ix++){
			for(iy=0;iy<nypadded;iy++){
				I33IZX = I33(iz,ix,iy);
				
				tmp_eps(0,0)=eps_xx[I33IZX];tmp_eps(0,1)=eps_xy[I33IZX];tmp_eps(0,2)=eps_xz[I33IZX];
				tmp_eps(1,0)=eps_xy[I33IZX];tmp_eps(1,1)=eps_yy[I33IZX];tmp_eps(1,2)=eps_yz[I33IZX];
				tmp_eps(2,0)=eps_xz[I33IZX];tmp_eps(2,1)=eps_yz[I33IZX];tmp_eps(2,2)=eps_zz[I33IZX];
				
				tmp_sig(0,0)=sig_xx[I33IZX];tmp_sig(0,1)=sig_xy[I33IZX];tmp_sig(0,2)=sig_xz[I33IZX];
				tmp_sig(1,0)=sig_xy[I33IZX];tmp_sig(1,1)=sig_yy[I33IZX];tmp_sig(1,2)=sig_yz[I33IZX];
				tmp_sig(2,0)=sig_xz[I33IZX];tmp_sig(2,1)=sig_yz[I33IZX];tmp_sig(2,2)=sig_zz[I33IZX];
				
				tmp_Q = (tmp_eps/dt+tmp_sig/2);
				tmp_Q = tmp_Q.inverse();
				tmp_P = tmp_Q*(tmp_eps/dt-tmp_sig/2);
				
				Pxx[I33IZX]=tmp_P(0,0);Pxy[I33IZX]=tmp_P(0,1);Pxz[I33IZX]=tmp_P(0,2);
				Pyy[I33IZX]=tmp_P(1,1);Pyz[I33IZX]=tmp_P(1,2);Pzz[I33IZX]=tmp_P(2,2);
				
				Qxx[I33IZX]=tmp_Q(0,0);Qxy[I33IZX]=tmp_Q(0,1);Qxz[I33IZX]=tmp_Q(0,2);
				Qyy[I33IZX]=tmp_Q(1,1);Qyz[I33IZX]=tmp_Q(1,2);Qzz[I33IZX]=tmp_Q(2,2);
			}
		}
	}
	delete [] eps_xx; delete [] eps_xy; delete [] eps_xz;
	delete [] eps_yy; delete [] eps_yz; delete [] eps_zz;
	delete [] sig_xx; delete [] sig_xy; delete [] sig_xz;
	delete [] sig_yy; delete [] sig_yz; delete [] sig_zz;
	
	/********************** set wave fields **********************/
	/* allocate space */
	Ex = new float[nzpadded*nxpadded*nypadded];
	Ey = new float[nzpadded*nxpadded*nypadded];
	Ez = new float[nzpadded*nxpadded*nypadded];
	
	Ex2 = new float[nzpadded*nxpadded*nypadded];
	Ey2 = new float[nzpadded*nxpadded*nypadded];
	Ez2 = new float[nzpadded*nxpadded*nypadded];
	
	Hx = new float[nzpadded*nxpadded*nypadded];
	Hy = new float[nzpadded*nxpadded*nypadded];
	Hz = new float[nzpadded*nxpadded*nypadded];
	
	/********************** set CPML parameters **********************/
	calc_df_rsg_cpml();
	
	psi_Exy = new float[nzpadded*nxpadded*nypadded];
	psi_Exz = new float[nzpadded*nxpadded*nypadded];
	psi_Eyx = new float[nzpadded*nxpadded*nypadded];
	psi_Eyz = new float[nzpadded*nxpadded*nypadded];
	psi_Ezx = new float[nzpadded*nxpadded*nypadded];
	psi_Ezy = new float[nzpadded*nxpadded*nypadded];
	
	psi_Hxy = new float[nzpadded*nxpadded*nypadded];
	psi_Hxz = new float[nzpadded*nxpadded*nypadded];
	psi_Hyx = new float[nzpadded*nxpadded*nypadded];
	psi_Hyz = new float[nzpadded*nxpadded*nypadded];
	psi_Hzx = new float[nzpadded*nxpadded*nypadded];
	psi_Hzy = new float[nzpadded*nxpadded*nypadded];
	
	/********************** set source function **********************/
	/* calculate nt */
	int nt=0;
	nt = NINT(lt/dt);
	
	/* set source position in grids */
	nsou = getArrayLen(sx);
	isx = new int[nsou];
	isy = new int[nsou];
	isz = new int[nsou];
	for(isou=0;isou<nsou;isou++){
		isz[isou] = NINT(sz[isou]/dz)+wbc[0];
		isx[isou] = NINT(sx[isou]/dx)+wbc[2];
		isy[isou] = NINT(sy[isou]/dy)+wbc[4];
	}
	
	/* get source function */
	ns = NINT(ts/dt);
	source = new float[nt];
	for(i=0;i<nt;i++) source[i]=0.0; /* initialize */
	t=0.0;
	float pi2,x,xx;
	pi2 = PI*PI;
	switch (wtype) {
		case 1 : /* gaussian */
			for(i=0;i<ns;i++){
				x=favg*(t-ts/2);xx=x*x;
				source[i]=(-1/(2*pi2*favg*favg))*exp(-pi2*xx);
				t = t+dt;
			}
			break;
		case 2 : /* derivative of gaussian */
			for(i=0;i<ns;i++){
				x=favg*(t-ts/2);xx=x*x;
				source[i]=(t-ts/2)*exp(-pi2*xx);
				t = t+dt;
			}
			break;
		case 3 : /* ricker */
			for(i=0;i<ns;i++){
				x=favg*(t-ts/2);xx=x*x;
				source[i]=(1-2*pi2*(xx))*exp(-pi2*xx);
				t = t+dt;
			}
			break;
		default :
			break;
	}
	
	/* save source file */
	source_fp=fopen(source_fn.data(),"wb+");
	fwrite(source,sizeof(source[0]),nt,source_fp);
	fclose(source_fp);
	
	/********************** recording preparation **********************/
	hEx_fp=fopen(hEx_fn.data(),"wb+");
	hEy_fp=fopen(hEy_fn.data(),"wb+");
	hEz_fp=fopen(hEz_fn.data(),"wb+");
	ihrz = NINT(hrz/dz)+wbc[0]; 
	
	/********************** snapshot preparation **********************/
	nsnap = getArrayLen(sntime);
	isnap = new int[nsnap];
	for(i=0;i<nsnap;i++)
		isnap[i]=NINT(sntime[i]/dt);
	
	/********************** fit current spacial accuracy **********************/
	/* Set coefficient of spatial fd operator for SSG/RSG */
	csg=new float[sohalf]; 
	switch (sorder) {
		case 2 :
			csg[0]=1;
			break;
		case 4 :
			csg[0]=1.125;csg[1]=-4.16667e-2;
			break;
		case 6 :
			csg[0]=1.17187;csg[1]=-6.51042e-2;csg[2]=4.6875e-3;
			break;
		case 8 :
			csg[0]=1.19629;csg[1]=-7.97526e-2;csg[2]=9.57031e-3;csg[3]=-6.97545e-4;
			break;
		case 10:
			csg[0]=1.21124;csg[1]=-8.97217e-2;csg[2]=1.38428e-2;csg[3]=-1.76566e-3;csg[4]=1.1868e-4;
			break;
	}
	
	/********************** begin propagation **********************/
	
	if(verbose) printf("...Sou...Iter...Energy...Time(s)\n");
	
	for (isou=0;isou<nsou;isou++) { /* loop over sources */
	
		/* zero arrays */
		for(iz=0;iz<nzpadded;iz++) {
			for(ix=0;ix<nxpadded;ix++){
				for(iy=0;iy<nypadded;iy++){
					I33IZX = I33(iz,ix,iy);
					
					Ex[I33IZX]=0.0; Ex2[I33IZX]=0.0; 
					Ey[I33IZX]=0.0; Ey2[I33IZX]=0.0;
					Ez[I33IZX]=0.0; Ez2[I33IZX]=0.0;
					Hx[I33IZX]=0.0; 
					Hy[I33IZX]=0.0;
					Hz[I33IZX]=0.0; 
					
					psi_Exy[I33IZX]=0.0; psi_Exz[I33IZX]=0.0; 
					psi_Eyx[I33IZX]=0.0; psi_Eyz[I33IZX]=0.0;
					psi_Ezx[I33IZX]=0.0; psi_Ezy[I33IZX]=0.0;
					
					psi_Hxy[I33IZX]=0.0; psi_Hxz[I33IZX]=0.0; 
					psi_Hyx[I33IZX]=0.0; psi_Hyz[I33IZX]=0.0;
					psi_Hzx[I33IZX]=0.0; psi_Hzy[I33IZX]=0.0;
				}
			}
		}
		
		/* loop over time */
		t = 0.0;
		for (iter=0; iter<nt;) {
			/* add source to E */
			if(stype==1 || stype==4) {
				// for (iz=-sr;iz<=sr-1;iz++) {
					// for (ix=-sr;ix<=sr-1;ix++) {
						// for (iy=-sr;iy<=sr-1;iy++) {
							// Ex[I33(isz[isou]+iz,isx[isou]+ix,isy[isou]+iy)] += source[iter];
						// }
					// }
				// }
				for (iz=0;iz<=sr;iz++) {
					for (ix=0;ix<=sr;ix++) {
						for (iy=0;iy<=sr;iy++) {
							Ex[I33(isz[isou]+iz,isx[isou]+ix,isy[isou]+iy)] += source[iter];
						}
					}
				}
			}
			if(stype==2 || stype==4) {
				for (iz=-sr;iz<=sr-1;iz++) {
					for (ix=-sr;ix<=sr-1;ix++) {
						for (iy=-sr;iy<=sr-1;iy++) {
							Ey[I33(isz[isou]+iz,isx[isou]+ix,isy[isou]+iy)] += source[iter];
						}
					}
				}
			}
			if(stype==3 || stype==4) {
				for (iz=-sr;iz<=sr-1;iz++) {
					for (ix=-sr;ix<=sr-1;ix++) {
						for (iy=-sr;iy<=sr-1;iy++) {
							Ez[I33(isz[isou]+iz,isx[isou]+ix,isy[isou]+iy)] += source[iter];
						}
					}
				}
			}
			
			/* update H at half-time grids */
			dExy=dExz=dEyx=dEyz=dEzx=dEzy=0.0;
			#pragma omp parallel for default(shared) \
				private(ix,iy,iz,in,iabk_cpml,\
						dEx1,dEx2,dEx3,dEx4,\
						dEy1,dEy2,dEy3,dEy4,\
						dEz1,dEz2,dEz3,dEz4,I33IZX,\
						dEyx,dEzx,dExy,dEzy,dExz,dEyz)
			for (iz=cb[0];iz<=cb[1];iz++) {
				for (ix=cb[2];ix<=cb[3];ix++) {
					for (iy=cb[4];iy<=cb[5];iy++) {
					
						I33IZX = I33(iz,ix,iy);
						
						dEx1=dEx2=dEx3=dEx4=0.0;
						dEy1=dEy2=dEy3=dEy4=0.0;
						dEz1=dEz2=dEz3=dEz4=0.0;
						
						for (in=0;in<sohalf;in++) {
						
							dEx1 += csg[in] * ( Ex[I33(iz+1+in,ix+1+in,iy+1+in)]-Ex[I33(iz-in,ix-in,iy-in)] );
							dEy1 += csg[in] * ( Ey[I33(iz+1+in,ix+1+in,iy+1+in)]-Ey[I33(iz-in,ix-in,iy-in)] );
							dEz1 += csg[in] * ( Ez[I33(iz+1+in,ix+1+in,iy+1+in)]-Ez[I33(iz-in,ix-in,iy-in)] );
							
							dEx2 += csg[in] * ( Ex[I33(iz-in,ix+1+in,iy+1+in)]-Ex[I33(iz+1+in,ix-in,iy-in)] );
							dEy2 += csg[in] * ( Ey[I33(iz-in,ix+1+in,iy+1+in)]-Ey[I33(iz+1+in,ix-in,iy-in)] );
							dEz2 += csg[in] * ( Ez[I33(iz-in,ix+1+in,iy+1+in)]-Ez[I33(iz+1+in,ix-in,iy-in)] );
							
							dEx3 += csg[in] * ( Ex[I33(iz+1+in,ix+1+in,iy-in)]-Ex[I33(iz-in,ix-in,iy+1+in)] );
							dEy3 += csg[in] * ( Ey[I33(iz+1+in,ix+1+in,iy-in)]-Ey[I33(iz-in,ix-in,iy+1+in)] );
							dEz3 += csg[in] * ( Ez[I33(iz+1+in,ix+1+in,iy-in)]-Ez[I33(iz-in,ix-in,iy+1+in)] );
							
							dEx4 += csg[in] * ( Ex[I33(iz-in,ix+1+in,iy-in)]-Ex[I33(iz+1+in,ix-in,iy+1+in)] );
							dEy4 += csg[in] * ( Ey[I33(iz-in,ix+1+in,iy-in)]-Ey[I33(iz+1+in,ix-in,iy+1+in)] );
							dEz4 += csg[in] * ( Ez[I33(iz-in,ix+1+in,iy-in)]-Ez[I33(iz+1+in,ix-in,iy+1+in)] );
							
						}
						
						dEyx = 0.25*(dEy1+dEy2+dEy3+dEy4)/dx;
						dEzx = 0.25*(dEz1+dEz2+dEz3+dEz4)/dx;
						
						dExy = 0.25*(dEx1+dEx2-dEx3-dEx4)/dy;
						dEzy = 0.25*(dEz1+dEz2-dEz3-dEz4)/dy;
						
						dExz = 0.25*(dEx1-dEx2+dEx3-dEx4)/dz;
						dEyz = 0.25*(dEy1-dEy2+dEy3-dEy4)/dz;
						
						/*** CPML boundary */
						/* y-direction */
						if (iy<mb[4]) {
							iabk_cpml = mb[2]-iy-1;
							/* x */
							psi_Exy[I33IZX] = b_h[iabk_cpml]*psi_Exy[I33IZX] + a_h[iabk_cpml]*dExy;
							dExy = dExy/kappa_h[iabk_cpml] + psi_Exy[I33IZX];
							/* z */
							psi_Ezy[I33IZX] = b_h[iabk_cpml]*psi_Ezy[I33IZX] + a_h[iabk_cpml]*dEzy;
							dEzy = dEzy/kappa_h[iabk_cpml] + psi_Ezy[I33IZX];
						}
						if (iy>mb[5]) {
							iabk_cpml = iy-mb[5]-1;
							/* x */
							psi_Exy[I33IZX] = b[iabk_cpml]*psi_Exy[I33IZX] + a[iabk_cpml]*dExy;
							dExy = dExy/kappa[iabk_cpml] + psi_Exy[I33IZX];
							/* z */
							psi_Ezy[I33IZX] = b[iabk_cpml]*psi_Ezy[I33IZX] + a[iabk_cpml]*dEzy;
							dEzy = dEzy/kappa[iabk_cpml] + psi_Ezy[I33IZX];
						}
						/* x-direction */
						if (ix<mb[2]) {
							iabk_cpml = mb[2]-ix-1;
							/* y */
							psi_Eyx[I33IZX] = b_h[iabk_cpml]*psi_Eyx[I33IZX] + a_h[iabk_cpml]*dEyx;
							dEyx = dEyx/kappa_h[iabk_cpml] + psi_Eyx[I33IZX];
							/* z */
							psi_Ezx[I33IZX] = b_h[iabk_cpml]*psi_Ezx[I33IZX] + a_h[iabk_cpml]*dEzx;
							dEzx = dEzx/kappa_h[iabk_cpml] + psi_Ezx[I33IZX];
						}
						if (ix>mb[3]) {
							iabk_cpml = ix-mb[3]-1;
							/* y */
							psi_Eyx[I33IZX] = b[iabk_cpml]*psi_Eyx[I33IZX] + a[iabk_cpml]*dEyx;
							dEyx = dEyx/kappa[iabk_cpml] + psi_Eyx[I33IZX];
							/* z */
							psi_Ezx[I33IZX] = b[iabk_cpml]*psi_Ezx[I33IZX] + a[iabk_cpml]*dEzx;
							dEzx = dEzx/kappa[iabk_cpml] + psi_Ezx[I33IZX];
						}
						/* z-direction */
						if (iz<mb[0]) {
							iabk_cpml = mb[0]-iz-1;
							/* y */
							psi_Eyz[I33IZX] = b_h[iabk_cpml]*psi_Eyz[I33IZX] + a_h[iabk_cpml]*dEyz;
							dEyz = dEyz/kappa_h[iabk_cpml] + psi_Eyz[I33IZX];
							/* x */
							psi_Exz[I33IZX] = b_h[iabk_cpml]*psi_Exz[I33IZX] + a_h[iabk_cpml]*dExz;
							dExz = dExz/kappa_h[iabk_cpml] + psi_Exz[I33IZX];
						}
						if (iz>mb[1]) {
							iabk_cpml = iz-mb[1]-1;
							/* y */
							psi_Eyz[I33IZX] = b[iabk_cpml]*psi_Eyz[I33IZX] + a[iabk_cpml]*dEyz;
							dEyz = dEyz/kappa[iabk_cpml] + psi_Eyz[I33IZX];
							/* x */
							psi_Exz[I33IZX] = b[iabk_cpml]*psi_Exz[I33IZX] + a[iabk_cpml]*dExz;
							dExz = dExz/kappa[iabk_cpml] + psi_Exz[I33IZX];
						}
						
						/* H */
						Hx[I33IZX] += dtu0 *(dEyz-dEzy);
						Hy[I33IZX] += dtu0 *(dEzx-dExz);
						Hz[I33IZX] += dtu0 *(dExy-dEyx);
						
					}/* End loop y-direction */
				}/* End loop x-direction */
			}/* End loop z-direction */
			
			/* update E at half-time grids */
			dHxy=dHxz=dHyx=dHyz=dHzx=dHzy=0.0;
			#pragma omp parallel for default(shared) \
				private(ix,iy,iz,in,iabk_cpml,\
						dHx1,dHx2,dHx3,dHx4,\
						dHy1,dHy2,dHy3,dHy4,\
						dHz1,dHz2,dHz3,dHz4,I33IZX,\
						dHyx,dHzx,dHxy,dHzy,dHxz,dHyz)
			for (iz=cb[0];iz<=cb[1];iz++) {
				for (ix=cb[2];ix<=cb[3];ix++) {
					for (iy=cb[4];iy<=cb[5];iy++) {
					
						I33IZX = I33(iz,ix,iy);
						
						dHx1=dHx2=dHx3=dHx4=0.0;
						dHy1=dHy2=dHy3=dHy4=0.0;
						dHz1=dHz2=dHz3=dHz4=0.0;
						
						for (in=0;in<sohalf;in++) {
							dHx1 += csg[in] * ( Hx[I33(iz+in,ix+in,iy+in)]-Hx[I33(iz-1-in,ix-1-in,iy-1-in)] );
							dHy1 += csg[in] * ( Hy[I33(iz+in,ix+in,iy+in)]-Hy[I33(iz-1-in,ix-1-in,iy-1-in)] );
							dHz1 += csg[in] * ( Hz[I33(iz+in,ix+in,iy+in)]-Hz[I33(iz-1-in,ix-1-in,iy-1-in)] );
							
							dHx2 += csg[in] * ( Hx[I33(iz-1-in,ix+in,iy+in)]-Hx[I33(iz+in,ix-1-in,iy-1-in)] );
							dHy2 += csg[in] * ( Hy[I33(iz-1-in,ix+in,iy+in)]-Hy[I33(iz+in,ix-1-in,iy-1-in)] );
							dHz2 += csg[in] * ( Hz[I33(iz-1-in,ix+in,iy+in)]-Hz[I33(iz+in,ix-1-in,iy-1-in)] );
							
							dHx3 += csg[in] * ( Hx[I33(iz+in,ix+in,iy-1-in)]-Hx[I33(iz-1-in,ix-1-in,iy+in)] );
							dHy3 += csg[in] * ( Hy[I33(iz+in,ix+in,iy-1-in)]-Hy[I33(iz-1-in,ix-1-in,iy+in)] );
							dHz3 += csg[in] * ( Hz[I33(iz+in,ix+in,iy-1-in)]-Hz[I33(iz-1-in,ix-1-in,iy+in)] );
							
							dHx4 += csg[in] * ( Hx[I33(iz-1-in,ix+in,iy-1-in)]-Hx[I33(iz+in,ix-1-in,iy+in)] );
							dHy4 += csg[in] * ( Hy[I33(iz-1-in,ix+in,iy-1-in)]-Hy[I33(iz+in,ix-1-in,iy+in)] );
							dHz4 += csg[in] * ( Hz[I33(iz-1-in,ix+in,iy-1-in)]-Hz[I33(iz+in,ix-1-in,iy+in)] );
						}
						
						dHyx = 0.25*(dHy1+dHy2+dHy3+dHy4)/dx;
						dHzx = 0.25*(dHz1+dHz2+dHz3+dHz4)/dx;
						
						dHxy = 0.25*(dHx1+dHx2-dHx3-dHx4)/dy;
						dHzy = 0.25*(dHz1+dHz2-dHz3-dHz4)/dy;
						
						dHxz = 0.25*(dHx1-dHx2+dHx3-dHx4)/dz;
						dHyz = 0.25*(dHy1-dHy2+dHy3-dHy4)/dz;
						
						/* CPML */
						/* y-direction */
						if (iy<mb[4]) {
							iabk_cpml = mb[4]-iy-1;
							/* x */
							psi_Hxy[I33IZX] = b[iabk_cpml]*psi_Hxy[I33IZX] + a[iabk_cpml]*dHxy;
							dHxy = dHxy/kappa[iabk_cpml] + psi_Hxy[I33IZX];
							/* z*/
							psi_Hzy[I33IZX] = b[iabk_cpml]*psi_Hzy[I33IZX] + a[iabk_cpml]*dHzy;
							dHzy = dHzy/kappa[iabk_cpml] + psi_Hzy[I33IZX];
						}
						if (iy>mb[5]) {
							iabk_cpml = iy-mb[5]-1;
							/* x */
							psi_Hxy[I33IZX] = b_h[iabk_cpml]*psi_Hxy[I33IZX] + a_h[iabk_cpml]*dHxy;
							dHxy = dHxy/kappa_h[iabk_cpml] + psi_Hxy[I33IZX];
							/* z*/
							psi_Hzy[I33IZX] = b_h[iabk_cpml]*psi_Hzy[I33IZX] + a_h[iabk_cpml]*dHzy;
							dHzy = dHzy/kappa_h[iabk_cpml] + psi_Hzy[I33IZX];
						}
						/* x-direction */
						if (ix<mb[2]) {
							iabk_cpml = mb[2]-ix-1;
							/* y */
							psi_Hyx[I33IZX] = b[iabk_cpml]*psi_Hyx[I33IZX] + a[iabk_cpml]*dHyx;
							dHyx = dHyx/kappa[iabk_cpml] + psi_Hyx[I33IZX];
							/* z*/
							psi_Hzx[I33IZX] = b[iabk_cpml]*psi_Hzx[I33IZX] + a[iabk_cpml]*dHzx;
							dHzx = dHzx/kappa[iabk_cpml] + psi_Hzx[I33IZX];
						}
						if (ix>mb[3]) {
							iabk_cpml = ix-mb[3]-1;
							/* y */
							psi_Hyx[I33IZX] = b_h[iabk_cpml]*psi_Hyx[I33IZX] + a_h[iabk_cpml]*dHyx;
							dHyx = dHyx/kappa_h[iabk_cpml] + psi_Hyx[I33IZX];
							/* z*/
							psi_Hzx[I33IZX] = b_h[iabk_cpml]*psi_Hzx[I33IZX] + a_h[iabk_cpml]*dHzx;
							dHzx = dHzx/kappa_h[iabk_cpml] + psi_Hzx[I33IZX];
						}
						/* z-direction */
						if (iz<mb[0]) {
							iabk_cpml = mb[0]-iz-1;
							/* x */
							psi_Hxz[I33IZX] = b[iabk_cpml]*psi_Hxz[I33IZX] + a[iabk_cpml]*dHxz;
							dHxz = dHxz/kappa[iabk_cpml] + psi_Hxz[I33IZX];
							/* y */
							psi_Hyz[I33IZX] = b[iabk_cpml]*psi_Hyz[I33IZX] + a[iabk_cpml]*dHyz;
							dHyz = dHyz/kappa[iabk_cpml] + psi_Hyz[I33IZX];
						}
						if (iz>mb[1]) {
							iabk_cpml = iz-mb[1]-1;
							/* x */
							psi_Hxz[I33IZX] = b_h[iabk_cpml]*psi_Hxz[I33IZX] + a_h[iabk_cpml]*dHxz;
							dHxz = dHxz/kappa_h[iabk_cpml] + psi_Hxz[I33IZX];
							/* y */
							psi_Hyz[I33IZX] = b_h[iabk_cpml]*psi_Hyz[I33IZX] + a_h[iabk_cpml]*dHyz;
							dHyz = dHyz/kappa_h[iabk_cpml] + psi_Hyz[I33IZX];
						}
						
						/* E */
						Ex2[I33IZX] = Pxx[I33IZX]*Ex[I33IZX] + Pxy[I33IZX]*Ey[I33IZX] + Pxz[I33IZX]*Ez[I33IZX] +
						              Qxx[I33IZX]*(dHzy-dHyz)+ Qxy[I33IZX]*(dHxz-dHzx)+ Qxz[I33IZX]*(dHyx-dHxy);
						Ey2[I33IZX] = Pxy[I33IZX]*Ex[I33IZX] + Pyy[I33IZX]*Ey[I33IZX] + Pyz[I33IZX]*Ez[I33IZX] +
						              Qxy[I33IZX]*(dHzy-dHyz)+ Qyy[I33IZX]*(dHxz-dHzx)+ Qyz[I33IZX]*(dHyx-dHxy);
						Ez2[I33IZX] = Pxz[I33IZX]*Ex[I33IZX] + Pyz[I33IZX]*Ey[I33IZX] + Pzz[I33IZX]*Ez[I33IZX] +
						              Qxz[I33IZX]*(dHzy-dHyz)+ Qyz[I33IZX]*(dHxz-dHzx)+ Qzz[I33IZX]*(dHyx-dHxy);
						
					}/* End loop y-direction */
				}/* End loop x-direction */
			}/* End loop z-direction */
			
			/* copy Ex2,Ey2,Ez2 to Ex,Ey,Ez */
			memcpy(Ex,Ex2,nzpadded*nxpadded*nypadded*sizeof(float));
			memcpy(Ey,Ey2,nzpadded*nxpadded*nypadded*sizeof(float));
			memcpy(Ez,Ez2,nzpadded*nxpadded*nypadded*sizeof(float));
			
			/* time increase */
			t=t+dt;
			iter = iter+1;
			
			/* verbose? */
			if(verbose && iter%50==0) {
				/* calculate energy */
				energy=0.0;
				for (iz=mb[0];iz<=mb[1];iz++) {
					for (ix=mb[2];ix<=mb[3];ix++) {
						for (iy=mb[4];iy<=mb[5];iy++) {
							I33IZX = I33(iz,ix,iy);
							energy = energy + Ez[I33IZX]*Ez[I33IZX]
											+ Ex[I33IZX]*Ex[I33IZX]
											+ Ey[I33IZX]*Ey[I33IZX];
						}
					}
				}
				cout<<"..."<<isou<<"..."<<iter<<"..."<<energy<<"..."<<omp_get_wtime()-start_time<<endl;
			}
			
			/* output snapshots */
			for (i=0;i<nsnap;i++) {
				/* save particle velocity */
				if (iter==isnap[i] && (snap==1 || snap==3)) {
					/* Ex */
					snap_fn_temp<<"S_"<<isou<<"_Ex_snap_"<<NINT(t*1e12)<<"ps.data"; /* ps(picosecond)=1e-12 second */
					snap_fn=snap_fn_temp.str();
					snap_fn_temp.str("");
					snap_Ex_fp=fopen(snap_fn.data(),"wb+");
					/* Ey */
					snap_fn_temp<<"S_"<<isou<<"_Ey_snap_"<<NINT(t*1e12)<<"ps.data";
					snap_fn=snap_fn_temp.str();
					snap_fn_temp.str("");
					snap_Ey_fp=fopen(snap_fn.data(),"wb+");
					/* Ez */
					snap_fn_temp<<"S_"<<isou<<"_Ez_snap_"<<NINT(t*1e12)<<"ps.data";
					snap_fn=snap_fn_temp.str();
					snap_fn_temp.str("");
					snap_Ez_fp=fopen(snap_fn.data(),"wb+");
					
					/* write files */
					for (iz=wbc[0];iz<nzpadded-wbc[1];iz++) {
						for (ix=wbc[2];ix<nxpadded-wbc[3];ix++) {
							for (iy=wbc[4];iy<nypadded-wbc[5];iy++) {
								I33IZX = I33(iz,ix,iy);
								fwrite(&Ex[I33IZX],sizeof(float),1,snap_Ex_fp);
								fwrite(&Ey[I33IZX],sizeof(float),1,snap_Ey_fp);
								fwrite(&Ez[I33IZX],sizeof(float),1,snap_Ez_fp);
							}
						}
					}
					
					fclose(snap_Ex_fp);
					fclose(snap_Ey_fp);
					fclose(snap_Ez_fp);
				}
			}
			
			/* output seismic record */
			for (ix=mb[2];ix<=mb[3];ix++) {
				for (iy=mb[4];iy<=mb[5];iy++) {
					fwrite(&Ex[I33(ihrz,ix,iy)], sizeof(float), 1, hEx_fp);
					fwrite(&Ey[I33(ihrz,ix,iy)], sizeof(float), 1, hEy_fp);
					fwrite(&Ez[I33(ihrz,ix,iy)], sizeof(float), 1, hEz_fp);
				}
			}
			
		}/* End loop over time */
		
	} /* End loop over sources */
	
	/* get end time */
	end_time = omp_get_wtime();
	printf("Elapsed time is %f seconds.\n",end_time-start_time);
	
	/* free spaces	*/
	delete [] Pxx;
	delete [] Pxy; 
	delete [] Pxz; 
	delete [] Pyy; 
	delete [] Pyz; 
	delete [] Pzz;
	delete [] Qxx;
	delete [] Qxy; 
	delete [] Qxz; 
	delete [] Qyy; 
	delete [] Qyz; 
	delete [] Qzz;
	
	delete [] Ex;
	delete [] Ey;
	delete [] Ez;
	delete [] Ex2;
	delete [] Ey2;
	delete [] Ez2;
	delete [] Hx;
	delete [] Hy;
	delete [] Hz;
	
	delete [] psi_Exy;
	delete [] psi_Exz;
	delete [] psi_Eyx;
	delete [] psi_Eyz;
	delete [] psi_Ezx;
	delete [] psi_Ezy;
	delete [] psi_Hxy;
	delete [] psi_Hxz;
	delete [] psi_Hyx;
	delete [] psi_Hyz;
	delete [] psi_Hzx;
	delete [] psi_Hzy;
	
	delete [] source;
	delete [] isnap;
	delete [] csg;
	delete [] isx;
	delete [] isy;
	delete [] isz;
	
} /* End main */


