#include "petiga.h"
#include "petiga.h"
#include <sys/types.h>
#include <sys/stat.h>
#include "petscsnes.h"
#include <petsc/private/tsimpl.h>
#define SQ(x) ((x)*(x))
typedef struct {
  PetscReal theta;
  PetscReal alpha;
  PetscReal cbar;
  PetscReal norm0_0;
  PetscReal PrintTimeInterval;
  PetscReal Eprev;
} Params;

EXTERN_C_BEGIN
PetscErrorCode IFunctionC99(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianC99(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
PetscErrorCode IFunctionCXX(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianCXX(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
PetscErrorCode IFunctionFAD(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianFAD(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  Params *user = (Params *)cctx;

  Vec Res;
  PetscScalar n2dof0;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);


  if (it_number == 0) {
    user->norm0_0 = n2dof0;
    if (n2dof0 == 0.0){
      user->norm0_0 = 1.0;
      PetscPrintf(PETSC_COMM_WORLD,"    The initial n2dof0 is NaN, reset to 1.0");
    }
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    n0: %.6e r %.1e \n", n2dof0, n2dof0/user->norm0_0);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;

  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);

  if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) ) {
    *reason = SNES_CONVERGED_FNORM_RELATIVE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitial(IGA iga,Vec C,Params *user)
{
  MPI_Comm       comm;
  PetscRandom    rctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rctx,user->cbar-0.05,user->cbar+0.05);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(C,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscReal GinzburgLandauFreeEnergy(PetscReal c,PetscReal cx,PetscReal cy,Params *user)
{
  PetscReal theta = user->theta;
  PetscReal alpha = user->alpha;
  PetscReal E = c*log(c) + (1-c)*log(1-c) + 2*theta*c*(1-c) + theta/(3*alpha)*(cx*cx+cy*cy);
  return E;
}

PetscErrorCode Stats(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  Params *user = (Params *)ctx;
  PetscFunctionBegin;

  PetscScalar c,c1[3];
  IGAPointFormValue(p,U,&c);
  IGAPointFormGrad(p,U,&c1[0]);
  PetscReal diff = c - user->cbar;

  S[0] = GinzburgLandauFreeEnergy(c,c1[0],c1[1],user); // Free energy
  S[1] = diff*diff;                                    // Second moment
  S[2] = S[1]*diff;                                    // Third moment

  PetscFunctionReturn(0);
}

PetscErrorCode StatsMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  Params         *user = (Params *)mctx;
  IGA            iga;
  PetscReal      dt;
  PetscScalar    stats[3] = {0,0,0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAComputeScalar(iga,U,3,&stats[0],Stats,mctx);CHKERRQ(ierr);

  if (step == 0) {ierr = PetscPrintf(PETSC_COMM_WORLD,"#Time        dt           Free Energy            Second moment          Third moment\n");CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e %.16e %.16e\n",(double)t,(double)dt,(double)stats[0],(double)stats[1],(double)stats[2]);CHKERRQ(ierr);

  if (step == 0) user->Eprev = PETSC_MAX_REAL;
  if((PetscReal)stats[0] > user->Eprev) {ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING: Ginzburg-Landau free energy increased!\n");CHKERRQ(ierr);}
  user->Eprev = PetscRealPart(stats[0]);
  PetscFunctionReturn(0);
}

PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  IGA            iga;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  ierr = PetscSNPrintf(filename,sizeof(filename),"./ch2d%d.dat",(int)step);CHKERRQ(ierr);
  ierr = IGAWriteVec(iga,U,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  /* Define simulation specific parameters */
  Params params;
  params.alpha = 3000.0; /* interface thickess parameter */
  params.theta = 1.5;    /* temperature/critical temperature */
  params.cbar  = 0.63;   /* average concentration */

  /* Set discretization options */
  char      initial[PETSC_MAX_PATH_LEN] = {0};
  PetscBool output  = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","CahnHilliard Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cbar","Initial average concentration",__FILE__,params.cbar,&params.cbar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Interface thickess parameter",__FILE__,params.alpha,&params.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-theta","Ratio temperature/critical temperature",__FILE__,params.theta,&params.theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-wrap",  "", "-iga_periodic");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-dim",  "2", "-iga_dim");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-deg",  "2", "-iga_degree");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-nel", "64", "-iga_elements");CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetOrder(iga,2);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFieldName(iga,0,"c");CHKERRQ(ierr);

  PetscBool c99 = IGAGetOptBool(NULL,"-c99",PETSC_TRUE);
  if (c99) {
    IGASetFormIFunction(iga,IFunctionC99,&params);
    IGASetFormIJacobian(iga,IJacobianC99,&params);
  }
  //PetscBool f90 = IGAGetOptBool(NULL,"-f90",PETSC_FALSE);
  //if (f90) {
  //  IGASetFormIFunction(iga,IFunctionF90,&params);
  //  IGASetFormIJacobian(iga,IJacobianF90,&params);
  //}
  PetscBool cxx = IGAGetOptBool(NULL,"-cxx",PETSC_FALSE);
  if (cxx) {
    IGASetFormIFunction(iga,IFunctionCXX,&params);
    IGASetFormIJacobian(iga,IJacobianCXX,&params);
  }
  PetscBool fad = IGAGetOptBool(NULL,"-fad",PETSC_FALSE);
  if (fad) {
    IGASetFormIFunction(iga,IFunctionFAD,&params);
    IGASetFormIJacobian(iga,IJacobianFAD,&params);
  }
  PetscBool fd = IGAGetOptBool(NULL,"-fd",PETSC_FALSE);
  if (fd) {IGASetFormIJacobian(iga,IGAFormIJacobianFD,&params);}

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-11);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);
  ts->time_next_print = INFINITY;
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&params,NULL);CHKERRQ(ierr);}
  ierr = TSMonitorSet(ts,StatsMonitor,&params,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&params,NULL);CHKERRQ(ierr);

  Vec C;
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  if (initial[0] == 0) { /* initial condition is random */
    ierr = FormInitial(iga,C,&params);CHKERRQ(ierr);
  } else {               /* initial condition from datafile */
    ierr = IGAReadVec(iga,C,initial);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts,C);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
