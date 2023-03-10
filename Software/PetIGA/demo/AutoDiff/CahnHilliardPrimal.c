#include <petiga.h>

typedef struct {
  PetscReal theta;
  PetscReal alpha;
  PetscReal cbar;
} Params;

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

  IGASetFormIFunction(iga,IFunctionFAD,&params);
  IGASetFormIJacobian(iga,IJacobianFAD,&params);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-11);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);

  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&params,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

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
