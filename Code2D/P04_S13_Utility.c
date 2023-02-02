#include "P04_S13_header.h"

PetscErrorCode SpongeZone(AppCtx *user){
  // Compute Sponge zone info
  PetscReal TempTemp,Temp;
  for (St.i=0;St.i<DIM;St.i++){
    TempTemp = (St.Coord[St.i] - (user->SpongeZone[St.i][0] + user->DomainSize[St.i][0])) / user->SpongeZone[St.i][0];
    if (TempTemp < Temp) Temp = TempTemp;
    if (Temp <= 0.0) user->sigma = - user->SpongeCoef * TR(Temp) * (10.0 + 15.0 * Temp + 6.0 * SQ(Temp));
  }
  for (St.i=0;St.i<DIM;St.i++){
    TempTemp = (St.Coord[St.i] - (user->DomainSize[St.i][1] - user->SpongeZone[St.i][1])) / user->SpongeZone[St.i][1];
    if (TempTemp > Temp) Temp = TempTemp;
    if (Temp >= 0.0) user->sigma = user->SpongeCoef * TR(Temp) * (10.0 - 15.0 * Temp + 6.0 * SQ(Temp));
  }
}

PetscErrorCode CalcC_DC(AppCtx *user){
  user->C_DC_C = user->C_DC;
  user->C_DC_M = user->C_DC;
  user->C_DC_E = user->C_DC;
  if (St.rho < 0.0){
    user->C_DC_C *= (1.0 - St.rho_0 * 1e+10);
    user->C_DC_M *= (1.0 - St.rho_0 * 1e+10);
    user->C_DC_E *= (1.0 - St.rho_0 * 1e+10);
  }
}

PetscErrorCode CalcViscosity(AppCtx *user){
  user->rRe = St.rho / user->Re + 1.0 / (user->mu_ratio * user->Re);
  user->rRe_deri = 1.0 / user->Re;
  user->rRe_0 = St.rho_0 / user->Re + 1.0 / (user->mu_ratio * user->Re);
  user->rRe_deri_0 = 1.0 / user->Re;
}

PetscErrorCode FreeEnergy(PetscScalar rho,PetscScalar theta,PetscScalar grad_rho[],PetscScalar u[],PetscScalar *E_tmp,AppCtx *user)
{
  PetscFunctionBegin;
  //CalcValue(user,rho,theta,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,E_tmp);
  PetscInt i;
  for (i=0;i<user->dim;i++){
    *E_tmp += 0.5 * SQ(user->Ca * grad_rho[i]);
    *E_tmp += 0.5 * rho * SQ(u[i]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Energy(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscInt i,j,dim = user->dim,dof = user->dof;
  PetscScalar sol[dof];
  PetscScalar grad_sol[dof][dim];
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
  PetscScalar rho,grad_rho[dim],u[dim],theta;
  
  rho = sol[0]; 
  for (i=0;i<dim;i++){
    u[i] = sol[1+i]; grad_rho[i] = grad_sol[0][i];
  }
  if (user->Energy) theta = sol[dim+1];
  else theta = user->theta;
  FreeEnergy(rho,theta,&grad_rho[0],&u[0],S,user);
  PetscFunctionReturn(0);
}

PetscErrorCode GetGeomtry(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscInt i,j,dim = user->dim;
  PetscReal Coord[dim]; IGAPointFormGeomMap(pnt,&Coord[0]);
  for (i=0;i<n;i++) S[i] = Coord[i];
  PetscFunctionReturn(0);
}

PetscErrorCode NSKMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscBool Condition = user->Condition;
  PetscReal InitialTime = user->InitialTime;
  PetscReal newTime;
  PetscScalar scalar = 0.;
  ierr = IGAComputeScalar(user->iga,U,1,&scalar,Energy,user);CHKERRQ(ierr);
  PetscReal energy = PetscRealPart(scalar);

  PetscReal dt;
  TSGetTimeStep(ts,&dt);

  if(Condition){
    newTime = InitialTime + t;
  }else{
    newTime = t;
  }
  if(step > 0 && energy > user->energy) {
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e  WARNING: Free energy increased!\n",newTime,dt,energy);
  }else{
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e\n",newTime,dt,energy);
  }
  user->energy = energy;

  PetscFunctionReturn(0);
}

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *ctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)ctx;
  PetscInt i,dof = user->dof,Result = 0;
  PetscReal atol,rtol,n2dof[dof];

  Vec Res; ierr = SNESGetFunction(snes,&Res,NULL,NULL);

  ierr = SNESGetTolerances(snes,&atol,&rtol,NULL,NULL,NULL);

  for (i=0;i<dof;i++) ierr = VecStrideNorm(Res,i,NORM_2,&n2dof[i]);

  if (it_number == 0) {
    for (i=0;i<dof;i++){
      user->norm0[i] = n2dof[i];
    }
  }

  if (it_number == 0){
    for (i=0;i<dof;i++){
      if (user->rRes < (n2dof[i] / user->norm0[i])) user->rRes = (n2dof[i] / user->norm0[i]);
    }
  }

  PetscPrintf(PETSC_COMM_WORLD,"    rRes = %.6e    IT_NUMBER: %d     fnorm: %.4e   \n",user->rRes,it_number,fnorm);
  for (i=0;i<dof;i++){
    PetscPrintf(PETSC_COMM_WORLD,"  n%d: %.6e r %.1e ", i, n2dof[i], n2dof[i]/user->norm0[i]);  
    Result += (n2dof[i] <= rtol*user->norm0[i] || n2dof[i] < atol);
  }
  PetscPrintf(PETSC_COMM_WORLD," \n");

  if (Result >= dof) *reason = SNES_CONVERGED_FNORM_RELATIVE;
  PetscFunctionReturn(0);
}

PetscBool file_exist (char *filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
  PetscFunctionReturn(0);
}

PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal RelativeTime,Vec U,void *ctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)ctx;
  char           filename[256];
  char           der_filename[256];
  PetscBool Condition = user->Condition;
  PetscReal InitialTime = user->InitialTime;
  PetscReal AbsoluteTime;
  PetscReal PrintTimeInterval = user->PrintTimeInterval;

  TS_Alpha        *th = (TS_Alpha*)ts->data;

  if(Condition){
    AbsoluteTime = InitialTime + RelativeTime;
  }else{
    AbsoluteTime = RelativeTime;
  }
  if (PetscAbsReal(RelativeTime-ts->time_next_print) < 1.0e-10){
    {
      sprintf(filename,"./NSK_%.6f.dat",AbsoluteTime);
      sprintf(der_filename,"./Der_NSK_%.6f.dat",AbsoluteTime);
      PetscPrintf(PETSC_COMM_WORLD,"File Print Iteration %d\n",it_number);
      ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
      ierr = IGAWriteVec(user->iga,th->V1,der_filename);CHKERRQ(ierr);
      ts->time_next_print = ts->time_next_print + PrintTimeInterval;
    }
  }
  PetscFunctionReturn(0);
}
