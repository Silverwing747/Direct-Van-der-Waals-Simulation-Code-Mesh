#include "P04_S13_header.h"

PetscErrorCode Jacobian_Wall(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;
  PetscInt a,b,i;

  // Interpolate solution vector at current pnt
  PetscScalar sol_t[DOF],sol[DOF],grad_sol[DOF][DIM],hess_sol[DOF][DIM][DIM];
  IGAPointFormValue(pnt,V,&sol_t[0]);
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad(pnt,U,&grad_sol[0][0]);
  IGAPointFormHess(pnt,U,&hess_sol[0][0][0]); 
  
  const PetscReal *N0,(*N1)[DIM],(*N2)[DIM][DIM];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);
  const PetscReal *normal = pnt->normal;
  IGAPointFormGeomMap(pnt,&St.Coord[0]); 

  PetscScalar (*J)[user->dof][St.nen][user->dof] = (PetscScalar (*)[user->dof][St.nen][user->dof])Je;

  return 0;
}

PetscErrorCode Jacobian(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;

  PetscInt axis,side;IGAPointAtBoundary(pnt,&axis,&side);
  if(pnt->atboundary){
    if (user->BCType_Identify[axis][side] == 1) return IGAFormIEJacobianFD(pnt,shift,V,t,U,t0,U0,V0,Je,ctx);
    else if (user->BCType_Identify[axis][side] == 2) return IGAFormIEJacobianFD(pnt,shift,V,t,U,t0,U0,V0,Je,ctx);
    else return Jacobian_Wall(pnt,shift,V,t,U,t0,U0,V0,Je,ctx);
  } 
  TSGetTimeStep(user->ts,&St.dt); 

  // Interpolate solution vector at current pnt
  PetscScalar sol_t[DOF],sol[DOF],grad_sol[DOF][DIM],hess_sol[DOF][DIM][DIM];
  PetscScalar sol_t0[DOF],sol0[DOF],grad_sol0[DOF][DIM],hess_sol0[DOF][DIM][DIM];
  PetscScalar grad_sol_t0[DOF][DIM], hess_sol_t0[DOF][DIM][DIM];
  IGAPointFormValue(pnt,V,&sol_t[0]); IGAPointFormValue(pnt,V0,&sol_t0[0]);
  IGAPointFormValue(pnt,U,&sol[0]); IGAPointFormValue(pnt,U0,&sol0[0]);
  IGAPointFormGrad(pnt,U,&grad_sol[0][0]); IGAPointFormGrad(pnt,U0,&grad_sol0[0][0]);
  IGAPointFormHess(pnt,U,&hess_sol[0][0][0]); IGAPointFormHess(pnt,U0,&hess_sol0[0][0][0]); 
  IGAPointFormGrad(pnt,V0,&grad_sol_t0[0][0]); IGAPointFormHess(pnt,V0,&hess_sol_t0[0][0][0]); 

  // Unpack solution and get EoS value (including TIM)
  UnpackSol(user,sol,sol_t,grad_sol,hess_sol); UnpackSol0(user,sol0,sol_t0,grad_sol0,hess_sol0);
  CalcC_DC(user);
  //if (St.rho <= 1e-10) St.rho = 1e-10; if (St.rho_0 <= 1e-10) St.rho_0 = 1e-10;
  CalcValue(user,1,St.rho,St.T,NULL); CalcValue0(user,St.rho_0,St.T_0); 
  TIMModification(user); CalcViscosity(user);
  
  // SUPG, DC and Sponge
  IGAPointFormGeomMap(pnt,&St.Coord[0]); 
  GetGeometricTensor(pnt);
  if (user->NSK_Mod){Calc_Inter_dp(user);Calc_Inter_dp0(user);}
  GetResStrong(user); GetResStrong0(user); 
  if (user->Sponge){
    SpongeZone(user);
    GetResSpongeStrong(user); GetResSpongeStrong0(user);
  }
  
  switch(user->SUPG){
    case 1: GetTauSUPG_Simple(user); break; 
    case 2: GetTauSUPG_Simple_0(user); break;
    case 3: GetTauSUPG(user); break;
    case 4: GetTauSUPG_0(user); break;
  }

  if (user->DC) GetKappaDC(user);

  const PetscReal *N0,(*N1)[DIM],(*N2)[DIM][DIM];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);

  PetscScalar (*J)[user->dof][St.nen][user->dof] = (PetscScalar (*)[user->dof][St.nen][user->dof])Je;
  for(St.b=0; St.b<St.nen; St.b++) {
    UnpackShape(user,0,1,N0,N1,N2); GetJacStrong(user);GetJacSpongeStrong(user);
    for(St.a=0; St.a<St.nen; St.a++) {
      UnpackShape(user,1,0,N0,N1,N2); GetJacWeak(user,J);
      if (user->NSK_Mod) Calc_Inter_ddp(user);
      if(user->SUPG == 1 || user->SUPG == 2) GetJacSUPG_Simple(user,J);
      else if(user->SUPG == 3 || user->SUPG == 4) GetJacSUPG(user,J);
      if(user->DC) GetJacDC(user,J);
      if(user->Sponge) GetJacSponge(user,J);
    }
  }
  return 0;
}

GetJacSponge(const AppCtx *user,PetscScalar (*J)[user->dof][St.nen][user->dof]){
  J[St.a][0][St.b][0] += St.N0_a * user->sigma * St.N0_b; 
  for (St.i=0;St.i<DIM;St.i++) J[St.a][St.i+1][St.b][St.i+1] += St.N0_a * user->sigma * St.N0_b;
  if (user->Energy){
    J[St.a][DIM+1][St.b][DIM+1] += St.N0_a * user->sigma * St.N0_b; 
    J[St.a][0][St.b][DIM+1] += St.N0_a * user->sigma * St.dp_dT * St.N0_b; 
  }
}
