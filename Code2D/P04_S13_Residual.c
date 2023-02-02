#include "P04_S13_header.h"

PetscErrorCode Residual_Inflow(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;
  PetscInt a,i;

  // Interpolate solution vector at current pnt
  PetscScalar sol_t[DOF],sol[DOF],grad_sol[DOF][DIM],hess_sol[DOF][DIM][DIM];
  IGAPointFormValue(pnt,V,&sol_t[0]);
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad(pnt,U,&grad_sol[0][0]);
  IGAPointFormHess(pnt,U,&hess_sol[0][0][0]); 

  UnpackSol(user,sol,sol_t,grad_sol,hess_sol);
  if (St.rho <= 1e-10) St.rho = 1e-10;
  CalcValue(user,1,St.rho,St.T,NULL);
  TIMModification(user); CalcViscosity(user);
  PetscReal c = 1e-4; if (St.dp_drho > SQ(c)) c = sqrt(St.dp_drho);

  const PetscReal *N0,(*N1)[DIM];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  const PetscReal *surf_normal = pnt->normal;
  IGAPointFormGeomMap(pnt,&St.Coord[0]); 

  // Map from global to local coordinate
  PetscScalar rho = sol[0], rho_t = sol_t[0];
  PetscScalar u[3] = {0.0}, u_t[3] = {0.0}; 
  PetscScalar grad_rho[3] = {0.0}, grad_u[3][3] = {{0.0}};

  // For inlet, norm should point inward
  PetscReal normal[DIM];
  for (i=0;i<DIM;i++) normal[i] = -surf_normal[i];

  if (PetscAbsReal(normal[0]) == 1 || PetscAbsReal(normal[1]) == 1){// Vertical or horizontal line
    u[0] = sol[1] * normal[0] + sol[2] * normal[1];
    u[1] = -sol[1] * normal[1] + sol[2] * normal[0];
    u_t[0] = sol_t[1] * normal[0] + sol_t[2] * normal[1];
    u_t[1] = -sol_t[1] * normal[1] + sol_t[2] * normal[0];    
    grad_rho[0] = grad_sol[0][0] * normal[0] + grad_sol[0][1] * normal[1];
    grad_rho[1] = grad_sol[0][1] * normal[0] - grad_sol[0][0] * normal[1];
    grad_u[0][0] = grad_sol[1][0] * SQ(normal[0]) + grad_sol[2][1] * SQ(normal[1]);
    grad_u[0][1] = grad_sol[1][1] * SQ(normal[0]) - grad_sol[2][0] * SQ(normal[1]);
    grad_u[1][0] = grad_sol[2][0] * SQ(normal[0]) - grad_sol[1][1] * SQ(normal[1]);
    grad_u[1][1] = grad_sol[2][1] * SQ(normal[0]) + grad_sol[1][0] * SQ(normal[1]);
  }else{//Curved coordinate, need to implement

  }

  // Characteristic velocity and wave
  PetscReal lambda[5] = {0.0};
  lambda[0] = u[0] - c;
  lambda[1] = u[0];
  lambda[2] = u[0];
  lambda[3] = u[0];
  lambda[4] = u[0] + c;

  PetscReal L[5] = {0.0};
  L[0] = lambda[0] * (St.dp_drho * grad_rho[0] - St.rho * c * grad_u[0][0]);
  L[4] = L[0];

  PetscReal d[5] = {0.0};
  d[0] = (L[1] + 0.5 * (L[4] + L[0])) / SQ(c);

  PetscReal L_norm = (St.u1 - c) * (St.dp_drho * St.rho_1 - St.rho * c * St.u1_1);

  PetscScalar (*R)[user->dof] = (PetscScalar (*)[user->dof])Re;
  for(a=0; a<St.nen; a++) {
    for(i=0;i<user->dof;i++) R[a][i] = 0.0;
    R[a][0] = user->Penalty_CoM * N0[a] * (rho_t + d[0] + rho * grad_u[1][1] + u[1] * grad_rho[1]);
  }
  return 0;
}

PetscErrorCode Residual_Outflow(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;
  PetscInt a,i;

  // Interpolate solution vector at current pnt
  PetscScalar sol_t[DOF],sol[DOF],grad_sol[DOF][DIM],hess_sol[DOF][DIM][DIM];
  IGAPointFormValue(pnt,V,&sol_t[0]);
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad(pnt,U,&grad_sol[0][0]);
  IGAPointFormHess(pnt,U,&hess_sol[0][0][0]); 

  UnpackSol(user,sol,sol_t,grad_sol,hess_sol);
  if (St.rho <= 1e-10) St.rho = 1e-10;
  CalcValue(user,1,St.rho,St.T,NULL);
  TIMModification(user); CalcViscosity(user);
  PetscReal c = 1e-4; if (St.dp_drho > SQ(c)) c = sqrt(St.dp_drho);
  PetscReal M = sqrt(St.u_norm2) / c;

  const PetscReal *N0,(*N1)[DIM];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  const PetscReal *normal = pnt->normal;
  IGAPointFormGeomMap(pnt,&St.Coord[0]); 

  // Map from global to local coordinate
  PetscScalar rho = sol[0], rho_t = sol_t[0];
  PetscScalar u[3] = {0.0}, u_t[3] = {0.0}; 
  PetscScalar grad_rho[3] = {0.0}, grad_u[3][3] = {{0.0}};

  if (PetscAbsReal(normal[0]) == 1 || PetscAbsReal(normal[1]) == 1){// Vertical or horizontal line
    u[0] = sol[1] * normal[0] + sol[2] * normal[1];
    u[1] = -sol[1] * normal[1] + sol[2] * normal[0];
    u_t[0] = sol_t[1] * normal[0] + sol_t[2] * normal[1];
    u_t[1] = -sol_t[1] * normal[1] + sol_t[2] * normal[0];    
    grad_rho[0] = grad_sol[0][0] * normal[0] + grad_sol[0][1] * normal[1];
    grad_rho[1] = grad_sol[0][1] * normal[0] - grad_sol[0][0] * normal[1];
    grad_u[0][0] = grad_sol[1][0] * SQ(normal[0]) + grad_sol[2][1] * SQ(normal[1]);
    grad_u[0][1] = grad_sol[1][1] * SQ(normal[0]) - grad_sol[2][0] * SQ(normal[1]);
    grad_u[1][0] = grad_sol[2][0] * SQ(normal[0]) - grad_sol[1][1] * SQ(normal[1]);
    grad_u[1][1] = grad_sol[2][1] * SQ(normal[0]) + grad_sol[1][0] * SQ(normal[1]);
  }else{//Curved coordinate, need to implement

  }

  // Characteristic velocity and wave
  PetscReal lambda[5] = {0.0};
  lambda[0] = u[0] - c;
  lambda[1] = u[0];
  lambda[2] = u[0];
  lambda[3] = u[0];
  lambda[4] = u[0] + c;

  PetscReal L[5] = {0.0};
  L[0] = lambda[0] * (St.dp_drho * grad_rho[0] - St.rho * c * grad_u[0][0]);
  L[1] = lambda[1] * (SQ(c) * grad_rho[0] - St.dp_drho * grad_rho[0]);
  L[2] = lambda[2] * grad_u[1][0];
  L[3] = lambda[3] * grad_u[2][0];
  L[4] = lambda[4] * (St.dp_drho * grad_rho[0] + St.rho * c * grad_u[0][0]);

  L[0] = user->K_outflow * (St.p - user->p_BC); // Modify L1 to impose pressure at infinity

  PetscReal d[5] = {0.0};
  d[0] = (L[1] + 0.5 * (L[4] + L[0])) / SQ(c);
  d[1] = 0.5 * (L[4] + L[0]);
  d[2] = 0.5 * (L[4] - L[0]) / (rho * c);
  d[3] = L[2];
  d[4] = L[3];


  PetscScalar (*R)[user->dof] = (PetscScalar (*)[user->dof])Re;
  for(a=0; a<St.nen; a++) {
    for(i=0;i<user->dof;i++) R[a][i] = 0.0;
      //R[a][0] = user->Penalty_CoM * N0[a] * (rho_t + d[0] + u[1] * grad_rho[1] + rho * grad_u[1][1]);
      //R[a][1] = user->Penalty_CoLM * N0[a] * (rho_t * u[0] + rho * u_t[0] + u[0] * d[0] + rho * d[2] + rho * u[0] * grad_u[1][1] + grad_rho[1] * u[0] * u[1] + rho * grad_u[0][1] * u[1]);
      //R[a][2] = user->Penalty_CoLM * N0[a] * (rho_t * u[1] + rho * u_t[1] + u[1] * d[0] + rho * d[3] + rho * u[1] * grad_u[1][1] + grad_rho[1] * u[1] * u[1] + rho * grad_u[1][1] * u[1] + St.dp_drho * grad_rho[1]);
      R[a][1] = N0[a] * user->p_BC * normal[0] * St.Coord[0];
      R[a][2] = N0[a] * user->p_BC * normal[1] * St.Coord[0];
  }
  return 0;
}

PetscErrorCode Residual_Wall(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;
  PetscInt a,i;

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

  PetscScalar (*R)[user->dof] = (PetscScalar (*)[user->dof])Re;
  for(a=0; a<St.nen; a++) {
    for(i=0;i<user->dof;i++) R[a][i] = 0.0;
    //R[a][user->dof-1] = 1000.0 * user->Ca2 * N0[a] * (grad_sol[0][0] * normal[0] + grad_sol[0][1] * normal[1]);
    //for (i = 0; i < user->dof; i++) R[St.a][i] *= St.Coord[0];
  }
  return 0;
}



PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx; 
  St.nen = pnt->nen; St.shift = shift;

  PetscInt axis,side;IGAPointAtBoundary(pnt,&axis,&side);
  if(pnt->atboundary){
    if (user->BCType_Identify[axis][side] == 1) return Residual_Inflow(pnt,shift,V,t,U,t0,U0,V0,Re,ctx);
    else if (user->BCType_Identify[axis][side] == 2) return Residual_Outflow(pnt,shift,V,t,U,t0,U0,V0,Re,ctx);
    else return Residual_Wall(pnt,shift,V,t,U,t0,U0,V0,Re,ctx);
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
    user->sigma = 0;
    SpongeZone(user);
    GetResSpongeStrong(user); GetResSpongeStrong0(user);
  }
  if(user->Coord == 1 && user->dim == 2){//Cylndrical coordinate, by default, the DIM = 2
      St.Res_Strong_0 += St.rho * St.u1 / St.Coord[0];
      St.Res_Strong0_0 += St.rho_0 * St.u1_0 / St.Coord[0];

      St.Res_Strong_1 += St.rho * St.u1 * St.u1 / St.Coord[0];
      St.Res_Strong_1 -= 4./3. * user->rRe * (St.u1_1 - St.u1 / St.Coord[0]) / St.Coord[0];
      St.Res_Strong0_1 += St.rho_0 * St.u1_0 * St.u1_0 / St.Coord[0];
      St.Res_Strong0_1 -= 4./3. * user->rRe_0 * (St.u1_1_0 - St.u1_0 / St.Coord[0]) / St.Coord[0];

      St.Res_Strong_2 += St.rho * St.u1 * St.u2 / St.Coord[0];
      St.Res_Strong_2 -= user->rRe * (St.u1_2 / 3.0 + St.u2_1); 
      St.Res_Strong0_2 += St.rho_0 * St.u1_0 * St.u2_0 / St.Coord[0];
      St.Res_Strong0_2 -= user->rRe_0 * (St.u1_2_0 / 3.0 + St.u2_1_0); 

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

  PetscScalar (*R)[user->dof] = (PetscScalar (*)[user->dof])Re;
  for(St.a=0; St.a<St.nen; St.a++) {
    UnpackShape(user,1,0,N0,N1,NULL);
    GetResWeak(user,R);
    if(user->SUPG == 1 || user->SUPG == 2) GetResSUPG_Simple(user,R);
    else if(user->SUPG == 3 || user->SUPG == 4) GetResSUPG(user,R);
    if(user->DC) GetResDC(user,R);
    if(user->Sponge) GetResSponge(user,sol,R);
    if(user->Coord == 1 && user->dim == 2){//Cylndrical coordinate, by default, the DIM = 2
      for (St.i = 0; St.i < user->dof; St.i++) R[St.a][St.i] *= St.Coord[0];
      R[St.a][0] += N0[St.a] * St.rho * St.u1; 

      R[St.a][1] += N0[St.a] * St.rho * St.u1 * St.u1;
      R[St.a][1] -= N0[St.a] * St.p;
      R[St.a][1] -= N0[St.a] * 2./3. * user->rRe * St.u2_2;

      R[St.a][2] += N0[St.a] * St.rho * St.u1 * St.u2;
      R[St.a][2] += N0[St.a] * 2./3. * user->rRe * St.u1_2;
    }
  }
  return 0;
}

GetResSponge(const AppCtx *user,PetscScalar (*sol),PetscScalar (*R)[user->dof]){
  R[St.a][0] += St.N0_a * user->sigma * (St.rho - user->rho_inlet); 
  for (St.i=0;St.i<DIM;St.i++) R[St.a][St.i+1] += St.N0_a * user->sigma * (sol[St.i+1] - user->u_BC[St.i]); 
  if (user->Energy) R[St.a][DIM+1] += St.N0_a * user->sigma * (St.T - user->T_inlet); 
  return 0;
}
