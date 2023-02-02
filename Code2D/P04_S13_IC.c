#include "P04_S13_header.h"

PetscErrorCode L2Projection_IC(IGAPoint pnt,PetscScalar *KK,PetscScalar *FF,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscInt dim = user->dim, dof = user->dof, nen = pnt->nen;
  PetscReal F = user->F, Ca = user->Ca * F; // Notice interface is actually thickened by F using TIM, instead of sqrt(F);
  PetscInt i,j,k;
  PetscReal Coord[dim]; IGAPointFormGeomMap(pnt,&Coord[0]);

  const PetscReal *N0,(*N1)[dim],(*N2)[dim][dim];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);

  PetscReal d1 = 0.0, d2 = 0.0, d3 = 0.0;
  for (i=0;i<dim;i++){
    d1 += SQ(Coord[i] - user->C1[i]);
    d2 += SQ(Coord[i] - user->C2[i]);
    d3 += SQ(Coord[i] - user->C3[i]);
  }
  d1 = sqrt(d1); d2 = sqrt(d2); d3 = sqrt(d3);

  // a = (rho_v * n - rho_l * (n-2)) / 2 where n is number of bubble, b = (rho_l - rho_v) / 2 always
  PetscReal a = (user->rho_v_IC * 3.0 - user->rho_l_IC) / 2.0; 
  PetscReal b = (user->rho_l_IC - user->rho_v_IC) / 2.0;
  //PetscReal a = (rho_l * 3.0 - rho_v) / 2.0;
  //PetscReal b = (rho_v - rho_l) / 2.0; 
  PetscReal rho_IC = a + b * (tanh(user->IC_Scale * (d1-user->R1)/Ca)
                            + tanh(user->IC_Scale * (d2-user->R2)/Ca)
                            + tanh(user->IC_Scale * (d3-user->R3)/Ca));

  if (user->Case == 2){ // Elliptics IC
    PetscReal Rb = user->R1 * user->Aspect1;
    d1 = (sqrt(SQ(Coord[0] - user->C1[0]) / SQ(user->R1) + SQ(Coord[1] - user->C1[1]) / SQ(Rb)) - 1.0) * user->R1;
    a = (user->rho_v_IC + user->rho_l_IC) / 2.0;  
    rho_IC = a + b * tanh(user->IC_Scale * d1 / Ca);
  }

  PetscReal theta_liquid = user->theta;
  PetscReal theta_vapor = user->theta;
  a = (theta_vapor * 3.0 - theta_liquid) / 2.0;
  b = (theta_liquid - theta_vapor) / 2.0;
  PetscReal T_IC = a + b * (tanh(user->IC_Scale * (d1-user->R1)/Ca)
                            + tanh(user->IC_Scale * (d2-user->R2)/Ca)
                            + tanh(user->IC_Scale * (d3-user->R3)/Ca));
  // Special IC for Interface oscillation case, only for 2D
  if (user->Case == 1){
    PetscReal Amplitude = 0.025;
    PetscReal x_c = Amplitude * 0.01 * F * 0.5 + Amplitude * 0.01 * F * 0.5 * (cos(2.0 * M_PI * Coord[1] / (0.005 * F)) - 1.0);
    d1 = Coord[0] - x_c;
    a = (St.rho_l + St.rho_v) / 2.0;
    b = (St.rho_l - St.rho_v) / 2.0;
    rho_IC = a + b * tanh(user->IC_Scale*(d1)/Ca);
  }

  PetscReal u_IC[3];
  u_IC[0] = user->u_IC[0];
  u_IC[1] = user->u_IC[1];
  u_IC[2] = user->u_IC[2];



/*   if (user->Case == 2){
    u_IC[0] = 0.0;
    PetscReal rho_l = 1.0, rho_r = 0.125;
    PetscReal T_l = 3.484e-3, T_r = 2.787e-3;
    d1 = Coord[0] - 0.5;
    a = (rho_l + rho_r) / 2.0;
    b = (rho_r - rho_l) / 2.0;
    rho_IC = a + b * tanh(2.0*(d1)/2e-3);
    a = (T_l + T_r) / 2.0;
    b = (T_r - T_l) / 2.0;
    T_IC = a + b * tanh(2.0*(d1)/2e-3);
  } */

/*   u_IC[0] = 0.0;
  PetscReal rho_l = user->rho_v_IC, rho_r = user->rho_l_IC;
  PetscReal T_l = user->theta-0.1, T_r = user->theta+0.3;
  d1 = Coord[0] - 0.5;
  a = (rho_l + rho_r) / 2.0;
  b = (rho_r - rho_l) / 2.0;
  rho_IC = a + b * tanh((d1)/Ca);
  a = (T_l + T_r) / 2.0;
  b = (T_r - T_l) / 2.0;
  T_IC = a + b * tanh((d1)/Ca); */

  if (user->Case == 3){// Need to change cylinder radius
    PetscReal r = sqrt(SQ(Coord[0]) + SQ(Coord[1]));
    PetscReal R = 0.5;
    PetscReal U_inf = user->u_IC[0];
    PetscReal cos_theta = Coord[0] / r;
    PetscReal sin_theta = Coord[1] / r;
    PetscReal R_r = SQ(R/r);

    u_IC[0] = U_inf * (1.0 - R_r) * SQ(cos_theta) + U_inf * (R_r + 1.0) * SQ(sin_theta);
    u_IC[1] = -U_inf * cos_theta * sin_theta * 2.0 * R_r;

  }
 
  PetscScalar (*K)[dof][nen][dof] = (typeof(K)) KK;
  PetscScalar (*Ff)[dof]           = (typeof(Ff)) FF;

  PetscInt aa,bb;
  for (aa=0; aa<nen; aa++) {
    for (bb=0; bb<nen; bb++) {
      for (i=0;i<dof;i++) K[aa][i][bb][i] = N0[aa] * N0[bb];
    }
    Ff[aa][0] = N0[aa] * rho_IC;
    if (user->Energy) Ff[aa][dim+1] = N0[aa] * T_IC;
    for (i=0;i<dim;i++) Ff[aa][i+1] = N0[aa] * u_IC[i];
  }
  return 0;
}
