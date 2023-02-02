#include <petiga.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>

#define SQ(x) ((x)*(x))
#define TR(x) ((x)*(x)*(x))
#define M_PI 3.14159265358979323846

typedef struct {
  // General
  IGA       iga;
  TS        ts;
  SNES      nonlin;
  PetscInt  dim,dof,BCType,Case,TIM,DC,Sponge,N,C,p,MeshSize[3];
  PetscInt  BCType_Identify[3][2]; // Flag for boundary integral, 0=no boundary integral, 1=inflow, 2=outflow, 3=no-slip wall
  PetscReal energy,norm0[6],rRes,DomainSize[3][2];
  PetscReal SpongeZone[3][2],SpongeCoef,sigma;
  PetscReal Penalty_CoM, Penalty_CoLM, K_outflow;
  PetscBool Energy; // solve isothermo or full energy

  //Physical parameters
  PetscReal Re,Ca,Ge,Pe,F,Angle,delta;
  PetscReal rRe_0,rRe_deri_0,rRe,rRe_deri,rPe,rPe0,Ca2,mu_ratio;
  PetscScalar theta;

  PetscReal d_l,d_v; // Modified EoS param

  // SUPG and DC related
  PetscInt  DBIter,SUPG,LocalScale;
  PetscReal C_SUPG,C_DC,C_DC_C,C_DC_M,C_DC_E,alpha_DC,Ca2_SUPG,NSK_Mod,dp_drho_original,dp_drho_original_0; // DC and SUPG Params
  PetscReal Coord; // Type of coordinate 

  // IC and BC
  PetscReal rho_v_IC,rho_l_IC,T_IC,u_BC[3],u_IC[3],ICScale_v,ICScale_l,IC_Scale;
  PetscReal rho_inlet,rho_outlet,T_inlet,T_outlet,p_BC;
  PetscReal InitialTime,PrintTimeInterval;
  PetscReal C1[3],C2[3],C3[3],R1,R2,R3,Aspect1;
  PetscBool Condition;
} AppCtx;

// typedef struct {
//   PetscReal stage_time;
//   PetscReal scale_F;
//   Vec       X0,Xa,X1;
//   Vec       V0,Va,V1;

//   PetscReal Alpha_m;
//   PetscReal Alpha_f;
//   PetscReal Gamma;
//   PetscInt  order;

//   Vec       vec_sol_prev;
//   Vec       vec_lte_work;

//   TSStepStatus status;
// } TS_Alpha;

#define DIM 2
#define DOF 5
typedef struct{
  PetscInt  a,b,i,j,k,nen;
  PetscReal dp_inter_1,dp_inter_2;
  PetscReal dp_inter_1_0,dp_inter_2_0;
  PetscReal ddp_inter_aux_1,ddp_inter_aux_2;
  PetscReal ddp_inter_rho_1,ddp_inter_rho_2;
  PetscReal ddp_inter_T_1,ddp_inter_T_2;
  PetscReal p,dp_drho,dp_dT,ddp_drho_dT,ddp_drho2,ddp_dT2,e,de_drho,de_dT,dde_drho_dT,dde_drho2,dde_dT2;
  PetscReal p_0,dp_drho_0,dp_dT_0,ddp_drho_dT_0,ddp_drho2_0,ddp_dT2_0,e_0,de_drho_0,de_dT_0,dde_drho_dT_0,dde_drho2_0,dde_dT2_0;
  PetscReal dt,shift,Coord[DIM],InvGradMap[DIM][DIM];
  PetscReal G00,G01,G10,G11;
  PetscReal Res_Strong_0,Res_Strong_1,Res_Strong_2,Res_Strong_3;
  PetscReal Res_Strong0_0,Res_Strong0_1,Res_Strong0_2,Res_Strong0_3;
  PetscReal Jac_Strong_00,Jac_Strong_01,Jac_Strong_02,Jac_Strong_03,Jac_Strong_04,Jac_Strong_10,Jac_Strong_11,Jac_Strong_12,Jac_Strong_13,Jac_Strong_14,Jac_Strong_20,Jac_Strong_21,Jac_Strong_22,Jac_Strong_23,Jac_Strong_24,Jac_Strong_30,Jac_Strong_31,Jac_Strong_32,Jac_Strong_33,Jac_Strong_34;
  PetscReal TauSUPG_00,TauSUPG_01,TauSUPG_02,TauSUPG_03,TauSUPG_10,TauSUPG_11,TauSUPG_12,TauSUPG_13,TauSUPG_20,TauSUPG_21,TauSUPG_22,TauSUPG_23,TauSUPG_30,TauSUPG_31,TauSUPG_32,TauSUPG_33;
  PetscReal TauSUPGJac_00_0,TauSUPGJac_01_0,TauSUPGJac_02_0,TauSUPGJac_03_0,TauSUPGJac_10_0,TauSUPGJac_11_0,TauSUPGJac_12_0,TauSUPGJac_13_0,TauSUPGJac_20_0,TauSUPGJac_21_0,TauSUPGJac_22_0,TauSUPGJac_23_0,TauSUPGJac_30_0,TauSUPGJac_31_0,TauSUPGJac_32_0,TauSUPGJac_33_0;
  PetscReal TauSUPGJac_00_1,TauSUPGJac_01_1,TauSUPGJac_02_1,TauSUPGJac_03_1,TauSUPGJac_10_1,TauSUPGJac_11_1,TauSUPGJac_12_1,TauSUPGJac_13_1,TauSUPGJac_20_1,TauSUPGJac_21_1,TauSUPGJac_22_1,TauSUPGJac_23_1,TauSUPGJac_30_1,TauSUPGJac_31_1,TauSUPGJac_32_1,TauSUPGJac_33_1;
  PetscReal TauSUPGJac_00_2,TauSUPGJac_01_2,TauSUPGJac_02_2,TauSUPGJac_03_2,TauSUPGJac_10_2,TauSUPGJac_11_2,TauSUPGJac_12_2,TauSUPGJac_13_2,TauSUPGJac_20_2,TauSUPGJac_21_2,TauSUPGJac_22_2,TauSUPGJac_23_2,TauSUPGJac_30_2,TauSUPGJac_31_2,TauSUPGJac_32_2,TauSUPGJac_33_2;
  PetscReal TauSUPGJac_00_3,TauSUPGJac_01_3,TauSUPGJac_02_3,TauSUPGJac_03_3,TauSUPGJac_10_3,TauSUPGJac_11_3,TauSUPGJac_12_3,TauSUPGJac_13_3,TauSUPGJac_20_3,TauSUPGJac_21_3,TauSUPGJac_22_3,TauSUPGJac_23_3,TauSUPGJac_30_3,TauSUPGJac_31_3,TauSUPGJac_32_3,TauSUPGJac_33_3;
  PetscReal TauSUPGJac_00_4,TauSUPGJac_01_4,TauSUPGJac_02_4,TauSUPGJac_03_4,TauSUPGJac_10_4,TauSUPGJac_11_4,TauSUPGJac_12_4,TauSUPGJac_13_4,TauSUPGJac_20_4,TauSUPGJac_21_4,TauSUPGJac_22_4,TauSUPGJac_23_4,TauSUPGJac_30_4,TauSUPGJac_31_4,TauSUPGJac_32_4,TauSUPGJac_33_4;
  PetscReal kappa_DC,kappa_DC_C,kappa_DC_M,kappa_DC_E,kappa_hat,Q[DOF-1][DOF-1],P[DOF-1][DOF-1];
  PetscReal b_EoS,c_EoS,d_EoS;
  PetscReal p_sat,dp_sat,ddp_sat,dddp_sat;
  PetscReal e_sat_l,de_sat_l,dde_sat_l,e_sat_v,de_sat_v,dde_sat_v;
  PetscReal rho_v,drho_v,ddrho_v,rho_l,drho_l,ddrho_l,dp_drho_v,dp_drho_l,rho_m;
  PetscReal rho_v_low,rho_l_low,e_sat_v_low,e_sat_l_low;
  PetscReal rho_v_high,rho_l_high,e_sat_v_high,e_sat_l_high;
  PetscComplex s_EoS,rho_l_temp,rho_v_temp,rho_m_temp; // Intrim scalar to find cubic root
  PetscScalar u_norm2,rho,u1,u2,T,aux;
  PetscScalar u_norm2_0,rho_0,u1_0,u2_0,T_0,aux_0;
  PetscScalar rho_t,u1_t,u2_t,T_t;
  PetscScalar rho_t_0,u1_t_0,u2_t_0,T_t_0;
  PetscScalar rho_1,rho_2,u1_1,u1_2,u2_1,u2_2,T_1,T_2,aux_1,aux_2;
  PetscScalar rho_1_0,rho_2_0,u1_1_0,u1_2_0,u2_1_0,u2_2_0,T_1_0,T_2_0,aux_1_0,aux_2_0;
  PetscScalar rho_11,rho_12,rho_21,rho_22,u1_11,u1_12,u1_21,u1_22,u2_11,u2_12,u2_21,u2_22,T_11,T_12,T_21,T_22;
  PetscScalar rho_11_0,rho_12_0,rho_21_0,rho_22_0,u1_11_0,u1_12_0,u1_21_0,u1_22_0,u2_11_0,u2_12_0,u2_21_0,u2_22_0,T_11_0,T_12_0,T_21_0,T_22_0;
  PetscReal N0_a,N0_b;
  PetscReal N1_a_1,N1_a_2;
  PetscReal N1_b_1,N1_b_2;
  PetscReal N2_b_11,N2_b_12,N2_b_21,N2_b_22;
} LocalStruct;

extern LocalStruct St;
PetscErrorCode PrintInfo();
PetscErrorCode CalcValue(const AppCtx *user,PetscInt ComputeEoS,PetscScalar rho,PetscScalar T,PetscScalar *Psi);
PetscErrorCode CalcValue0(const AppCtx *user,PetscScalar rho_0,PetscScalar T_0);
PetscErrorCode GetGeometricTensor(const IGAPoint pnt);
PetscErrorCode UnpackSol(const AppCtx *user,PetscScalar *sol,PetscScalar *sol_t,PetscScalar (*grad_sol)[DIM],PetscScalar (*hess_sol)[DIM][DIM]);
PetscErrorCode UnpackSol0(const AppCtx *user,PetscScalar *sol0,PetscScalar *sol_t0,PetscScalar (*grad_sol0)[DIM],PetscScalar (*hess_sol0)[DIM][DIM]);
PetscErrorCode UnpackShape(const AppCtx *user,PetscInt Get_a,PetscInt Get_b,const PetscReal *N0,const PetscReal (*N1)[DIM],const PetscReal (*N2)[DIM][DIM]);
PetscErrorCode GetResWeak(const AppCtx *user,PetscScalar (*R)[user->dof]);
PetscErrorCode GetJacWeak(const AppCtx *user,PetscScalar (*J)[user->dof][St.nen][user->dof]);
PetscErrorCode GetResStrong(const AppCtx *user);
PetscErrorCode GetResStrong0(const AppCtx *user);
PetscErrorCode GetJacStrong(const AppCtx *user);
PetscErrorCode GetTauSUPG(const AppCtx *user);
PetscErrorCode GetTauSUPG_0(const AppCtx *user);
PetscErrorCode GetResSUPG(const AppCtx *user,PetscScalar (*R)[user->dof]);
PetscErrorCode GetJacSUPG(const AppCtx *user,PetscScalar (*J)[user->dof][St.nen][user->dof]);
PetscErrorCode GetTauSUPG_Simple(const AppCtx *user);
PetscErrorCode GetTauSUPG_Simple_0(const AppCtx *user);
PetscErrorCode GetResSUPG_Simple(const AppCtx *user,PetscScalar (*R)[user->dof]);
PetscErrorCode GetJacSUPG_Simple(const AppCtx *user,PetscScalar (*J)[user->dof][St.nen][user->dof]);
PetscErrorCode GetKappaDC(const AppCtx *user);
PetscErrorCode GetResDC(const AppCtx *user,PetscScalar (*R)[user->dof]);
PetscErrorCode GetJacDC(const AppCtx *user,PetscScalar (*J)[user->dof][St.nen][user->dof]);
PetscErrorCode GetResSpongeStrong(const AppCtx *user);
PetscErrorCode GetResSpongeStrong0(const AppCtx *user);
PetscErrorCode GetJacSpongeStrong(const AppCtx *user);
PetscErrorCode CalcSaturation(AppCtx *user,PetscReal T,PetscReal *rho_l,PetscReal *rho_v,PetscReal *rho_m,PetscReal *e_sat_l,PetscReal *e_sat_v);
PetscErrorCode TIMModifyEoS(AppCtx *user,PetscInt Phase);
PetscErrorCode TIMModifyEoS_0(AppCtx *user,PetscInt Phase);
PetscErrorCode Calc_Inter_dp(AppCtx *user);
PetscErrorCode Calc_Inter_dp0(AppCtx *user);
PetscErrorCode Calc_Inter_ddp(AppCtx *user);

// TIMFunc
PetscErrorCode TIMModification(AppCtx *user);

// MatrixOperation
PetscErrorCode viewmatrix(PetscInt m,PetscInt n, PetscReal A[m][n]);
PetscErrorCode DenmanBeavers(const AppCtx *user);

// Utility
PetscErrorCode SpongeZone(AppCtx *user);
PetscErrorCode CalcViscosity(AppCtx *user);
PetscErrorCode CalcC_DC(AppCtx *user);
PetscErrorCode GetGeomtry(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx);
PetscErrorCode NSKMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx);
PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm,PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *ctx);
PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal RelativeTime,Vec U,void *ctx);
PetscBool file_exist (char *filename);

// IC
PetscErrorCode L2Projection_IC(IGAPoint pnt,PetscScalar *KK,PetscScalar *FF,void *ctx);

// Residual and Jacobian
PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Re,void *ctx);
PetscErrorCode Jacobian(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscReal t0,const PetscScalar *U0, const PetscScalar *V0,
                        PetscScalar *Je,void *ctx);