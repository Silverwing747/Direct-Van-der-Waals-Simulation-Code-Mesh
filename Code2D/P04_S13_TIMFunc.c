#include "P04_S13_header.h"

PetscErrorCode TIMModification(AppCtx *user)
{
  PetscInt Phase = 1; // use vapor phase for TIM energy
  // Implicit part
  user->dp_drho_original = St.dp_drho;
  if (user->energy){ // Re-compute saturation condition and their derivatives using FD
    CalcSaturation(user,St.T-1e-4,&St.rho_l_low,&St.rho_v_low,&St.rho_m,&St.e_sat_l_low,&St.e_sat_v_low); // Perturb low temp
    CalcSaturation(user,St.T+1e-4,&St.rho_l_high,&St.rho_v_high,&St.rho_m,&St.e_sat_l_high,&St.e_sat_v_high); // Perturb high temp
    CalcSaturation(user,St.T,&St.rho_l,&St.rho_v,&St.rho_m,&St.e_sat_l,&St.e_sat_v); // Current

    // Compute first order derivative
    St.drho_v = (St.rho_v_high - St.rho_v_low) / 2e-4;
    St.drho_l = (St.rho_l_high - St.rho_l_low) / 2e-4;
    St.de_sat_v = (St.e_sat_v_high - St.e_sat_v_low) / 2e-4;
    St.de_sat_l = (St.e_sat_l_high - St.e_sat_l_low) / 2e-4;

    // Compute second order derivative
    St.ddrho_v = (St.rho_v_high - 2.0 * St.rho_v + St.rho_v_low) / 1e-8;
    St.ddrho_l = (St.rho_l_high - 2.0 * St.rho_l + St.rho_l_low) / 1e-8;
    St.dde_sat_v = (St.e_sat_v_high - 2.0 * St.e_sat_v + St.e_sat_v_low) / 1e-8;
    St.dde_sat_l = (St.e_sat_l_high - 2.0 * St.e_sat_l + St.e_sat_l_low) / 1e-8;
  }
  if ((St.T < 1.0) && user->TIM){ // If temperature higher than critical temperature, do no TIM
    if (St.rho > St.rho_v && St.rho <= St.rho_l) TIMModifyEoS(user,Phase);
    if (user->TIM == 2){ // TIM2
      PetscReal p_mod = 0.0,dp_drho_mod = 0.0,ddp_drho2_mod = 0.0; 
      if (St.rho < St.rho_v) CalcModEoS(user,0,St.rho,St.rho_v,St.dp_drho_v,&p_mod,&dp_drho_mod,&ddp_drho2_mod);
      else if (St.rho > St.rho_l) CalcModEoS(user,1,St.rho,St.rho_l,St.dp_drho_l,&p_mod,&dp_drho_mod,&ddp_drho2_mod);
      St.p = St.p + p_mod;
      St.dp_drho = St.dp_drho + dp_drho_mod;
      St.ddp_drho2 = St.ddp_drho2 + ddp_drho2_mod;
    }
  }

  //Explicit part
  user->dp_drho_original_0 = St.dp_drho_0;
  if (user->energy){ // Re-compute saturation condition and their derivatives using FD
    CalcSaturation(user,St.T_0-1e-4,&St.rho_l_low,&St.rho_v_low,&St.rho_m,&St.e_sat_l_low,&St.e_sat_v_low); // Perturb low temp
    CalcSaturation(user,St.T_0+1e-4,&St.rho_l_high,&St.rho_v_high,&St.rho_m,&St.e_sat_l_high,&St.e_sat_v_high); // Perturb high temp
    CalcSaturation(user,St.T_0,&St.rho_l,&St.rho_v,&St.rho_m,&St.e_sat_l,&St.e_sat_v); // Current

    // Compute first order derivative
    St.drho_v = (St.rho_v_high - St.rho_v_low) / 2e-4;
    St.drho_l = (St.rho_l_high - St.rho_l_low) / 2e-4;
    St.de_sat_v = (St.e_sat_v_high - St.e_sat_v_low) / 2e-4;
    St.de_sat_l = (St.e_sat_l_high - St.e_sat_l_low) / 2e-4;

    // Compute second order derivative
    St.ddrho_v = (St.rho_v_high - 2.0 * St.rho_v + St.rho_v_low) / 1e-8;
    St.ddrho_l = (St.rho_l_high - 2.0 * St.rho_l + St.rho_l_low) / 1e-8;
    St.dde_sat_v = (St.e_sat_v_high - 2.0 * St.e_sat_v + St.e_sat_v_low) / 1e-8;
    St.dde_sat_l = (St.e_sat_l_high - 2.0 * St.e_sat_l + St.e_sat_l_low) / 1e-8;
  }
  if ((St.T_0 < 1.0) && user->TIM){ // If temperature higher than critical temperature, do no TIM
    if (St.rho_0 > St.rho_v && St.rho_0 <= St.rho_l) TIMModifyEoS_0(user,Phase); 
    if (user->TIM == 2){ // TIM2
      PetscReal p_mod_0 = 0.0,dp_drho_mod_0 = 0.0,ddp_drho2_mod_0 = 0.0; 
      if (St.rho_0 < St.rho_v) CalcModEoS(user,0,St.rho_0,St.rho_v,St.dp_drho_v,&p_mod_0,&dp_drho_mod_0,&ddp_drho2_mod_0);
      else if (St.rho_0 > St.rho_l) CalcModEoS(user,1,St.rho_0,St.rho_l,St.dp_drho_l,&p_mod_0,&dp_drho_mod_0,&ddp_drho2_mod_0);
      St.p_0 = St.p_0 + p_mod_0;
      St.dp_drho_0 = St.dp_drho_0 + dp_drho_mod_0;
      St.ddp_drho2_0 = St.ddp_drho2_0 + ddp_drho2_mod_0;
    }
  }
}

PetscErrorCode CalcModEoS(AppCtx *user,PetscBool Phase,PetscScalar rho,PetscReal rho_s,PetscReal dp_s,PetscReal *p_mod,PetscReal *dp_mod,PetscReal *ddp_mod)
{
  // Phase = 0 -> vapor; Phase = 1 -> Liquid
  PetscReal   F = user->F;
  PetscReal   d_l = user->d_l;
  PetscReal   d_v = user->d_v;
  if (!Phase){
  PetscReal   a_v = -((d_v*d_v)*dp_s*rho_s*(F-1.0))/F;
  PetscReal   b_v = (d_v*dp_s*(F-1.0))/F;
  PetscReal   c_v = -(d_v*dp_s*(F-1.0)*(log(rho_s)+d_v*log(rho_s)-d_v*log(1.0/d_v)))/(F*(d_v+1.0));
  if (p_mod)  *p_mod = b_v*rho-(a_v*rho)/(rho-rho_s*(d_v+1.0));
  if (dp_mod)  *dp_mod = b_v-a_v/(rho-rho_s*(d_v+1.0))+a_v*rho*1.0/pow(rho-rho_s*(d_v+1.0),2.0);
  if (ddp_mod)  *ddp_mod = a_v*1.0/pow(rho-rho_s*(d_v+1.0),2.0)*2.0-a_v*rho*1.0/pow(rho-rho_s*(d_v+1.0),3.0)*2.0;
  }else{
  PetscReal   a_l = ((d_l*d_l)*dp_s*rho_s*(F-1.0))/F;
  PetscReal   b_l = -(d_l*dp_s*(F-1.0))/F;
  PetscReal   c_l = -(d_l*dp_s*(F-1.0)*(log(rho_s)-d_l*log(rho_s)+d_l*log(1.0/d_l)))/(F*(d_l-1.0));
  if (p_mod)  *p_mod = b_l*rho+(a_l*rho)/(rho+rho_s*(d_l-1.0));
  if (dp_mod)  *dp_mod = b_l+a_l/(rho+rho_s*(d_l-1.0))-a_l*rho*1.0/pow(rho+rho_s*(d_l-1.0),2.0);
  if (ddp_mod)  *ddp_mod = a_l*1.0/pow(rho+rho_s*(d_l-1.0),2.0)*-2.0+a_l*rho*1.0/pow(rho+rho_s*(d_l-1.0),3.0)*2.0;
  }
}