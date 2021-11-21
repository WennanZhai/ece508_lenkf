#ifndef LENKF_CONFIG_OPTS_H
#define LENKF_CONFIG_OPTS_H

#include <confuse.h>

static cfg_opt_t lenkf_config_opts[] = {
  CFG_STR("dir", "./", CFGF_NONE),
  CFG_INT("rank", 0, CFGF_NONE),
  CFG_INT_LIST("n", "{0}", CFGF_NONE),
  
  CFG_INT("N", 0, CFGF_NONE),
  CFG_INT("M_block", 0, CFGF_NONE),
  CFG_INT("T", 0, CFGF_NONE),
  CFG_INT("L", 0, CFGF_NONE),

  CFG_INT("seed", -1, CFGF_NONE),
  
  CFG_STR("x_hat_filename", "", CFGF_NONE),
  CFG_STR("trace_filename", "", CFGF_NONE),

  CFG_STR("x0_filename", "", CFGF_NONE),
  CFG_STR("PI_sqrt_filename", "", CFGF_NONE),
  CFG_FLOAT("alpha_PI_sqrt", 1, CFGF_NONE),
  
  CFG_STR("y_list_filename", "", CFGF_NONE),
  CFG_STR("H_list_filename", "", CFGF_NONE),
  CFG_STR("R_sqrt_list_filename", "", CFGF_NONE),

  CFG_STR("D_filename", "", CFGF_NONE),
  CFG_FLOAT("lambda", 0, CFGF_NONE),
  
  CFG_STR("C_filename", "", CFGF_NONE),

  CFG_STR("F_filename", "", CFGF_NONE),
  CFG_STR("Q_sqrt_filename", "", CFGF_NONE),
  
  CFG_FLOAT("update_epsilon", 0, CFGF_NONE),
  
  CFG_BOOL("poisson_noise", cfg_false, CFGF_NONE),
  CFG_FLOAT("poisson_eps", 1, CFGF_NONE),
  
  CFG_BOOL("regularize", cfg_false, CFGF_NONE),
  CFG_BOOL("F_equal_I", cfg_false, CFGF_NONE),
  CFG_BOOL("randn_conv", cfg_false, CFGF_NONE),
  
  CFG_BOOL("quiet_mode", cfg_false, CFGF_NONE),
  CFG_BOOL("save_trace", cfg_false, CFGF_NONE),
  CFG_BOOL("save_intermediate", cfg_false, CFGF_NONE),
  CFG_BOOL("randn_debug", cfg_false, CFGF_NONE),
  CFG_BOOL("lenkf_debug", cfg_false, CFGF_NONE),
  CFG_BOOL("enable_profiling", cfg_false, CFGF_NONE),

  CFG_END()
};

#endif
