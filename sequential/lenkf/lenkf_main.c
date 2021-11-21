#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>
#include <confuse.h>

#include "lenkf.h"
#include "lenkf_config_opts.h"
#include "randn.h"


static void usage(char *arg0);
static void parse_config_file(char *config_file, lenkf_config *config);
static lenkf_config *process_args_and_config_file(int argc, char **argv);

/******************************************************************************/

static void usage(char *arg0) {
  printf("usage: %s <-h> <-v> <-p> [config file]\n", arg0);
  printf("-v: Verbose mode - print all output\n");
  printf("-p: Enable profiling output\n");
}

void parse_config_file(char *config_file, lenkf_config *config) {
  cfg_t *cfg;
  int r;

  assert(config_file);
  assert(config);
  assert(cfg_true == True && cfg_false == False);


  cfg = cfg_init(lenkf_config_opts, CFGF_NONE);

  r = cfg_parse(cfg, config_file);
  assert(r == CFG_SUCCESS);

  lenkf_config_get_options(cfg, config);
  lenkf_config_check(cfg, config);

  cfg_free(cfg);
}

lenkf_config *process_args_and_config_file(int argc, char **argv) {
  boolean verbose_flag, profile_flag;
  lenkf_config *config;
  char *config_filename;

  int opt;
  char optstring[] = "hvp";

  assert(argv);
  assert(*argv);

  verbose_flag = False;
  profile_flag = False;
  while ((opt = getopt(argc, argv, optstring)) != -1) {
    switch (opt) {
    case 'h':
      usage(argv[0]);
      exit(0);
      break;
    case 'v':
      verbose_flag = True;
      break;
    case 'p':
      profile_flag = True;
      break;
    default:
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (argc - optind != 1) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  else {
    config_filename = argv[optind+0];
  }

  config = lenkf_config_init();
  parse_config_file(config_filename, config);

  config->quiet_mode &= !verbose_flag;
  if (profile_flag) config->enable_profiling = cfg_true;

  return config;
}

/******************************************************************************/

int main(int argc, char **argv) {
  lenkf_config *config;
  lenkf_stats stats;
  full_c *x_mean_final;


  config = process_args_and_config_file(argc, argv);

  randn_init();
  if (config->seed >= 1) {
    randn_seed(config->seed);
  }

  x_mean_final = full_c_create(config->N, config->T);

  lenkf(x_mean_final, config, &stats);

  if (!config->save_intermediate) {
    full_c_export(config->x_hat_filename, x_mean_final);
  }

  full_c_destroy(&x_mean_final);
  randn_exit();

  if (!config->quiet_mode) {
    printf("\n");
    printf("Elapsed time: ");
    printf_hms(stats.exec_time);
    printf("\n");
    printf("Res: %.1f Mb\n", stats.res);
  }

  lenkf_config_destroy(&config);

  return EXIT_SUCCESS;
}
