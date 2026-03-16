/**
 * @file wandb_c.h
 * @brief C interface for calling Weights & Biases (wandb) from Fortran.
 *
 * This file has been generated using LLM AI code agents and has not yet been thoroughly tested.
 *
 * This header declares the C functions that embed the Python interpreter
 * and call wandb through the Python C API.  The Fortran module
 * `athena_wandb` binds to these symbols via iso_c_binding.
 */
#ifndef WANDB_C_H
#define WANDB_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialise a wandb run.
 *
 * Starts the embedded Python interpreter (if not already running),
 * imports wandb, and calls wandb.init().
 *
 * @param project  Project name (required, null-terminated C string).
 * @param name     Run name (may be NULL or "" for auto-generated name).
 * @param entity   wandb entity / team (may be NULL or "" for default).
 * @return 0 on success, non-zero on failure.
 */
int wandb_init_c(const char *project, const char *name, const char *entity);

/**
 * Log a single scalar metric.
 *
 * @param key   Metric name (null-terminated C string).
 * @param value Metric value.
 * @param step  Global step.  Pass -1 to let wandb auto-increment.
 */
void wandb_log_metric_c(const char *key, double value, int step);

/**
 * Log multiple scalar metrics in one call.
 *
 * @param keys   Array of metric names (null-terminated C strings).
 * @param values Array of metric values (same length as keys).
 * @param count  Number of metrics.
 * @param step   Global step.  Pass -1 to let wandb auto-increment.
 */
void wandb_log_metrics_c(const char **keys, const double *values,
                          int count, int step);

/**
 * Set an integer config value.
 */
void wandb_config_set_int_c(const char *key, int value);

/**
 * Set a floating-point config value.
 */
void wandb_config_set_real_c(const char *key, double value);

/**
 * Set a string config value.
 */
void wandb_config_set_str_c(const char *key, const char *value);

/**
 * Finish the current wandb run and shut down the embedded interpreter.
 */
void wandb_finish_c(void);

#ifdef __cplusplus
}
#endif

#endif /* WANDB_C_H */
