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
 * @param project   Project name (required, null-terminated C string).
 * @param name      Run name (may be NULL or "" for auto-generated name).
 * @param entity    wandb entity / team (may be NULL or "" for default).
 * @param sweep_id  Sweep ID returned by wandb_sweep_c.  Pass NULL or "" when
 *                  not running within a sweep.  When non-empty, the env var
 *                  WANDB_SWEEP_ID is set before wandb.init() is called so
 *                  that the run joins the sweep and receives sampled config.
 * @return 0 on success, non-zero on failure.
 */
int wandb_init_c(const char *project, const char *name,
                 const char *entity,  const char *sweep_id);

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

/* ------------------------------------------------------------------ */
/*  Config read-back (sweep parameters)                                */
/* ------------------------------------------------------------------ */

/**
 * Read an integer config value set by the sweep agent.
 *
 * @param key           Config key (null-terminated).
 * @param default_value Returned when the key is absent or the run is not init'd.
 * @return The integer value from wandb.config, or default_value.
 */
int    wandb_config_get_int_c (const char *key, int    default_value);

/**
 * Read a floating-point config value set by the sweep agent.
 *
 * @param key           Config key (null-terminated).
 * @param default_value Returned when the key is absent or the run is not init'd.
 * @return The double value from wandb.config, or default_value.
 */
double wandb_config_get_real_c(const char *key, double default_value);

/**
 * Read a string config value set by the sweep agent.
 *
 * Copies at most buf_len-1 characters into buf and always null-terminates.
 * If the key is absent the buffer is left unchanged.
 *
 * @param key     Config key (null-terminated).
 * @param buf     Output buffer.
 * @param buf_len Size of buf in bytes (including the null terminator).
 * @return 1 if the key was found, 0 otherwise.
 */
int    wandb_config_get_str_c (const char *key, char *buf, int buf_len);

/* ------------------------------------------------------------------ */
/*  Sweep orchestration                                                 */
/* ------------------------------------------------------------------ */

/**
 * Register a hyperparameter sweep with wandb.
 *
 * Calls wandb.sweep(config, project=project, entity=entity) and writes
 * the resulting sweep_id into sweep_id_buf (null-terminated, up to
 * sweep_id_buf_len bytes).
 *
 * The config argument should be a Python-dict-compatible JSON-like
 * string with keys "method", "metric", and "parameters", for example:
 *
 *   {"method":"bayes","metric":{"name":"loss","goal":"minimize"},
 *    "parameters":{"lr":{"min":0.0001,"max":0.1},
 *                  "hidden":{"values":[16,32,64]}}}
 *
 * @param config_json       JSON string describing the sweep config.
 * @param project           Project name.
 * @param entity            Entity/team name (may be "" for default).
 * @param sweep_id_buf      Output buffer for the sweep ID string.
 * @param sweep_id_buf_len  Size of sweep_id_buf.
 * @return 0 on success, non-zero on failure.
 */
int wandb_sweep_c(const char *config_json,
                  const char *project,
                  const char *entity,
                  char       *sweep_id_buf,
                  int         sweep_id_buf_len);

/**
 * Start a wandb sweep agent in a background Python thread.
 *
 * The agent runs a built-in callback that calls wandb.init() to receive
 * sweep-sampled hyperparameters, writes them to an internal buffer, then
 * waits for wandb_sweep_run_done_c() before calling wandb.finish() and
 * requesting the next run from the sweep controller.
 *
 * Call sequence per sweep:
 *   1. wandb_sweep_c(...)          -- register sweep, get sweep_id
 *   2. wandb_sweep_start_agent_c(sweep_id, project, entity, count)
 *   3. for i = 1 .. count:
 *        a. wandb_sweep_params_c(buf, buf_len, timeout_s) -- get params JSON
 *        b. parse params, train, log metrics via wandb_log_metric_c
 *        c. wandb_sweep_run_done_c()                      -- signal done
 *   4. wandb_shutdown_c()
 *
 * @param sweep_id  Sweep ID from wandb_sweep_c.
 * @param project   Project name (may be "").
 * @param entity    Entity/team name (may be "").
 * @param count     Total number of runs for this agent (> 0).
 * @return 0 on success, non-zero on failure.
 */
int wandb_sweep_start_agent_c(const char *sweep_id,
                               const char *project,
                               const char *entity,
                               int         count);

/**
 * Block until the sweep agent callback has called wandb.init() and
 * produced a new set of sampled hyperparameters.
 *
 * Copies the params as a JSON object string into buf (null-terminated).
 * Also internally updates wandb_run so that wandb_log / wandb_config_set
 * calls are routed to the current sweep run.
 *
 * @param buf       Output buffer for the JSON params string.
 * @param buf_len   Size of buf in bytes.
 * @param timeout_s Seconds to wait before giving up (e.g. 120.0).
 * @return 1 if params were received, 0 on timeout or error.
 */
int wandb_sweep_params_c(char *buf, int buf_len, double timeout_s);

/**
 * Signal that Fortran has finished one sweep run.
 *
 * The agent callback will then call wandb.finish() and the agent will
 * ask the sweep controller for the next run's hyperparameters.
 */
void wandb_sweep_run_done_c(void);

/**
 * Run a wandb sweep agent for count runs (blocking, legacy).
 *
 * Calls wandb.agent(sweep_id, function=None, count=count, ...).
 * Prefer the non-blocking wandb_sweep_start_agent_c / wandb_sweep_params_c /
 * wandb_sweep_run_done_c API instead.
 *
 * @param sweep_id  Sweep ID returned by wandb_sweep_c.
 * @param project   Project name (may be "").
 * @param entity    Entity/team name (may be "").
 * @param count     Number of runs to execute (0 = run until the sweep is done).
 * @return 0 on success, non-zero on failure.
 */
int wandb_agent_c(const char *sweep_id,
                  const char *project,
                  const char *entity,
                  int         count);

/**
 * Finish the current wandb run but keep the Python interpreter alive.
 * Call this between sweep runs when NOT using the agent threading API.
 */
void wandb_finish_c(void);

/**
 * Shut down the Python interpreter.  Call once after all runs are complete.
 * After this call no other wandb_*_c function may be used.
 */
void wandb_shutdown_c(void);

#ifdef __cplusplus
}
#endif

#endif /* WANDB_C_H */
