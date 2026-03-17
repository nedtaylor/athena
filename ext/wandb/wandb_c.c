/**
 * @file wandb_c.c
 * @brief C implementation that embeds Python to call wandb.
 *
 * This file has been generated using LLM AI code agents and has not yet been thoroughly tested.
 *
 * Uses the Python C API to:
 *   1. Start an embedded Python interpreter.
 *   2. Import the `wandb` package.
 *   3. Forward init / log / config / finish calls to wandb.
 */
#include "wandb_c.h"

/* Python.h must come before any standard headers on some platforms. */
#include <Python.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Module-level state                                                 */
/* ------------------------------------------------------------------ */
static int         python_started = 0;   /* 1 after Py_Initialize()  */
static PyObject   *wandb_module   = NULL;
static PyObject   *wandb_run      = NULL;

/* Sweep-agent threading state (lives in '__athena_sweep__' module dict) */
static PyObject   *sweep_globals  = NULL;  /* dict holding threading objects */


/* ------------------------------------------------------------------ */
/*  Helper: print the current Python exception, then clear it.         */
/* ------------------------------------------------------------------ */
static void print_py_error(const char *ctx)
{
    if (PyErr_Occurred()) {
        fprintf(stderr, "[wandb_c] Python error in %s:\n", ctx);
        PyErr_Print();
        PyErr_Clear();
    }
}


/* ------------------------------------------------------------------ */
/*  wandb_init_c                                                       */
/* ------------------------------------------------------------------ */
int wandb_init_c(const char *project, const char *name,
                 const char *entity,  const char *sweep_id)
{
    PyObject *init_fn   = NULL;
    PyObject *kwargs    = NULL;
    PyObject *result    = NULL;
    int       rc        = -1;

    /* Start the interpreter if needed. */
    if (!python_started) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            fprintf(stderr,
                    "[wandb_c] Failed to initialise the Python interpreter.\n");
            return -1;
        }
        python_started = 1;
    }

    /* If a sweep_id was supplied, set WANDB_SWEEP_ID so that wandb.init()
     * joins the sweep and receives the sweep-sampled hyperparameters. */
    if (sweep_id && sweep_id[0] != '\0') {
        PyObject *os_mod = PyImport_ImportModule("os");
        if (os_mod) {
            PyObject *environ = PyObject_GetAttrString(os_mod, "environ");
            if (environ) {
                PyObject *py_sid = PyUnicode_FromString(sweep_id);
                if (py_sid) {
                    PyObject_SetItem(environ,
                                     PyUnicode_FromString("WANDB_SWEEP_ID"),
                                     py_sid);
                    Py_DECREF(py_sid);
                }
                Py_DECREF(environ);
            }
            Py_DECREF(os_mod);
        }
    }

    /* Import wandb. */
    printf("[wandb_c] Importing wandb module...\n");
    if (!wandb_module) {
        wandb_module = PyImport_ImportModule("wandb");
        if (!wandb_module) {
            print_py_error("import wandb");
            return -1;
        }
    }
    printf("[wandb_c] wandb module imported successfully.\n");

    /* Build keyword arguments for wandb.init(). */
    kwargs = PyDict_New();
    if (!kwargs) { print_py_error("PyDict_New"); goto cleanup; }

    {
        PyObject *py_project = PyUnicode_FromString(project);
        if (!py_project) { print_py_error("project string"); goto cleanup; }
        PyDict_SetItemString(kwargs, "project", py_project);
        Py_DECREF(py_project);
    }

    if (name && name[0] != '\0') {
        PyObject *py_name = PyUnicode_FromString(name);
        if (!py_name) { print_py_error("name string"); goto cleanup; }
        PyDict_SetItemString(kwargs, "name", py_name);
        Py_DECREF(py_name);
    }

    if (entity && entity[0] != '\0') {
        PyObject *py_entity = PyUnicode_FromString(entity);
        if (!py_entity) { print_py_error("entity string"); goto cleanup; }
        PyDict_SetItemString(kwargs, "entity", py_entity);
        Py_DECREF(py_entity);
    }

    /* Call wandb.init(**kwargs). */
    init_fn = PyObject_GetAttrString(wandb_module, "init");
    if (!init_fn) { print_py_error("wandb.init lookup"); goto cleanup; }

    result = PyObject_Call(init_fn, PyTuple_New(0), kwargs);
    if (!result) { print_py_error("wandb.init call"); goto cleanup; }

    /* Keep a reference to the run object. */
    Py_XDECREF(wandb_run);
    wandb_run = result;
    result    = NULL;  /* ownership transferred */
    rc        = 0;

cleanup:
    Py_XDECREF(result);
    Py_XDECREF(init_fn);
    Py_XDECREF(kwargs);
    return rc;
}


/* ------------------------------------------------------------------ */
/*  wandb_log_metric_c                                                 */
/* ------------------------------------------------------------------ */
void wandb_log_metric_c(const char *key, double value, int step)
{
    PyObject *log_fn  = NULL;
    PyObject *dict    = NULL;
    PyObject *kwargs  = NULL;
    PyObject *result  = NULL;

    if (!wandb_module) return;

    log_fn = PyObject_GetAttrString(wandb_module, "log");
    if (!log_fn) { print_py_error("wandb.log lookup"); return; }

    dict = PyDict_New();
    if (!dict) { print_py_error("PyDict_New"); goto done; }
    {
        PyObject *py_val = PyFloat_FromDouble(value);
        PyDict_SetItemString(dict, key, py_val);
        Py_DECREF(py_val);
    }

    kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "data", dict);  /* wandb.log(data=...) */
    if (step >= 0) {
        PyObject *py_step = PyLong_FromLong((long)step);
        PyDict_SetItemString(kwargs, "step", py_step);
        Py_DECREF(py_step);
    }

    result = PyObject_Call(log_fn, PyTuple_New(0), kwargs);
    if (!result) print_py_error("wandb.log call");

done:
    Py_XDECREF(result);
    Py_XDECREF(kwargs);
    Py_XDECREF(dict);
    Py_XDECREF(log_fn);
}


/* ------------------------------------------------------------------ */
/*  wandb_log_metrics_c                                                */
/* ------------------------------------------------------------------ */
void wandb_log_metrics_c(const char **keys, const double *values,
                          int count, int step)
{
    PyObject *log_fn  = NULL;
    PyObject *dict    = NULL;
    PyObject *kwargs  = NULL;
    PyObject *result  = NULL;
    int       i;

    if (!wandb_module || count <= 0) return;

    log_fn = PyObject_GetAttrString(wandb_module, "log");
    if (!log_fn) { print_py_error("wandb.log lookup"); return; }

    dict = PyDict_New();
    for (i = 0; i < count; i++) {
        PyObject *py_val = PyFloat_FromDouble(values[i]);
        PyDict_SetItemString(dict, keys[i], py_val);
        Py_DECREF(py_val);
    }

    kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "data", dict);
    if (step >= 0) {
        PyObject *py_step = PyLong_FromLong((long)step);
        PyDict_SetItemString(kwargs, "step", py_step);
        Py_DECREF(py_step);
    }

    result = PyObject_Call(log_fn, PyTuple_New(0), kwargs);
    if (!result) print_py_error("wandb.log call");

    Py_XDECREF(result);
    Py_XDECREF(kwargs);
    Py_XDECREF(dict);
    Py_XDECREF(log_fn);
}


/* ------------------------------------------------------------------ */
/*  wandb_config_set_int_c                                             */
/* ------------------------------------------------------------------ */
void wandb_config_set_int_c(const char *key, int value)
{
    PyObject *config = NULL;
    PyObject *update = NULL;
    PyObject *dict   = NULL;
    PyObject *result = NULL;

    if (!wandb_run) return;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { print_py_error("run.config"); return; }

    update = PyObject_GetAttrString(config, "update");
    if (!update) { print_py_error("config.update"); goto done; }

    dict = PyDict_New();
    {
        PyObject *py_val = PyLong_FromLong((long)value);
        PyDict_SetItemString(dict, key, py_val);
        Py_DECREF(py_val);
    }

    result = PyObject_CallFunctionObjArgs(update, dict, NULL);
    if (!result) print_py_error("config.update call");

done:
    Py_XDECREF(result);
    Py_XDECREF(dict);
    Py_XDECREF(update);
    Py_XDECREF(config);
}


/* ------------------------------------------------------------------ */
/*  wandb_config_set_real_c                                            */
/* ------------------------------------------------------------------ */
void wandb_config_set_real_c(const char *key, double value)
{
    PyObject *config = NULL;
    PyObject *update = NULL;
    PyObject *dict   = NULL;
    PyObject *result = NULL;

    if (!wandb_run) return;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { print_py_error("run.config"); return; }

    update = PyObject_GetAttrString(config, "update");
    if (!update) { print_py_error("config.update"); goto done; }

    dict = PyDict_New();
    {
        PyObject *py_val = PyFloat_FromDouble(value);
        PyDict_SetItemString(dict, key, py_val);
        Py_DECREF(py_val);
    }

    result = PyObject_CallFunctionObjArgs(update, dict, NULL);
    if (!result) print_py_error("config.update call");

done:
    Py_XDECREF(result);
    Py_XDECREF(dict);
    Py_XDECREF(update);
    Py_XDECREF(config);
}


/* ------------------------------------------------------------------ */
/*  wandb_config_set_str_c                                             */
/* ------------------------------------------------------------------ */
void wandb_config_set_str_c(const char *key, const char *value)
{
    PyObject *config = NULL;
    PyObject *update = NULL;
    PyObject *dict   = NULL;
    PyObject *result = NULL;

    if (!wandb_run) return;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { print_py_error("run.config"); return; }

    update = PyObject_GetAttrString(config, "update");
    if (!update) { print_py_error("config.update"); goto done; }

    dict = PyDict_New();
    {
        PyObject *py_val = PyUnicode_FromString(value);
        PyDict_SetItemString(dict, key, py_val);
        Py_DECREF(py_val);
    }

    result = PyObject_CallFunctionObjArgs(update, dict, NULL);
    if (!result) print_py_error("config.update call");

done:
    Py_XDECREF(result);
    Py_XDECREF(dict);
    Py_XDECREF(update);
    Py_XDECREF(config);
}


/* ------------------------------------------------------------------ */
/*  wandb_config_get_int_c                                             */
/* ------------------------------------------------------------------ */
int wandb_config_get_int_c(const char *key, int default_value)
{
    PyObject *config = NULL;
    PyObject *val    = NULL;
    int       result = default_value;

    if (!wandb_run) return default_value;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { PyErr_Clear(); return default_value; }

    val = PyObject_GetItem(config, PyUnicode_FromString(key));
    if (!val) {
        PyErr_Clear();
    } else {
        if (PyLong_Check(val))
            result = (int)PyLong_AsLong(val);
        else if (PyFloat_Check(val))
            result = (int)PyFloat_AsDouble(val);
        Py_DECREF(val);
    }

    Py_DECREF(config);
    return result;
}


/* ------------------------------------------------------------------ */
/*  wandb_config_get_real_c                                            */
/* ------------------------------------------------------------------ */
double wandb_config_get_real_c(const char *key, double default_value)
{
    PyObject *config = NULL;
    PyObject *val    = NULL;
    double    result = default_value;

    if (!wandb_run) return default_value;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { PyErr_Clear(); return default_value; }

    val = PyObject_GetItem(config, PyUnicode_FromString(key));
    if (!val) {
        PyErr_Clear();
    } else {
        if (PyFloat_Check(val))
            result = PyFloat_AsDouble(val);
        else if (PyLong_Check(val))
            result = (double)PyLong_AsLong(val);
        Py_DECREF(val);
    }

    Py_DECREF(config);
    return result;
}


/* ------------------------------------------------------------------ */
/*  wandb_config_get_str_c                                             */
/* ------------------------------------------------------------------ */
int wandb_config_get_str_c(const char *key, char *buf, int buf_len)
{
    PyObject   *config  = NULL;
    PyObject   *val     = NULL;
    const char *s       = NULL;
    int         found   = 0;

    if (!wandb_run || !buf || buf_len <= 0) return 0;

    config = PyObject_GetAttrString(wandb_run, "config");
    if (!config) { PyErr_Clear(); return 0; }

    val = PyObject_GetItem(config, PyUnicode_FromString(key));
    if (!val) {
        PyErr_Clear();
    } else {
        if (PyUnicode_Check(val)) {
            s = PyUnicode_AsUTF8(val);
            if (s) {
                strncpy(buf, s, (size_t)(buf_len - 1));
                buf[buf_len - 1] = '\0';
                found = 1;
            }
        }
        Py_DECREF(val);
    }

    Py_DECREF(config);
    return found;
}


/* ------------------------------------------------------------------ */
/*  wandb_sweep_c                                                      */
/* ------------------------------------------------------------------ */
int wandb_sweep_c(const char *config_json,
                  const char *project,
                  const char *entity,
                  char       *sweep_id_buf,
                  int         sweep_id_buf_len)
{
    PyObject *json_mod   = NULL;
    PyObject *loads_fn   = NULL;
    PyObject *cfg_dict   = NULL;
    PyObject *sweep_fn   = NULL;
    PyObject *kwargs     = NULL;
    PyObject *py_str     = NULL;
    PyObject *sweep_id   = NULL;
    const char *id_str   = NULL;
    int        rc        = -1;

    /* Ensure Python + wandb are initialised (reuse wandb_init path). */
    if (!python_started) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            fprintf(stderr,
                    "[wandb_c] Failed to initialise the Python interpreter.\n");
            return -1;
        }
        python_started = 1;
    }
    if (!wandb_module) {
        wandb_module = PyImport_ImportModule("wandb");
        if (!wandb_module) { print_py_error("import wandb"); return -1; }
    }

    /* Parse the JSON config string into a Python dict. */
    json_mod = PyImport_ImportModule("json");
    if (!json_mod) { print_py_error("import json"); return -1; }

    loads_fn = PyObject_GetAttrString(json_mod, "loads");
    Py_DECREF(json_mod);
    if (!loads_fn) { print_py_error("json.loads"); return -1; }

    py_str = PyUnicode_FromString(config_json);
    cfg_dict = PyObject_CallFunctionObjArgs(loads_fn, py_str, NULL);
    Py_DECREF(py_str);
    Py_DECREF(loads_fn);
    if (!cfg_dict) { print_py_error("json.loads call"); return -1; }

    /* Build keyword arguments for wandb.sweep(). */
    kwargs = PyDict_New();
    if (!kwargs) { Py_DECREF(cfg_dict); return -1; }

    if (project && project[0] != '\0') {
        PyObject *p = PyUnicode_FromString(project);
        PyDict_SetItemString(kwargs, "project", p);
        Py_DECREF(p);
    }
    if (entity && entity[0] != '\0') {
        PyObject *e = PyUnicode_FromString(entity);
        PyDict_SetItemString(kwargs, "entity", e);
        Py_DECREF(e);
    }

    /* Call wandb.sweep(cfg_dict, **kwargs). */
    sweep_fn = PyObject_GetAttrString(wandb_module, "sweep");
    if (!sweep_fn) { print_py_error("wandb.sweep lookup"); goto cleanup; }

    {
        PyObject *args = PyTuple_Pack(1, cfg_dict);
        sweep_id = PyObject_Call(sweep_fn, args, kwargs);
        Py_DECREF(args);
    }
    if (!sweep_id) { print_py_error("wandb.sweep call"); goto cleanup; }

    /* Copy the sweep ID string into the caller's buffer. */
    if (PyUnicode_Check(sweep_id)) {
        id_str = PyUnicode_AsUTF8(sweep_id);
        if (id_str && sweep_id_buf && sweep_id_buf_len > 0) {
            strncpy(sweep_id_buf, id_str, (size_t)(sweep_id_buf_len - 1));
            sweep_id_buf[sweep_id_buf_len - 1] = '\0';
        }
    }
    rc = 0;

cleanup:
    Py_XDECREF(sweep_id);
    Py_XDECREF(sweep_fn);
    Py_XDECREF(kwargs);
    Py_XDECREF(cfg_dict);
    return rc;
}


/* ------------------------------------------------------------------ */
/*  wandb_agent_c                                                      */
/* ------------------------------------------------------------------ */
int wandb_agent_c(const char *sweep_id,
                  const char *project,
                  const char *entity,
                  int         count)
{
    PyObject *agent_fn = NULL;
    PyObject *kwargs   = NULL;
    PyObject *result   = NULL;
    int       rc       = -1;

    if (!wandb_module) {
        fprintf(stderr, "[wandb_c] wandb_agent_c: wandb not initialised. "
                        "Call wandb_sweep_c (or wandb_init_c) first.\n");
        return -1;
    }

    agent_fn = PyObject_GetAttrString(wandb_module, "agent");
    if (!agent_fn) { print_py_error("wandb.agent lookup"); return -1; }

    kwargs = PyDict_New();
    if (!kwargs) { Py_DECREF(agent_fn); return -1; }

    if (project && project[0] != '\0') {
        PyObject *p = PyUnicode_FromString(project);
        PyDict_SetItemString(kwargs, "project", p);
        Py_DECREF(p);
    }
    if (entity && entity[0] != '\0') {
        PyObject *e = PyUnicode_FromString(entity);
        PyDict_SetItemString(kwargs, "entity", e);
        Py_DECREF(e);
    }
    if (count > 0) {
        PyObject *c = PyLong_FromLong((long)count);
        PyDict_SetItemString(kwargs, "count", c);
        Py_DECREF(c);
    }

    {
        PyObject *py_id = PyUnicode_FromString(sweep_id);
        PyObject *args  = PyTuple_Pack(1, py_id);
        result = PyObject_Call(agent_fn, args, kwargs);
        Py_DECREF(args);
        Py_DECREF(py_id);
    }
    if (!result) { print_py_error("wandb.agent call"); goto cleanup; }
    rc = 0;

cleanup:
    Py_XDECREF(result);
    Py_XDECREF(kwargs);
    Py_XDECREF(agent_fn);
    return rc;
}


/* ================================================================== */
/*  Sweep-agent threading API                                          */
/*                                                                     */
/*  Architecture:                                                      */
/*    - wandb_sweep_start_agent_c  : starts wandb.agent in a Python   */
/*      thread with a callback that (a) calls wandb.init, (b) writes  */
/*      the sampled config to sweep_globals["params_json"], (c) sets  */
/*      "params_ready" Event, then (d) waits on "run_done" Event.      */
/*    - wandb_sweep_params_c       : waits for "params_ready", copies  */
/*      the params JSON into the caller's buffer, then clears the      */
/*      event.  Also updates wandb_run from sweep_globals["run"].      */
/*    - wandb_sweep_run_done_c     : sets "run_done" Event so the      */
/*      callback can call wandb.finish() and the agent can request     */
/*      the next run.                                                  */
/* ================================================================== */

/* Inline Python that sets up the threading infrastructure.
 * Executed once by wandb_sweep_start_agent_c. */
static const char *SWEEP_SETUP_CODE =
"import threading, json as _json, wandb as _wandb\n"
"\n"
"params_ready = threading.Event()\n"
"run_done     = threading.Event()\n"
"params_json  = '{}'\n"
"_run_obj     = None\n"
"\n"
"def _agent_callback():\n"
"    global params_json, _run_obj\n"
"    _run_obj = _wandb.init()\n"
"    cfg = dict(_run_obj.config)\n"
"    params_json = _json.dumps(cfg)\n"
"    params_ready.set()\n"
"    while not run_done.is_set():\n"
"        import time; time.sleep(0.02)\n"
"    run_done.clear()\n"
"    _wandb.finish()\n"
;


/* ------------------------------------------------------------------ */
/*  wandb_sweep_start_agent_c                                          */
/*  Starts wandb.agent(sweep_id, function=_agent_callback, count=N)   */
/*  in a background Python thread.  Must be called after              */
/*  wandb_sweep_c.                                                     */
/* ------------------------------------------------------------------ */
int wandb_sweep_start_agent_c(const char *sweep_id,
                               const char *project,
                               const char *entity,
                               int         count)
{
    PyObject *glb    = NULL;
    PyObject *result = NULL;
    int       rc     = -1;

    if (!python_started || !wandb_module) {
        fprintf(stderr, "[wandb_c] wandb_sweep_start_agent_c: "
                        "call wandb_sweep_c first.\n");
        return -1;
    }

    /* Create a fresh dict to serve as our module globals. */
    Py_XDECREF(sweep_globals);
    sweep_globals = PyDict_New();
    if (!sweep_globals) return -1;

    /* Inject __builtins__ so exec works. */
    PyDict_SetItemString(sweep_globals, "__builtins__",
                         PyEval_GetBuiltins());

    /* Run the setup code — defines _agent_callback, Events, etc. */
    result = PyRun_String(SWEEP_SETUP_CODE, Py_file_input,
                          sweep_globals, sweep_globals);
    if (!result) { print_py_error("sweep setup code"); return -1; }
    Py_DECREF(result);

    /* Build the agent-launch code string. */
    {
        char launch[1024];
        const char *proj_arg   = (project && project[0]) ? project : "";
        const char *entity_arg = (entity  && entity[0])  ? entity  : "";
        snprintf(launch, sizeof(launch),
                 "import threading as _t\n"
                 "_agent_thread = _t.Thread(\n"
                 "    target=_wandb.agent,\n"
                 "    kwargs={'sweep_id': '%s',\n"
                 "            'function': _agent_callback,\n"
                 "            'count': %d,\n"
                 "            'project': '%s',\n"
                 "            'entity': '%s'})\n"
                 "_agent_thread.daemon = True\n"
                 "_agent_thread.start()\n",
                 sweep_id, count, proj_arg, entity_arg);

        result = PyRun_String(launch, Py_file_input,
                              sweep_globals, sweep_globals);
        if (!result) { print_py_error("agent thread launch"); return -1; }
        Py_DECREF(result);
    }

    rc = 0;
    return rc;
}


/* ------------------------------------------------------------------ */
/*  wandb_sweep_params_c                                               */
/*  Blocks until the agent callback has called wandb.init() and        */
/*  written the sampled params as JSON.  Copies the JSON into buf.     */
/*  Also updates the internal wandb_run pointer.                       */
/*  Returns 1 if params were received, 0 on timeout/error.            */
/* ------------------------------------------------------------------ */
int wandb_sweep_params_c(char *buf, int buf_len, double timeout_s)
{
    PyObject *params_ready = NULL;
    PyObject *wait_fn      = NULL;
    PyObject *wait_result  = NULL;
    PyObject *params_str   = NULL;
    const char *s          = NULL;
    int        got_params  = 0;

    if (!sweep_globals) {
        fprintf(stderr, "[wandb_c] wandb_sweep_params_c: "
                        "call wandb_sweep_start_agent_c first.\n");
        return 0;
    }

    params_ready = PyDict_GetItemString(sweep_globals, "params_ready");
    if (!params_ready) { print_py_error("params_ready lookup"); return 0; }

    /* Call params_ready.wait(timeout=timeout_s) */
    wait_fn = PyObject_GetAttrString(params_ready, "wait");
    if (!wait_fn) { print_py_error("Event.wait"); return 0; }

    {
        PyObject *py_timeout = PyFloat_FromDouble(timeout_s);
        PyObject *args       = PyTuple_Pack(1, py_timeout);
        wait_result = PyObject_Call(wait_fn, args, NULL);
        Py_DECREF(args);
        Py_DECREF(py_timeout);
    }
    Py_DECREF(wait_fn);

    if (!wait_result) { print_py_error("Event.wait call"); return 0; }
    got_params = PyObject_IsTrue(wait_result);  /* True = event set (params ready) */
    Py_DECREF(wait_result);

    if (!got_params) {
        fprintf(stderr, "[wandb_c] wandb_sweep_params_c: timed out waiting "
                        "for sweep agent callback.\n");
        return 0;
    }

    /* Clear the event: params_ready.clear() */
    {
        PyObject *clear_fn = PyObject_GetAttrString(params_ready, "clear");
        if (clear_fn) {
            PyObject *r = PyObject_CallNoArgs(clear_fn);
            Py_XDECREF(r);
            Py_DECREF(clear_fn);
        }
    }

    /* Copy params JSON into caller's buffer. */
    params_str = PyDict_GetItemString(sweep_globals, "params_json");
    if (params_str && PyUnicode_Check(params_str)) {
        s = PyUnicode_AsUTF8(params_str);
        if (s && buf && buf_len > 0) {
            strncpy(buf, s, (size_t)(buf_len - 1));
            buf[buf_len - 1] = '\0';
        }
    }

    /* Also grab the wandb_run object so wandb_log / wandb_config_set work. */
    {
        PyObject *run_obj = PyDict_GetItemString(sweep_globals, "_run_obj");
        if (run_obj && run_obj != Py_None) {
            Py_XDECREF(wandb_run);
            Py_INCREF(run_obj);
            wandb_run = run_obj;
        }
    }

    return 1;
}


/* ------------------------------------------------------------------ */
/*  wandb_sweep_run_done_c                                             */
/*  Signals the agent callback that training is done.                  */
/*  The callback will then call wandb.finish() and the agent will      */
/*  request the next run from the sweep controller.                    */
/* ------------------------------------------------------------------ */
void wandb_sweep_run_done_c(void)
{
    PyObject *run_done = NULL;
    PyObject *set_fn   = NULL;
    PyObject *result   = NULL;

    if (!sweep_globals) return;

    run_done = PyDict_GetItemString(sweep_globals, "run_done");
    if (!run_done) { print_py_error("run_done lookup"); return; }

    set_fn = PyObject_GetAttrString(run_done, "set");
    if (set_fn) {
        result = PyObject_CallNoArgs(set_fn);
        Py_XDECREF(result);
        Py_DECREF(set_fn);
    }

    /* Release run reference — the callback will call wandb.finish(). */
    Py_XDECREF(wandb_run);
    wandb_run = NULL;
}


/* ================================================================== */
/*  End sweep-agent threading API                                      */
/* ================================================================== */


/* ------------------------------------------------------------------ */
/*  wandb_finish_c                                                     */
/*  Finishes the current wandb run but keeps Python alive so that      */
/*  wandb_init_c can be called again (e.g. for sweep runs).            */
/* ------------------------------------------------------------------ */
void wandb_finish_c(void)
{
    PyObject *finish_fn = NULL;
    PyObject *result    = NULL;

    if (wandb_module) {
        finish_fn = PyObject_GetAttrString(wandb_module, "finish");
        if (finish_fn) {
            result = PyObject_CallNoArgs(finish_fn);
            if (!result) print_py_error("wandb.finish call");
            Py_XDECREF(result);
            Py_XDECREF(finish_fn);
        }
    }

    /* Release the run object so wandb_init_c can create a new one. */
    Py_XDECREF(wandb_run);
    wandb_run = NULL;
    /* Keep wandb_module and python_started — Python stays alive. */
}


/* ------------------------------------------------------------------ */
/*  wandb_shutdown_c                                                   */
/*  Full teardown: release all Python objects and finalise the         */
/*  interpreter.  Call once after all runs are done.                   */
/* ------------------------------------------------------------------ */
void wandb_shutdown_c(void)
{
    /* Give wandb's background service thread a moment to flush the
     * last run's data before we tear down the interpreter. */
    {
        PyObject *time_mod = PyImport_ImportModule("time");
        if (time_mod) {
            PyObject *sleep_fn = PyObject_GetAttrString(time_mod, "sleep");
            if (sleep_fn) {
                PyObject *t = PyFloat_FromDouble(2.0);
                PyObject *r = PyObject_CallFunctionObjArgs(sleep_fn, t, NULL);
                Py_XDECREF(r);
                Py_DECREF(t);
                Py_DECREF(sleep_fn);
            }
            Py_DECREF(time_mod);
        }
    }

    Py_XDECREF(wandb_run);
    wandb_run = NULL;

    Py_XDECREF(sweep_globals);
    sweep_globals = NULL;

    Py_XDECREF(wandb_module);
    wandb_module = NULL;

    if (python_started) {
        Py_Finalize();
        python_started = 0;
    }
}
