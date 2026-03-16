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
int wandb_init_c(const char *project, const char *name, const char *entity)
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
/*  wandb_finish_c                                                     */
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

    Py_XDECREF(wandb_run);
    wandb_run = NULL;

    Py_XDECREF(wandb_module);
    wandb_module = NULL;

    if (python_started) {
        Py_Finalize();
        python_started = 0;
    }
}
