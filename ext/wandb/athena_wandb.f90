module athena_wandb
  !! Fortran interface to Weights & Biases (wandb) experiment tracking.
  !!
  !! This module provides a high-level Fortran API for logging training
  !! metrics and hyper-parameters to wandb.  Under the hood it calls a
  !! thin C layer (`wandb_c.c`) that embeds the Python interpreter and
  !! invokes the `wandb` Python library.
  !!
  !! This file has been generated using LLM AI code agents and has not yet been
  !! thoroughly tested.
  !!
  !! ## Quick start
  !!
  !! ```fortran
  !! use athena_wandb
  !!
  !! call wandb_init(project="my_project", name="run-01")
  !! call wandb_config_set("learning_rate", 0.001)
  !! call wandb_config_set("epochs", 100)
  !!
  !! do epoch = 1, 100
  !!    ! ... training ...
  !!    call wandb_log("loss", loss_val, step=epoch)
  !! end do
  !!
  !! call wandb_finish()
  !! ```
  !!
  !! ## Prerequisites
  !!
  !! - Python ≥ 3.8 with `wandb` installed (`pip install wandb`)
  !! - Link against `libathena_wandb` (which itself links Python)
  !!
  use iso_c_binding, only: c_int, c_double, c_char, c_null_char, c_ptr, &
       c_loc, c_null_ptr
  implicit none

  private

  public :: wandb_init
  public :: wandb_log
  public :: wandb_config_set
  public :: wandb_config_get
  public :: wandb_sweep
  public :: wandb_agent
  public :: wandb_sweep_start_agent
  public :: wandb_sweep_next_params
  public :: wandb_sweep_run_done
  public :: wandb_finish
  public :: wandb_shutdown
  public :: wandb_sweep_config_type


  !-----------------------------------------------------------------------------
  ! Generic interfaces
  !-----------------------------------------------------------------------------

  !! Log a scalar metric to the current wandb run.
  interface wandb_log
     module procedure wandb_log_real64
     module procedure wandb_log_real32
     module procedure wandb_log_integer
  end interface wandb_log

  !! Set a wandb config hyper-parameter.
  interface wandb_config_set
     module procedure wandb_config_set_integer
     module procedure wandb_config_set_real64
     module procedure wandb_config_set_real32
     module procedure wandb_config_set_string
  end interface wandb_config_set

  !! Read back a config value injected by the sweep agent after wandb_init.
  interface wandb_config_get
     module procedure wandb_config_get_integer
     module procedure wandb_config_get_real64
     module procedure wandb_config_get_real32
     module procedure wandb_config_get_string
  end interface wandb_config_get

  !! Register a sweep — accepts either a raw JSON string or a
  !! `wandb_sweep_config_type` built with the helper API.
  interface wandb_sweep
     module procedure wandb_sweep_json_str
     module procedure wandb_sweep_cfg_obj
  end interface wandb_sweep


  !-----------------------------------------------------------------------------
  ! wandb_sweep_config_type — programmatic sweep configuration builder
  !-----------------------------------------------------------------------------
  !! Builder for a wandb sweep configuration.
  !!
  !! Build the configuration with the type-bound procedures, then pass
  !! it directly to `wandb_sweep` or call `to_json()` for the raw string.
  !!
  !! ## Example
  !!
  !! ```fortran
  !! type(wandb_sweep_config_type) :: cfg
  !!
  !! call cfg%set_method("bayes")
  !! call cfg%set_metric("val_loss", "minimize")
  !!
  !! ! continuous range with explicit distribution
  !! call cfg%add_param_range("learning_rate", 1e-5_real64, 1e-2_real64, &
  !!                           distribution="log_uniform_values")
  !! call cfg%add_param_range("dropout_rate", 0.0_real64, 0.3_real64, &
  !!                           distribution="uniform")
  !!
  !! ! discrete integer values
  !! call cfg%add_param_values("hidden_size", [32, 64, 128, 256])
  !!
  !! ! discrete real values
  !! call cfg%add_param_values("weight_decay", [1e-4_real64, 1e-3_real64])
  !!
  !! ! discrete string values
  !! call cfg%add_param_values("activation", ["relu   ", "tanh   ", "sigmoid"])
  !!
  !! call wandb_sweep(cfg, project="my-project", sweep_id=id)
  !! ```
  type :: wandb_sweep_config_type
     character(len=:), allocatable :: method_str
     character(len=:), allocatable :: metric_name_str
     character(len=:), allocatable :: metric_goal_str
     character(len=:), allocatable :: params_buf  ! accumulated parameter JSON
     integer :: num_params = 0
  contains
     procedure :: set_method        => swcfg_set_method
     procedure :: set_metric        => swcfg_set_metric
     procedure, private :: add_range_r32   => swcfg_add_range_r32
     procedure, private :: add_range_r64   => swcfg_add_range_r64
     procedure, private :: add_vals_int    => swcfg_add_values_int
     procedure, private :: add_vals_r32    => swcfg_add_values_r32
     procedure, private :: add_vals_r64    => swcfg_add_values_r64
     procedure, private :: add_vals_str    => swcfg_add_values_str
     generic   :: add_param_range  => add_range_r32, add_range_r64
     generic   :: add_param_values => add_vals_int, add_vals_r32, &
                                       add_vals_r64, add_vals_str
     procedure :: to_json => swcfg_to_json
  end type wandb_sweep_config_type


  !-----------------------------------------------------------------------------
  ! C bindings (private)
  !-----------------------------------------------------------------------------

  interface
     integer(c_int) function wandb_init_c(project, name, entity, sweep_id) &
          bind(C, name="wandb_init_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in) :: project(*)
       character(kind=c_char), intent(in) :: name(*)
       character(kind=c_char), intent(in) :: entity(*)
       character(kind=c_char), intent(in) :: sweep_id(*)
     end function wandb_init_c

     subroutine wandb_log_metric_c(key, value, step) &
          bind(C, name="wandb_log_metric_c")
       import :: c_char, c_double, c_int
       character(kind=c_char), intent(in) :: key(*)
       real(c_double), value, intent(in) :: value
       integer(c_int), value, intent(in) :: step
     end subroutine wandb_log_metric_c

     subroutine wandb_config_set_int_c(key, value) &
          bind(C, name="wandb_config_set_int_c")
       import :: c_char, c_int
       character(kind=c_char), intent(in) :: key(*)
       integer(c_int), value, intent(in) :: value
     end subroutine wandb_config_set_int_c

     subroutine wandb_config_set_real_c(key, value) &
          bind(C, name="wandb_config_set_real_c")
       import :: c_char, c_double
       character(kind=c_char), intent(in) :: key(*)
       real(c_double), value, intent(in) :: value
     end subroutine wandb_config_set_real_c

     subroutine wandb_config_set_str_c(key, value) &
          bind(C, name="wandb_config_set_str_c")
       import :: c_char
       character(kind=c_char), intent(in) :: key(*)
       character(kind=c_char), intent(in) :: value(*)
     end subroutine wandb_config_set_str_c

     subroutine wandb_finish_c() bind(C, name="wandb_finish_c")
     end subroutine wandb_finish_c

     subroutine wandb_shutdown_c() bind(C, name="wandb_shutdown_c")
     end subroutine wandb_shutdown_c

     integer(c_int) function wandb_config_get_int_c(key, default_value) &
          bind(C, name="wandb_config_get_int_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in) :: key(*)
       integer(c_int), value, intent(in)  :: default_value
     end function wandb_config_get_int_c

     real(c_double) function wandb_config_get_real_c(key, default_value) &
          bind(C, name="wandb_config_get_real_c")
       import :: c_double, c_char
       character(kind=c_char), intent(in) :: key(*)
       real(c_double), value, intent(in)  :: default_value
     end function wandb_config_get_real_c

     integer(c_int) function wandb_config_get_str_c(key, buf, buf_len) &
          bind(C, name="wandb_config_get_str_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in)  :: key(*)
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value, intent(in)   :: buf_len
     end function wandb_config_get_str_c

     integer(c_int) function wandb_sweep_c( &
          config_json, project, entity, sweep_id_buf, sweep_id_buf_len) &
          bind(C, name="wandb_sweep_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in)  :: config_json(*)
       character(kind=c_char), intent(in)  :: project(*)
       character(kind=c_char), intent(in)  :: entity(*)
       character(kind=c_char), intent(out) :: sweep_id_buf(*)
       integer(c_int), value, intent(in)   :: sweep_id_buf_len
     end function wandb_sweep_c

     integer(c_int) function wandb_agent_c(sweep_id, project, entity, count) &
          bind(C, name="wandb_agent_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in) :: sweep_id(*)
       character(kind=c_char), intent(in) :: project(*)
       character(kind=c_char), intent(in) :: entity(*)
       integer(c_int), value, intent(in)  :: count
     end function wandb_agent_c

     integer(c_int) function wandb_sweep_start_agent_c( &
          sweep_id, project, entity, count) &
          bind(C, name="wandb_sweep_start_agent_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in) :: sweep_id(*)
       character(kind=c_char), intent(in) :: project(*)
       character(kind=c_char), intent(in) :: entity(*)
       integer(c_int), value, intent(in)  :: count
     end function wandb_sweep_start_agent_c

     integer(c_int) function wandb_sweep_params_c(buf, buf_len, timeout_s) &
          bind(C, name="wandb_sweep_params_c")
       import :: c_int, c_char, c_double
       character(kind=c_char), intent(out) :: buf(*)
       integer(c_int), value, intent(in)   :: buf_len
       real(c_double), value, intent(in)   :: timeout_s
     end function wandb_sweep_params_c

     subroutine wandb_sweep_run_done_c() &
          bind(C, name="wandb_sweep_run_done_c")
     end subroutine wandb_sweep_run_done_c

  end interface


contains


  !-----------------------------------------------------------------------------
  ! wandb_init
  !-----------------------------------------------------------------------------
  subroutine wandb_init(project, name, entity, sweep_id)
    !! Initialise a wandb run.
    !!
    !! @param project   Project name (required).
    !! @param name      Run display name (optional).
    !! @param entity    wandb entity / team (optional).
    !! @param sweep_id  Sweep ID returned by `wandb_sweep`.  When supplied,
    !!                  the run joins the sweep and its `wandb.config` is
    !!                  populated with the sweep-sampled hyperparameters.
    character(len=*), intent(in) :: project
    character(len=*), intent(in), optional :: name
    character(len=*), intent(in), optional :: entity
    character(len=*), intent(in), optional :: sweep_id

    integer(c_int) :: rc
    character(len=:), allocatable :: c_name, c_entity, c_sweep_id

    if(present(name))then
       c_name = name // c_null_char
    else
       c_name = c_null_char
    end if

    if(present(entity))then
       c_entity = entity // c_null_char
    else
       c_entity = c_null_char
    end if

    if(present(sweep_id))then
       c_sweep_id = trim(sweep_id) // c_null_char
    else
       c_sweep_id = c_null_char
    end if

    rc = wandb_init_c( &
         project // c_null_char, &
         c_name,    &
         c_entity,  &
         c_sweep_id &
    )

    if(rc /= 0)then
       write(0,*) "[athena_wandb] WARNING: wandb_init failed (rc=", rc, ")"
       write(0,*) "  Logging will be silently skipped."
    end if

  end subroutine wandb_init


  !-----------------------------------------------------------------------------
  ! wandb_log  (real64)
  !-----------------------------------------------------------------------------
  subroutine wandb_log_real64(key, value, step)
    character(len=*), intent(in) :: key
    real(c_double), intent(in) :: value
    integer, intent(in), optional :: step

    integer(c_int) :: c_step

    c_step = -1_c_int
    if(present(step)) c_step = int(step, c_int)
    call wandb_log_metric_c(key // c_null_char, value, c_step)

  end subroutine wandb_log_real64


  !-----------------------------------------------------------------------------
  ! wandb_log  (real32)
  !-----------------------------------------------------------------------------
  subroutine wandb_log_real32(key, value, step)
    use iso_c_binding, only: c_float
    character(len=*), intent(in) :: key
    real(c_float), intent(in) :: value
    integer, intent(in), optional :: step

    call wandb_log_real64(key, real(value, c_double), step)

  end subroutine wandb_log_real32


  !-----------------------------------------------------------------------------
  ! wandb_log  (integer)
  !-----------------------------------------------------------------------------
  subroutine wandb_log_integer(key, value, step)
    character(len=*), intent(in) :: key
    integer, intent(in) :: value
    integer, intent(in), optional :: step

    call wandb_log_real64(key, real(value, c_double), step)

  end subroutine wandb_log_integer


  !-----------------------------------------------------------------------------
  ! wandb_config_set  (integer)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_set_integer(key, value)
    character(len=*), intent(in) :: key
    integer, intent(in) :: value

    call wandb_config_set_int_c(key // c_null_char, int(value, c_int))

  end subroutine wandb_config_set_integer


  !-----------------------------------------------------------------------------
  ! wandb_config_set  (real64)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_set_real64(key, value)
    character(len=*), intent(in) :: key
    real(c_double), intent(in) :: value

    call wandb_config_set_real_c(key // c_null_char, value)

  end subroutine wandb_config_set_real64


  !-----------------------------------------------------------------------------
  ! wandb_config_set  (real32)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_set_real32(key, value)
    use iso_c_binding, only: c_float
    character(len=*), intent(in) :: key
    real(c_float), intent(in) :: value

    call wandb_config_set_real_c(key // c_null_char, real(value, c_double))

  end subroutine wandb_config_set_real32


  !-----------------------------------------------------------------------------
  ! wandb_config_set  (string)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_set_string(key, value)
    character(len=*), intent(in) :: key
    character(len=*), intent(in) :: value

    call wandb_config_set_str_c( &
         key // c_null_char, &
         value // c_null_char  &
    )

  end subroutine wandb_config_set_string


  !-----------------------------------------------------------------------------
  ! wandb_config_get  (integer)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_get_integer(key, value, default_value)
    !! Read an integer hyper-parameter from the current wandb config.
    !! Useful inside a sweep run to retrieve the sampled value.
    !! @param key           Config key.
    !! @param value         Receives the value (or default_value if absent).
    !! @param default_value Fallback when the key is not present.
    character(len=*), intent(in)  :: key
    integer,          intent(out) :: value
    integer,          intent(in), optional :: default_value

    integer(c_int) :: dflt
    dflt = 0_c_int
    if(present(default_value)) dflt = int(default_value, c_int)
    value = int(wandb_config_get_int_c(key // c_null_char, dflt))
  end subroutine wandb_config_get_integer


  !-----------------------------------------------------------------------------
  ! wandb_config_get  (real64)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_get_real64(key, value, default_value)
    !! Read a double-precision hyper-parameter from the current wandb config.
    character(len=*), intent(in)  :: key
    real(c_double),   intent(out) :: value
    real(c_double), intent(in), optional :: default_value

    real(c_double) :: dflt
    dflt = 0.0_c_double
    if(present(default_value)) dflt = default_value
    value = wandb_config_get_real_c(key // c_null_char, dflt)
  end subroutine wandb_config_get_real64


  !-----------------------------------------------------------------------------
  ! wandb_config_get  (real32)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_get_real32(key, value, default_value)
    !! Read a single-precision hyper-parameter from the current wandb config.
    use iso_c_binding, only: c_float
    character(len=*), intent(in)  :: key
    real(c_float),    intent(out) :: value
    real(c_float), intent(in), optional :: default_value

    real(c_double) :: dflt, result
    dflt = 0.0_c_double
    if(present(default_value)) dflt = real(default_value, c_double)
    result = wandb_config_get_real_c(key // c_null_char, dflt)
    value  = real(result, c_float)
  end subroutine wandb_config_get_real32


  !-----------------------------------------------------------------------------
  ! wandb_config_get  (string)
  !-----------------------------------------------------------------------------
  subroutine wandb_config_get_string(key, value, default_value)
    !! Read a string hyper-parameter from the current wandb config.
    !! The output string `value` is space-padded to its declared length.
    character(len=*), intent(in)    :: key
    character(len=*), intent(inout) :: value
    character(len=*), intent(in), optional :: default_value

    integer :: buf_len, found, i
    character(kind=c_char), allocatable :: buf(:)

    buf_len = len(value) + 1
    allocate(buf(buf_len))
    buf = c_null_char

    found = wandb_config_get_str_c(key // c_null_char, buf, int(buf_len, c_int))

    if(found /= 0)then
       value = ' '
       do i = 1, len(value)
          if(buf(i) == c_null_char) exit
          value(i:i) = buf(i)
       end do
    else if(present(default_value))then
       value = default_value
    end if
    deallocate(buf)
  end subroutine wandb_config_get_string


  !-----------------------------------------------------------------------------
  ! wandb_sweep  (raw JSON string variant)
  !-----------------------------------------------------------------------------
  subroutine wandb_sweep_json_str(config_json, project, sweep_id, entity)
    !! Register a hyperparameter sweep with wandb.
    !!
    !! @param config_json  JSON string describing the sweep (method, metric,
    !!                     parameters).  Example:
    !!   '{"method":"bayes","metric":{"name":"loss","goal":"minimize"},'
    !! // '"parameters":{"lr":{"min":0.0001,"max":0.1},'  &
    !! // '"hidden":{"values":[16,32,64]}}}'
    !! @param project      wandb project name.
    !! @param sweep_id     Output: the sweep ID string assigned by wandb.
    !! @param entity       Optional wandb entity/team.
    character(len=*),              intent(in)  :: config_json
    character(len=*),              intent(in)  :: project
    character(len=*),              intent(out) :: sweep_id
    character(len=*), optional,    intent(in)  :: entity

    integer, parameter :: BUF_LEN = 256
    character(kind=c_char) :: id_buf(BUF_LEN)
    character(len=:), allocatable :: c_entity
    integer(c_int) :: rc
    integer :: i

    if(present(entity))then
       c_entity = entity // c_null_char
    else
       c_entity = c_null_char
    end if

    id_buf = c_null_char
    rc = wandb_sweep_c( &
         config_json // c_null_char, &
         project     // c_null_char, &
         c_entity,                   &
         id_buf,                     &
         int(BUF_LEN, c_int)         &
    )

    if(rc /= 0)then
       write(0,*) "[athena_wandb] WARNING: wandb_sweep failed (rc=", rc, ")"
       sweep_id = ' '
       return
    end if

    sweep_id = ' '
    do i = 1, len(sweep_id)
       if(id_buf(i) == c_null_char) exit
       sweep_id(i:i) = id_buf(i)
    end do
  end subroutine wandb_sweep_json_str


  !-----------------------------------------------------------------------------
  ! wandb_sweep  (config-type variant)
  !-----------------------------------------------------------------------------
  subroutine wandb_sweep_cfg_obj(config, project, sweep_id, entity)
    !! Register a sweep from a `wandb_sweep_config_type` builder object.
    type(wandb_sweep_config_type), intent(in)  :: config
    character(len=*),              intent(in)  :: project
    character(len=*),              intent(out) :: sweep_id
    character(len=*), optional,    intent(in)  :: entity

    character(len=:), allocatable :: json
    json = config%to_json()
    if(present(entity))then
       call wandb_sweep_json_str(json, project, sweep_id, entity)
    else
       call wandb_sweep_json_str(json, project, sweep_id)
    end if
  end subroutine wandb_sweep_cfg_obj


  !-----------------------------------------------------------------------------
  ! wandb_agent
  !-----------------------------------------------------------------------------
  subroutine wandb_agent(sweep_id, project, count, entity)
    !! Run a wandb sweep agent.
    !!
    !! Calls wandb.agent(sweep_id, count=count, project=project).  After this
    !! returns, call wandb_config_get to read the sampled hyperparameters,
    !! run your training loop, then call wandb_finish.
    !!
    !! @param sweep_id  Sweep ID returned by wandb_sweep.
    !! @param project   wandb project name.
    !! @param count     Number of runs to execute (0 = until sweep is done).
    !! @param entity    Optional wandb entity/team.
    character(len=*),           intent(in) :: sweep_id
    character(len=*),           intent(in) :: project
    integer, optional,          intent(in) :: count
    character(len=*), optional, intent(in) :: entity

    character(len=:), allocatable :: c_entity
    integer(c_int) :: rc, c_count

    c_count = 0_c_int
    if(present(count)) c_count = int(count, c_int)

    if(present(entity))then
       c_entity = entity // c_null_char
    else
       c_entity = c_null_char
    end if

    rc = wandb_agent_c( &
         sweep_id // c_null_char, &
         project  // c_null_char, &
         c_entity,                 &
         c_count                   &
    )

    if(rc /= 0)then
       write(0,*) "[athena_wandb] WARNING: wandb_agent failed (rc=", rc, ")"
    end if
  end subroutine wandb_agent


  !-----------------------------------------------------------------------------
  ! wandb_sweep_start_agent
  !-----------------------------------------------------------------------------
  subroutine wandb_sweep_start_agent(sweep_id, project, count, entity)
    !! Start a wandb sweep agent in a background Python thread.
    !!
    !! The agent runs `count` runs, each time calling its internal callback
    !! which calls `wandb.init()` (populating `wandb.config` with sweep-sampled
    !! hyperparameters), signals `wandb_sweep_next_params`, then waits until
    !! `wandb_sweep_run_done` is called before finishing the run.
    !!
    !! ## Usage pattern
    !!
    !! ```fortran
    !! call wandb_sweep(config=cfg, project="my-proj", sweep_id=sid)
    !! call wandb_sweep_start_agent(sid, "my-proj", count=5)
    !! do i = 1, 5
    !!    call wandb_sweep_next_params(params_json)
    !!    ! parse params_json, train, log ...
    !!    call wandb_sweep_run_done()
    !! end do
    !! call wandb_shutdown()
    !! ```
    character(len=*),           intent(in) :: sweep_id
    character(len=*),           intent(in) :: project
    integer,                    intent(in) :: count
    character(len=*), optional, intent(in) :: entity

    character(len=:), allocatable :: c_entity
    integer(c_int) :: rc

    if(present(entity))then
       c_entity = trim(entity) // c_null_char
    else
       c_entity = c_null_char
    end if

    rc = wandb_sweep_start_agent_c( &
         trim(sweep_id) // c_null_char, &
         trim(project)  // c_null_char, &
         c_entity,                       &
         int(count, c_int)               &
    )

    if(rc /= 0)then
       write(0,*) "[athena_wandb] WARNING: wandb_sweep_start_agent failed (rc=", rc, ")"
    end if
  end subroutine wandb_sweep_start_agent


  !-----------------------------------------------------------------------------
  ! wandb_sweep_next_params
  !-----------------------------------------------------------------------------
  subroutine wandb_sweep_next_params(params_json, timeout_s)
    !! Block until the sweep agent has started the next run and populated
    !! `wandb.config` with the sweep-sampled hyperparameters.
    !!
    !! After this call returns, the module-level `wandb_run` in the C layer
    !! is updated so that `wandb_log` and `wandb_config_set` calls are
    !! automatically routed to the current sweep run.
    !!
    !! @param params_json  Receives the sampled hyperparameters as a JSON
    !!                     object string, e.g. `{"lr":0.01,"hidden":32}`.
    !!                     Use `wandb_config_get` to retrieve individual values.
    !! @param timeout_s    Seconds to wait before giving up (default: 120.0).
    character(len=*),          intent(out) :: params_json
    real(c_double), optional, intent(in)  :: timeout_s

    character(kind=c_char), dimension(4096) :: c_buf
    real(c_double) :: tval
    integer(c_int) :: ok
    integer :: i

    tval = 120.0_c_double
    if(present(timeout_s)) tval = timeout_s

    ok = wandb_sweep_params_c(c_buf, int(4096, c_int), tval)

    if(ok == 0_c_int)then
       write(0,*) "[athena_wandb] WARNING: wandb_sweep_next_params timed out."
       params_json = '{}'
       return
    end if

    params_json = ' '
    do i = 1, len(params_json)
       if(c_buf(i) == c_null_char) exit
       params_json(i:i) = c_buf(i)
    end do
  end subroutine wandb_sweep_next_params


  !-----------------------------------------------------------------------------
  ! wandb_sweep_run_done
  !-----------------------------------------------------------------------------
  subroutine wandb_sweep_run_done()
    !! Signal that the current sweep run's training is finished.
    !!
    !! The sweep agent callback will then call `wandb.finish()` and request
    !! the next set of hyperparameters from the sweep controller.
    !! Call this once per sweep run, after all logging is done.
    call wandb_sweep_run_done_c()
  end subroutine wandb_sweep_run_done


  !-----------------------------------------------------------------------------
  ! wandb_finish
  !-----------------------------------------------------------------------------
  subroutine wandb_finish()
    !! Finish the current wandb run (calls wandb.finish()) but keeps the
    !! Python interpreter alive.  Safe to call between sweep runs.
    !! For the very last teardown, call wandb_shutdown() instead or in
    !! addition to this.
    call wandb_finish_c()
  end subroutine wandb_finish


  !-----------------------------------------------------------------------------
  ! wandb_shutdown
  !-----------------------------------------------------------------------------
  subroutine wandb_shutdown()
    !! Shut down the Python interpreter and release all resources.
    !! Call once after all wandb runs (including sweep runs) are finished.
    !! No wandb calls should be made after this.
    call wandb_shutdown_c()
  end subroutine wandb_shutdown


  !=============================================================================
  ! wandb_sweep_config_type — type-bound procedure implementations
  !=============================================================================

  subroutine swcfg_set_method(self, method)
    !! Set the sweep search method: "bayes" | "grid" | "random".
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: method
    self%method_str = trim(method)
  end subroutine swcfg_set_method


  subroutine swcfg_set_metric(self, name, goal)
    !! Set the optimisation metric.
    !! @param name  Metric key logged via wandb_log (e.g. "val_loss").
    !! @param goal  "minimize" or "maximize".
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    character(len=*),               intent(in)    :: goal
    self%metric_name_str = trim(name)
    self%metric_goal_str = trim(goal)
  end subroutine swcfg_set_metric


  !---------------------------------------------------------------------------
  ! add_param_range  (real32 / real64)
  !---------------------------------------------------------------------------
  subroutine swcfg_add_range_r32(self, name, min_val, max_val, distribution)
    use iso_c_binding, only: c_float
    !! Add a continuous hyperparameter with min/max bounds.
    !! @param name          Parameter name.
    !! @param min_val       Minimum value.
    !! @param max_val       Maximum value.
    !! @param distribution  wandb distribution string
    !!                      (e.g. "uniform", "log_uniform_values").
    !!                      Defaults to "uniform" when absent.
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    real(c_float),                  intent(in)    :: min_val, max_val
    character(len=*), optional,     intent(in)    :: distribution
    call swcfg_add_range_r64(self, name, &
         real(min_val, c_double), real(max_val, c_double), distribution)
  end subroutine swcfg_add_range_r32


  subroutine swcfg_add_range_r64(self, name, min_val, max_val, distribution)
    !! Add a continuous hyperparameter with min/max bounds.
    !! @param name          Parameter name.
    !! @param min_val       Minimum value.
    !! @param max_val       Maximum value.
    !! @param distribution  wandb distribution string
    !!                      (e.g. "uniform", "log_uniform_values").
    !!                      Defaults to "uniform" when absent.
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    real(c_double),                 intent(in)    :: min_val, max_val
    character(len=*), optional,     intent(in)    :: distribution

    character(len=:), allocatable :: distrib, fragment

    if(present(distribution))then
       distrib = trim(distribution)
    else
       distrib = "uniform"
    end if

    fragment = '"' // trim(name) // '":{' // &
               '"distribution":"' // distrib // '",' // &
               '"min":' // r64_to_json(min_val) // ',' // &
               '"max":' // r64_to_json(max_val) // '}'

    if(.not. allocated(self%params_buf)) self%params_buf = ''
    if(self%num_params > 0)then
       self%params_buf = self%params_buf // ',' // fragment
    else
       self%params_buf = self%params_buf // fragment
    end if
    self%num_params = self%num_params + 1
  end subroutine swcfg_add_range_r64


  !---------------------------------------------------------------------------
  ! add_param_values  (integer array)
  !---------------------------------------------------------------------------
  subroutine swcfg_add_values_int(self, name, values)
    !! Add a discrete integer hyperparameter.
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    integer,                        intent(in)    :: values(:)

    character(len=:), allocatable :: arr, fragment
    character(len=32) :: buf
    integer :: i

    arr = '['
    do i = 1, size(values)
       write(buf, '(i0)') values(i)
       if(i > 1) arr = arr // ','
       arr = arr // trim(buf)
    end do
    arr = arr // ']'

    fragment = '"' // trim(name) // '":{"values":' // arr // '}'
    if(.not. allocated(self%params_buf)) self%params_buf = ''
    if(self%num_params > 0)then
       self%params_buf = self%params_buf // ',' // fragment
    else
       self%params_buf = self%params_buf // fragment
    end if
    self%num_params = self%num_params + 1
  end subroutine swcfg_add_values_int


  !---------------------------------------------------------------------------
  ! add_param_values  (real32 array)
  !---------------------------------------------------------------------------
  subroutine swcfg_add_values_r32(self, name, values)
    use iso_c_binding, only: c_float
    !! Add a discrete real-valued hyperparameter (single precision).
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    real(c_float),                  intent(in)    :: values(:)

    real(c_double), allocatable :: tmp(:)
    integer :: i
    allocate(tmp(size(values)))
    do i = 1, size(values)
       tmp(i) = real(values(i), c_double)
    end do
    call swcfg_add_values_r64(self, name, tmp)
  end subroutine swcfg_add_values_r32


  !---------------------------------------------------------------------------
  ! add_param_values  (real64 array)
  !---------------------------------------------------------------------------
  subroutine swcfg_add_values_r64(self, name, values)
    !! Add a discrete real-valued hyperparameter (double precision).
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    real(c_double),                 intent(in)    :: values(:)

    character(len=:), allocatable :: arr, fragment
    integer :: i

    arr = '['
    do i = 1, size(values)
       if(i > 1) arr = arr // ','
       arr = arr // r64_to_json(values(i))
    end do
    arr = arr // ']'

    fragment = '"' // trim(name) // '":{"values":' // arr // '}'
    if(.not. allocated(self%params_buf)) self%params_buf = ''
    if(self%num_params > 0)then
       self%params_buf = self%params_buf // ',' // fragment
    else
       self%params_buf = self%params_buf // fragment
    end if
    self%num_params = self%num_params + 1
  end subroutine swcfg_add_values_r64


  !---------------------------------------------------------------------------
  ! add_param_values  (string array)
  !---------------------------------------------------------------------------
  subroutine swcfg_add_values_str(self, name, values)
    !! Add a discrete string hyperparameter.
    !! All elements of `values` must have the same declared length (Fortran
    !! character array requirement); trailing spaces are trimmed automatically.
    class(wandb_sweep_config_type), intent(inout) :: self
    character(len=*),               intent(in)    :: name
    character(len=*),               intent(in)    :: values(:)

    character(len=:), allocatable :: arr, fragment
    integer :: i

    arr = '['
    do i = 1, size(values)
       if(i > 1) arr = arr // ','
       arr = arr // '"' // trim(values(i)) // '"'
    end do
    arr = arr // ']'

    fragment = '"' // trim(name) // '":{"values":' // arr // '}'
    if(.not. allocated(self%params_buf)) self%params_buf = ''
    if(self%num_params > 0)then
       self%params_buf = self%params_buf // ',' // fragment
    else
       self%params_buf = self%params_buf // fragment
    end if
    self%num_params = self%num_params + 1
  end subroutine swcfg_add_values_str


  !---------------------------------------------------------------------------
  ! to_json
  !---------------------------------------------------------------------------
  function swcfg_to_json(self) result(json)
    !! Serialise the configuration to a JSON string suitable for `wandb_sweep`.
    class(wandb_sweep_config_type), intent(in) :: self
    character(len=:), allocatable :: json

    character(len=:), allocatable :: method, metric_part, params_part

    if(allocated(self%method_str))then
       method = self%method_str
    else
       method = 'bayes'
    end if

    if(allocated(self%metric_name_str))then
       metric_part = ',"metric":{"name":"' // self%metric_name_str // &
                     '","goal":"'           // self%metric_goal_str // '"}'
    else
       metric_part = ''
    end if

    if(allocated(self%params_buf) .and. len(self%params_buf) > 0)then
       params_part = ',"parameters":{' // self%params_buf // '}'
    else
       params_part = ',"parameters":{}'
    end if

    json = '{"method":"' // trim(method) // '"' // &
            metric_part // params_part // '}'
  end function swcfg_to_json


  !---------------------------------------------------------------------------
  ! Private helper: format a real64 as a compact JSON number string.
  !---------------------------------------------------------------------------
  function r64_to_json(x) result(s)
    real(c_double), intent(in) :: x
    character(len=32) :: s
    integer :: i
    write(s, '(ES23.8E3)') x
    s = adjustl(s)
    ! JSON numbers use lowercase 'e'
    do i = 1, len_trim(s)
       if(s(i:i) == 'E') s(i:i) = 'e'
    end do
    s = trim(s)
  end function r64_to_json


end module athena_wandb
