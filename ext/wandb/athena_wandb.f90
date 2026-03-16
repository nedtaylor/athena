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
  public :: wandb_finish


  !-----------------------------------------------------------------------------
  ! Generic interfaces
  !-----------------------------------------------------------------------------

  !! Log a scalar metric to the current wandb run.
  interface wandb_log
     module procedure wandb_log_real64
     module procedure wandb_log_real32
     module procedure wandb_log_integer
  end interface wandb_log

  !! Set a wand config hyper-parameter.
  interface wandb_config_set
     module procedure wandb_config_set_integer
     module procedure wandb_config_set_real64
     module procedure wandb_config_set_real32
     module procedure wandb_config_set_string
  end interface wandb_config_set


  !-----------------------------------------------------------------------------
  ! C bindings (private)
  !-----------------------------------------------------------------------------

  interface
     integer(c_int) function wandb_init_c(project, name, entity) &
          bind(C, name="wandb_init_c")
       import :: c_int, c_char
       character(kind=c_char), intent(in) :: project(*)
       character(kind=c_char), intent(in) :: name(*)
       character(kind=c_char), intent(in) :: entity(*)
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
  end interface


contains


  !-----------------------------------------------------------------------------
  ! wandb_init
  !-----------------------------------------------------------------------------
  subroutine wandb_init(project, name, entity)
    !! Initialise a wandb run.
    !!
    !! @param project  Project name (required).
    !! @param name     Run display name (optional).
    !! @param entity   wandb entity / team (optional).
    character(len=*), intent(in) :: project
    character(len=*), intent(in), optional :: name
    character(len=*), intent(in), optional :: entity

    integer(c_int) :: rc
    character(len=:), allocatable :: c_name, c_entity

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

    rc = wandb_init_c( &
         project // c_null_char, &
         c_name, &
         c_entity  &
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
  ! wandb_finish
  !-----------------------------------------------------------------------------
  subroutine wandb_finish()
    !! Finish the current wandb run and release resources.
    call wandb_finish_c()
  end subroutine wandb_finish


end module athena_wandb
