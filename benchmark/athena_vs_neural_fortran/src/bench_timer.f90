module bench_timer
  !! Monotonic wall-clock timing for the benchmark apps.
  use iso_c_binding, only: c_double
  implicit none
  private
  public :: wall_time

  interface
     function bench_monotonic() bind(c, name="bench_monotonic") result(seconds)
       import :: c_double
       real(c_double) :: seconds
     end function bench_monotonic
  end interface

contains

  function wall_time() result(seconds)
    real(c_double) :: seconds

    seconds = bench_monotonic()
    if(seconds < 0._c_double) error stop "bench_monotonic failed"
  end function wall_time
end module bench_timer
