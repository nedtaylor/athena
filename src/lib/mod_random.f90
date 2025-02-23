module athena__random
  !! Module containing functions to initialise the random number generator
  use athena__io_utils, only: stop_program
  implicit none
  logical :: l_random_initialised=.false.


  private

  public :: random_setup



contains

!###############################################################################
  subroutine random_setup(seed, num_seed, restart, already_initialised)
    !! Initialise the random number generator
    implicit none

    ! Arguments
    integer, dimension(..), optional, intent(in) :: seed
    !! Seed for the random number generator
    integer, optional, intent(out) :: num_seed
    !! Number of seeds
    logical, optional, intent(in) :: restart
    !! Restart the random number generator
    logical, optional, intent(out) :: already_initialised
    !! Check if the random number generator is already initialised

    ! Local variables
    integer :: l
    !! Loop index
    integer :: itmp1
    !! Temporary integer
    integer :: num_seed_
    !! Number of seeds
    logical :: restart_
    !! Restart the random number generator
    integer, allocatable, dimension(:) :: seed_arr
    !! Seed array
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! Check if restart is defined
    !---------------------------------------------------------------------------
    if(present(restart))then
       restart_ = restart
    else
       restart_ = .false.
    end if
    if(present(already_initialised)) already_initialised = .false.

    !---------------------------------------------------------------------------
    ! Check if already initialised
    !---------------------------------------------------------------------------
    if(l_random_initialised.and..not.restart_)then
       if(present(already_initialised)) already_initialised = .true.
       return !! no need to initialise if already initialised
    else
       call random_seed(size=num_seed_)
       allocate(seed_arr(num_seed_))
       if(present(seed))then
          select rank(seed)
          rank(0)
             seed_arr = seed
          rank(1)
             if(size(seed,dim=1).ne.1)then
                if(size(seed,dim=1).eq.num_seed_)then
                   seed_arr = seed
                else
                   write(err_msg,'(A)') &
                        "seed size not consistent with &
                        &seed size returned by implementation" // &
                        achar(13) // achar(10) // &
                        "Cannot resolve"
                   call stop_program(err_msg)
                   return
                end if
             else
                seed_arr = seed(1)
             end if
          end select
       else
          call system_clock(count=itmp1)
          seed_arr = itmp1 + 37* (/ (l-1,l=1,num_seed_) /)
       end if
       call random_seed(put=seed_arr)
       l_random_initialised = .true.
    end if

    if(present(num_seed)) num_seed = num_seed_

  end subroutine random_setup
!###############################################################################

end module athena__random
