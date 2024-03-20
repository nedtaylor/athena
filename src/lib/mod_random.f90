!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains random number generator initialisation
!!! module contains the following procedures:
!!! - random_setup - seed random number generator from seed vector or randomly
!!!#############################################################################
module random
  implicit none
  logical :: l_random_initialised=.false.

  private

  public :: random_setup


contains

!!!#############################################################################
!!! seed random number generator from vector of seeds
!!!#############################################################################
  subroutine random_setup(seed, num_seed, restart, already_initialised)
    implicit none
    integer, dimension(..), optional, intent(in) :: seed !dimension(..1)
    integer, optional, intent(out) :: num_seed
    logical, optional, intent(in) :: restart
    logical, optional, intent(out) :: already_initialised

    integer :: l
    integer :: itmp1
    integer :: num_seed_
    logical :: restart_
    integer, allocatable, dimension(:) :: seed_arr

    !! check if restart is defined
    if(present(restart))then
       restart_ = restart       
    else
       restart_ = .false.
    end if
    if(present(already_initialised)) already_initialised = .false.

    !! check if already initialised
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
                   write(0,*) "ERROR: seed size not consistent with &
                        &seed size returned by implementation"
                   write(0,*) "Cannot resolve"
                   write(0,*) "Exiting..."
                   stop 1
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
!!!#############################################################################

end module random
!!!#############################################################################
