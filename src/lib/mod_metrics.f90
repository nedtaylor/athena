!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains a type for storing and handling metrics
!!! module includes the following derived types:
!!! - metric_dict_type - a type for storing and handling metric data
!!!##################
!!! the metric derived type contains the following procedures:
!!! - check - checks if the metric has converged
!!! - add_t_t - adds two metric_dict_type together
!!!##################
!!! module contains the following procedures:
!!! - metric_dict_alloc - allocates memory for a metric_dict_type
!!!#############################################################################
module metrics
  use constants, only: real32
  implicit none

  type metric_dict_type
     character(10) :: key
     real(real32) :: val
     logical :: active
     real(real32) :: threshold
     real(real32), allocatable, dimension(:) :: history
   contains
     procedure :: check => metric_dict_check
     procedure :: add_t_t => metric_dict_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t
  end type metric_dict_type


  private

  public :: metric_dict_type
  public :: metric_dict_alloc


contains

!!!#############################################################################
!!! custom operation for summing metric_dict_type
!!!#############################################################################
  elemental function metric_dict_add(a, b) result(output)
    implicit none
    class(metric_dict_type), intent(in) :: a,b
    type(metric_dict_type) :: output
    
    output%key = a%key
    output%val = a%val + b%val
    output%threshold = a%threshold
    output%active = a%active
    if(allocated(a%history)) output%history = a%history

  end function metric_dict_add
!!!#############################################################################


!!!#############################################################################
!!! custom operation for allocating metric_dict_type
!!!#############################################################################
  subroutine metric_dict_alloc(input, source, length)
    implicit none
    type(metric_dict_type), dimension(:), intent(out) :: input
    type(metric_dict_type), dimension(:), optional, intent(in) :: source
    integer, optional, intent(in) :: length
    integer :: i
    
    if(present(length))then
       do i=1,size(input,dim=1)
          allocate(input(i)%history(length))
       end do
    else
       if(present(source))then
          do i=1,size(input,dim=1)
             input(i)%key = source(i)%key
             allocate(input(i)%history(size(source(i)%history,dim=1)))
             input(i)%threshold = source(i)%threshold
          end do
       else
          write(0,*) &
               "ERROR: metric_dict_alloc requires either a source or length"
          stop 1
       end if
    end if

  end subroutine metric_dict_alloc
!!!#############################################################################


!!!#############################################################################
!!! custom operation for checking metric convergence
!!!#############################################################################
  subroutine metric_dict_check(this,plateau_threshold,converged)
    implicit none
    class(metric_dict_type), intent(inout) :: this
    real(real32), intent(in) :: plateau_threshold
    integer, intent(out) :: converged
    
    converged = 0
    if(this%active)then
       this%history = eoshift(this%history, shift=-1, dim=1, boundary=this%val)
       if(&
            (trim(this%key).eq."loss".and.&
            abs(sum(this%history))/size(this%history,dim=1).lt.&
            this%threshold).or.&
            (trim(this%key).eq."accuracy".and.&
            abs(sum(1._real32-this%history))/size(this%history,dim=1).lt.&
            this%threshold) )then
          write(6,*) &
               "Convergence achieved, "//trim(this%key)//" threshold reached"
          write(6,*) "Exiting training loop"
          converged = 1
       elseif(all(abs(this%history-this%val).lt.plateau_threshold))then        
          write(0,'("ERROR: ",A," has remained constant for ",I0," runs")') &
               trim(this%key), size(this%history,dim=1)
          write(0,*) this%history
          write(0,*) "Exiting..."
          converged = -1
       end if
    end if

  end subroutine metric_dict_check
!!!#############################################################################

end module metrics
