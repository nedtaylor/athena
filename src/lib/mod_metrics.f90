module athena__metrics
  !! Module containing functions to compute the accuracy of a model
  !!
  !! This module contains a derived type for storing and handling metric data
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  implicit none


  private

  public :: metric_dict_type
  public :: metric_dict_alloc


  type metric_dict_type
     !! Type for storing and handling metric data
     character(10) :: key
     !! Key for the metric
     real(real32) :: val
     !! Value of the metric
     logical :: active
     !! Flag to indicate if the metric is active
     real(real32) :: threshold
     !! Threshold for the metric
     real(real32), allocatable, dimension(:) :: history
     !! History of the metric
   contains
     procedure :: check => metric_dict_check
     !! Check if the metric has converged
     procedure :: add_t_t => metric_dict_add
     !! Add two metric_dict_type together
     generic :: operator(+) => add_t_t
     !! Overload the addition operator
  end type metric_dict_type



contains

!###############################################################################
  elemental function metric_dict_add(a, b) result(output)
    !! Operation to add two metric_dict_type together
    implicit none

    ! Arguments
    class(metric_dict_type), intent(in) :: a,b
    !! Instances of metric data
    type(metric_dict_type) :: output
    !! Sum of the metric data
    
    output%key = a%key
    output%val = a%val + b%val
    output%threshold = a%threshold
    output%active = a%active
    if(allocated(a%history)) output%history = a%history

  end function metric_dict_add
!###############################################################################


!###############################################################################
  subroutine metric_dict_alloc(input, source, length)
    !! Allocate memory for a metric_dict_type
    implicit none

    ! Arguments
    type(metric_dict_type), dimension(:), intent(out) :: input
    !! Instance of metric data
    type(metric_dict_type), dimension(:), optional, intent(in) :: source
    !! Source of the metric data to copy
    integer, optional, intent(in) :: length
    !! Length of the metric data

    ! Local variables
    integer :: i
    !! Loop index

    
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
          call stop_program( &
               "metric_dict_alloc requires either a source or length" &
          )
       end if
    end if

  end subroutine metric_dict_alloc
!###############################################################################


!###############################################################################
  subroutine metric_dict_check(this,plateau_threshold,converged)
    !! Check if the metric has converged
    implicit none

    ! Arguments
    class(metric_dict_type), intent(inout) :: this
    !! Instance of metric data
    real(real32), intent(in) :: plateau_threshold
    !! Threshold for plateau
    integer, intent(out) :: converged
    !! Boolean whether the metric has converged


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
!###############################################################################

end module athena__metrics