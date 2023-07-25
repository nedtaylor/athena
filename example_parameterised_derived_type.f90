module my_module
  implicit none

  ! Parameterized derived type
  type, public :: my_type(n)
    integer, len :: n
    real, dimension(n) :: data
  end type my_type

contains
  subroutine tester(input)
    implicit none
    type(my_type(*)), dimension(0:), intent(in) :: input
    integer :: i

    write(*,*) "START"
    do i = lbound(input,dim=1), ubound(input,dim=1)
       write(*,*) i
       write(*,*) size(input(i)%data)
       write(*,*) input(i)%data
       write(*,*)
    end do

  end subroutine tester


end module my_module

program main
  use my_module
  implicit none

  type(my_type(:)), allocatable, dimension(:) :: my_array
  integer :: i, j
  integer, allocatable, dimension(:) :: n_list
  !real, allocatable, dimension(:) :: tmp_data


  n_list = [1, 4, 3, 2]
  !allocate(tmp_data(size(n_list,dim=1)), source=0.0)

  !my_array = (/ (&
  !     my_type(&
  !     n=n_list(i),&
  !     data=(/(0.0, j=1,n_list(i))/)&!tmp_data(:n_list(i))&
  !     ), i=1,size(n_list) ) /)
  allocate(my_array(0:size(n_list)-1), source=(/ (&
       my_type(&
       n=n_list(i),&
       data=(/(0.0, j=1,n_list(i))/)&!tmp_data(:n_list(i))&
       ), i=1,size(n_list) ) /))
  write(*,*) "HERE", lbound(my_array), ubound(my_array)

  ! Allocate the array and initialize each element
  !allocate(my_array(1:3), n=2)
  !my_array = [my_type(n=1, data=[0.0,0.0]), my_type(n=2, data=[0.0,0.0])]
  do i = lbound(my_array,dim=1), ubound(my_array,dim=1)
     write(*,*) i
     write(*,*) size(my_array(i)%data)
     write(*,*) my_array(i)%data
     write(*,*)
  end do

  call tester(my_array)


  ! ... Use the my_array here ...

  ! Deallocate the array
  !deallocate(my_array)

end program main
