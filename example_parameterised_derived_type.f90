module my_module
  implicit none

  type, public :: my_type(n)
     sequence
    integer, len :: n
    integer :: m=1
    real, dimension(n) :: data
  !contains
  !  procedure, pass(this) :: print => print_out
  end type my_type

!contains
!  subroutine print_out(this)
!    implicit none
!    class(my_type(*)), intent(in) :: this
!  
!    write(*,*) size(this%data)
!    write(*,*) this%data
!    write(*,*)
!  
!  end subroutine print_out

end module my_module

program main
  use my_module
  implicit none

  type(my_type(:)), allocatable, dimension(:) :: my_array
  integer :: i, j
  integer, allocatable, dimension(:) :: n_list
  !integer, dimension(4) :: arr

  n_list = [1, 4, 3, 2]
  allocate(my_array(0:size(n_list)-1), source=(/ (&
       my_type(&
       n=n_list(i),&
       data=(/(real(i), j=1,n_list(i))/)&
       ), i=1,size(n_list) ) /))

  !allocate(my_array(0:4))
  !do i = lbound(my_array,dim=1), ubound(my_array,dim=1)
  !   write(*,*) i
  !   call my_array(i)%print()
  !end do
  call tester(my_array)

  !arr(5) = 14
  !write(*,*) arr(5)

contains

  subroutine tester(input)
    implicit none
    type(my_type(*)), dimension(0:), intent(inout) :: input
    integer :: i
    integer, pointer :: p1

    do i = lbound(input,dim=1), ubound(input,dim=1)
       write(*,*) i
       write(*,*) size(input(i)%data), input(i)%data
       !write(*,*) input(i)%data(:)
       input(i)%data(4) = 16.0*i
       !write(*,*) input(i)%data(4)
       write(*,*) size(input(i)%data), input(i)%data(:4)
       write(*,*) loc(input(i))
       write(*,*) loc(input(i)%n), loc(input(i)%m)
       write(*,*) (loc(input(i)%data(j)),j=1,4)!size(input(i)%data,dim=1))
       !p1 = loc(input(i)%data(4))
       write(*,*) p1
       write(*,*) associated(p1)
!       write(*,*) (findloc(input(i)%data(j)),j=1,size(input(i)%data,dim=1))
       write(*,*)
    end do

  end subroutine tester

end program main

