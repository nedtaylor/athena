module athena__initialiser_ident
  !! Module containing the implementation of the identity initialiser
  !!
  !! This module contains the implementation of the identity initialiser
  !! for the weights and biases of a layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__misc_types, only: initialiser_type
  implicit none


  private

  public :: ident_type


  type, extends(initialiser_type) :: ident_type
     !! Type for the identity initialiser
   contains
     procedure, pass(this) :: initialise => ident_initialise
     !! Initialise the weights and biases using the identity matrix
  end type ident_type


  interface ident_type
     module function initialiser_ident_setup() result(initialiser)
       !! Interface for the Identity initialiser
       type(ident_type) :: initialiser
       !! Identity initialiser object
     end function initialiser_ident_setup
  end interface ident_type



contains

!###############################################################################
  module function initialiser_ident_setup() result(initialiser)
    !! Interface for the Identity initialiser
    implicit none

    type(ident_type) :: initialiser
    !! Identity initialiser object

    initialiser%name = "ident"

  end function initialiser_ident_setup
!###############################################################################


!###############################################################################
  subroutine ident_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the identity matrix
    implicit none

    ! Arguments
    class(ident_type), intent(inout) :: this
    !! Instance of the identity initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    ! Local variables
    integer :: i, j
    !! Loop index
    integer :: ndim
    !! Number of dimensions
    integer, dimension(:), allocatable :: iprime, iprime2
    !! Index variables


    if(all(shape(input).ne.size(input,1)))then
       call stop_program( &
            'A non-square tensor cannot be initialised as an identity matrix' &
       )
       return
    end if

    select rank(input)
    rank(0)
       input = 1._real32
    rank(1)
       if(size(input).ne.1)then
          if(.not.present(spacing))then
             call stop_program( &
                  'A vector of length greater than 1 cannot be &
                  &initialised as an identity matrix' &
             )
             return
          else
             ndim = size(spacing)
             if(ndim.eq.1)then
                do i = 1, size(input)/spacing(1)
                   input(1 + ( i - 1 ) * ( spacing(1) + 1) ) = 1._real32
                end do
             elseif(ndim.gt.1)then
                allocate(iprime(ndim))
                allocate(iprime2(ndim))
                iprime2 = 0
                iprime2(1) = 1
                do i = 1, size(input)/spacing(1)
                   iprime(ndim) = &
                        mod( &
                             (i - 1) / product( spacing(:ndim-1) ), &
                             product(spacing(:ndim)) &
                        )
                   iprime(ndim) = iprime(ndim) * product(spacing(:ndim-1))
                   do j = ndim - 1, 1, -1
                      if(sum(iprime(j+1:)).eq.0) then
                         iprime(j) = 0
                      else
                         iprime(j) = &
                              mod( &
                                   (i - 1), &
                                   sum(iprime(j+1:)) &
                              ) / product(spacing(:j-1))
                      end if
                      iprime(j) = iprime(j) * product(spacing(:j-1))
                   end do
                   input(1 + sum(iprime * ( spacing(1) + iprime2 ))) = 1._real32
                end do
             end if
          end if
       else
          input = 1._real32
       end if
    rank(2)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i) = 1._real32
       end do
    rank(3)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i) = 1._real32
       end do
    rank(4)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i) = 1._real32
       end do
    rank(5)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i,i) = 1._real32
       end do
    rank(6)
       input = 0._real32
       do i = 1, size(input,1)
          input(i,i,i,i,i,i) = 1._real32
       end do
    end select

  end subroutine ident_initialise
!###############################################################################

end module athena__initialiser_ident
