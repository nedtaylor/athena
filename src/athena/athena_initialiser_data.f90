module athena__initialiser_data
  !! Module containing the implementation of the data initialiser
  !!
  !! This module contains the implementation of the data initialiser
  !! for the weights and biases of a layer
  use coreutils, only: real32, stop_program
  use athena__misc_types, only: base_init_type
  implicit none


  private

  public :: data_init_type


  type, extends(base_init_type) :: data_init_type
     !! Type for the data initialiser
     real(real32), dimension(:), allocatable :: data
     !! Data to initialise the weights or biases with
   contains
     procedure, pass(this) :: initialise => data_initialise
     !! Initialise the weights and biases using the data distribution
  end type data_init_type


  interface data_init_type
     module function initialiser_data_setup(data) result(initialiser)
       !! Interface for the data initialiser
       type(data_init_type) :: initialiser
       !! data initialiser object
       real(real32), dimension(..), intent(in) :: data
       !! Data to initialise the weights and biases with
     end function initialiser_data_setup
  end interface data_init_type



contains

!###############################################################################
  module function initialiser_data_setup(data) result(initialiser)
    !! Interface for the data initialiser
    implicit none

    ! Arguments
    real(real32), dimension(..), intent(in) :: data
    !! Data to initialise the weights and biases with

    type(data_init_type) :: initialiser
    !! data initialiser object

    initialiser%name = "data"
    allocate(initialiser%data(size(data)))
    select rank(data)
    rank(0)
       initialiser%data(1) = data
    rank(1)
       initialiser%data(:) = data(:)
    rank(2)
       initialiser%data(:) = reshape(data, [size(data)])
    rank(3)
       initialiser%data(:) = reshape(data, [size(data)])
    rank(4)
       initialiser%data(:) = reshape(data, [size(data)])
    rank(5)
       initialiser%data(:) = reshape(data, [size(data)])
    rank(6)
       initialiser%data(:) = reshape(data, [size(data)])
    rank default
       call stop_program("initialiser_data_setup: Unsupported rank of data array")
    end select

  end function initialiser_data_setup
!###############################################################################


!###############################################################################
  pure subroutine data_initialise(this, input, fan_in, fan_out, spacing)
    !! Initialise the weights and biases using the data distribution
    implicit none

    ! Arguments
    class(data_init_type), intent(inout) :: this
    !! Instance of the data initialiser
    real(real32), dimension(..), intent(out) :: input
    !! Weights and biases to initialise
    integer, optional, intent(in) :: fan_in, fan_out
    !! Number of input and output parameters
    integer, dimension(:), optional, intent(in) :: spacing
    !! Spacing of the input and output units

    select rank(input)
    rank(0)
       input = this%data(1)
    rank(1)
       input(:) = this%data(:)
    rank(2)
       input(:,:) = reshape(this%data, shape(input))
    rank(3)
       input(:,:,:) = reshape(this%data, shape(input))
    rank(4)
       input(:,:,:,:) = reshape(this%data, shape(input))
    rank(5)
       input(:,:,:,:,:) = reshape(this%data, shape(input))
    rank(6)
       input(:,:,:,:,:,:) = reshape(this%data, shape(input))
    end select

  end subroutine data_initialise
!###############################################################################

end module athena__initialiser_data
