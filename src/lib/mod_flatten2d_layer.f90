!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module flatten2d_layer
  use constants, only: real12
  use base_layer, only: flatten_layer_type
  implicit none
  
  
  type, extends(flatten_layer_type) :: flatten2d_layer_type
     real(real12), allocatable, dimension(:,:,:,:) :: di
   contains
     procedure, pass(this) :: init => init_flatten2d
     procedure, pass(this) :: set_batch_size => set_batch_size_flatten2d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
  end type flatten2d_layer_type

  interface flatten2d_layer_type
     module function layer_setup(input_shape, batch_size, num_addit_outputs) &
          result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: num_addit_outputs
       type(flatten2d_layer_type) :: layer
     end function layer_setup
  end interface flatten2d_layer_type

  
  private
  public :: flatten2d_layer_type


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(flatten2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       this%output(:this%num_outputs, :this%batch_size) = &
            reshape(input, [this%num_outputs, this%batch_size])
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(flatten2d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(gradient); rank(2)
       this%di = reshape(gradient(:this%num_outputs,:), shape(this%di))
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup(input_shape, batch_size, num_addit_outputs) &
         result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: num_addit_outputs

    type(flatten2d_layer_type) :: layer


    layer%name = "flatten2d"
    layer%input_rank = 3
    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(num_addit_outputs)) layer%num_addit_outputs = num_addit_outputs
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_flatten2d(this, input_shape, batch_size, verbose)
    implicit none
    class(flatten2d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size

    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    
    this%num_outputs = product(this%input_shape)
    this%output_shape = [this%num_outputs]


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_flatten2d
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_flatten2d(this, batch_size, verbose)
    implicit none
    class(flatten2d_layer_type), intent(inout) :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
 
    integer :: verbose_ = 0
 
 
    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size
 
 
    !!--------------------------------------------------------------------------
    !! allocate arrays
    !!--------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       allocate(this%output( &
            (this%num_outputs + this%num_addit_outputs), this%batch_size ), &
            source=0._real12)
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di( &
            this%input_shape(1), &
            this%input_shape(2), &
            this%input_shape(3), this%batch_size), &
            source=0._real12)
    end if
 
  end subroutine set_batch_size_flatten2d
 !!!#############################################################################

end module flatten2d_layer
!!!#############################################################################
