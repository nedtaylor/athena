!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a deep setp layer
!!!#############################################################################
module deepset_layer
  use constants, only: real32
  use base_layer, only: learnable_layer_type
  use custom_types, only: activation_type, initialiser_type
  implicit none
  

!!!-----------------------------------------------------------------------------
!!! fully connected network layer type
!!!-----------------------------------------------------------------------------
  type, extends(learnable_layer_type) :: deepset_layer_type
     integer :: num_inputs, num_addit_inputs = 0
     integer :: num_outputs
     real(real32) :: lambda, gamma, bias
     real(real32), allocatable, dimension(:) :: dg, dl, db ! weight gradient
     real(real32), allocatable, dimension(:,:) :: output !output and activation
     real(real32), allocatable, dimension(:,:) :: di ! input gradient (i.e. delta)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_deepset
     procedure, pass(this) :: get_params => get_params_deepset
     procedure, pass(this) :: set_params => set_params_deepset
     procedure, pass(this) :: get_gradients => get_gradients_deepset
     procedure, pass(this) :: set_gradients => set_gradients_deepset
     procedure, pass(this) :: get_output => get_output_deepset

     procedure, pass(this) :: print => print_deepset
     procedure, pass(this) :: init => init_deepset
     procedure, pass(this) :: set_batch_size => set_batch_size_deepset
     
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_2d
     procedure, private, pass(this) :: backward_2d

     procedure, pass(this) :: reduce => layer_reduction
     procedure, pass(this) :: merge => layer_merge
     procedure :: add_t_t => layer_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
  end type deepset_layer_type


!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  interface deepset_layer_type
     module function layer_setup( &
          input_shape, batch_size, &
          num_inputs, &
          lambda, gamma) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: num_inputs
       integer, optional, intent(in) :: batch_size
       real(real32), optional, intent(in) :: lambda, gamma
       type(deepset_layer_type) :: layer
     end function layer_setup
  end interface deepset_layer_type


  private
  public :: deepset_layer_type
  public :: read_deepset_layer


contains

!!!#############################################################################
!!! layer reduction
!!!#############################################################################
  subroutine layer_reduction(this, rhs)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: rhs

    select type(rhs)
    type is(deepset_layer_type)
       this%dl = this%dl + rhs%dl
       this%dg = this%dg + rhs%dg
       this%db = this%db + rhs%db
    end select

  end subroutine  layer_reduction
!!!#############################################################################


!!!#############################################################################
!!! layer addition
!!!#############################################################################
  function layer_add(a, b) result(output)
    implicit none
    class(deepset_layer_type), intent(in) :: a, b
    type(deepset_layer_type) :: output

    output = a
    output%dl = output%dl + b%dl
    output%dg = output%dg + b%dg
    output%db = output%db + b%db

  end function layer_add
!!!#############################################################################


!!!#############################################################################
!!! layer merge
!!!#############################################################################
  subroutine layer_merge(this, input)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    select type(input)
    class is(deepset_layer_type)
       this%dl = this%dl + input%dl
       this%dg = this%dg + input%dg
       this%db = this%db + input%db
    end select

  end subroutine layer_merge
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! get number of parameters
!!!#############################################################################
  pure function get_num_params_deepset(this) result(num_params)
    implicit none
    class(deepset_layer_type), intent(in) :: this
    integer :: num_params

    num_params = 3

  end function get_num_params_deepset
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!!#############################################################################
  pure function get_params_deepset(this) result(params)
    implicit none
    class(deepset_layer_type), intent(in) :: this
    real(real32), dimension(this%num_params) :: params
  
    params = [this%lambda, this%gamma, this%bias]
  
  end function get_params_deepset
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!!#############################################################################
  subroutine set_params_deepset(this, params)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_params), intent(in) :: params
  
    this%lambda = params(1)
    this%gamma  = params(2)
    this%bias   = params(3)
  
  end subroutine set_params_deepset
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!!#############################################################################
  pure function get_gradients_deepset(this, clip_method) result(gradients)
    use clipper, only: clip_type
    implicit none
    class(deepset_layer_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real32), allocatable, dimension(:) :: gradients
  
    gradients = [ this%dl/this%batch_size, &
                  this%dg/this%batch_size, &
                  this%db/this%batch_size ]
  
    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  
  end function get_gradients_deepset
!!!#############################################################################


!!!#############################################################################
!!! set gradients
!!!#############################################################################
  subroutine set_gradients_deepset(this, gradients)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dl = gradients * this%batch_size
       this%dg = gradients * this%batch_size
       this%db = gradients * this%batch_size
    rank(1)
        this%dl = gradients(1) * this%batch_size
        this%dg = gradients(2) * this%batch_size
        this%db = gradients(3) * this%batch_size
    end select
  
  end subroutine set_gradients_deepset
!!!#############################################################################


!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_deepset(this, output)
  implicit none
  class(deepset_layer_type), intent(in) :: this
  real(real32), allocatable, dimension(..), intent(out) :: output

  select rank(output)
  rank(1)
     output = reshape(this%output, [size(this%output)])
  rank(2)
     output = this%output
  end select

end subroutine get_output_deepset
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input); rank(2)
       call forward_2d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient

    select rank(input); rank(2)
    select rank(gradient); rank(2)
       call backward_2d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       num_inputs, &
       lambda, gamma) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: num_inputs
    integer, optional, intent(in) :: batch_size
    real(real32), optional, intent(in) :: lambda, gamma
    
    type(deepset_layer_type) :: layer


    layer%name = "deepset"
    layer%input_rank = 1
    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! set up lambda and gamma
    !!--------------------------------------------------------------------------
    if(present(lambda))then
       layer%lambda = lambda
    else
       layer%lambda = 1._real32
    end if
    if(present(gamma))then
       layer%gamma = gamma
    else
       layer%gamma = 1._real32
    end if


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape))then
      call layer%init(input_shape=input_shape)
   elseif(present(num_inputs))then
      call layer%init(input_shape=[num_inputs])
   end if

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_deepset(this, input_shape, batch_size, verbose)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise number of inputs
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = [this%num_outputs]


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_deepset
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_deepset(this, batch_size, verbose)
   implicit none
   class(deepset_layer_type), intent(inout) :: this
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
           this%input_shape(1), &
           this%batch_size), source=0._real32)
      if(allocated(this%dl)) deallocate(this%dl)
      allocate(this%dl(this%batch_size), source=0._real32)
      if(allocated(this%dg)) deallocate(this%dg)
      allocate(this%dg(this%batch_size), source=0._real32)
      if(allocated(this%di)) deallocate(this%di)
      allocate(this%di( &
           this%input_shape(1), &
           this%batch_size), source=0._real32)
   end if

 end subroutine set_batch_size_deepset
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_deepset(this, file)
    implicit none
    class(deepset_layer_type), intent(in) :: this
    character(*), intent(in) :: file


  end subroutine print_deepset
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_deepset_layer(unit, verbose) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    class(deepset_layer_type), allocatable :: layer


  end function read_deepset_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_2d(this, input)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input

    integer :: s


    !! generate outputs from weights (lambda and gamma), biases, and inputs
    do concurrent(s=1:this%batch_size)
       this%output(:,s) = this%lambda * this%output(:,s) + &
            this%gamma * sum(input(:,s), dim=1) + this%bias
    end do

  end subroutine forward_2d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
  pure subroutine backward_2d(this, input, gradient)
    implicit none
    class(deepset_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient

    integer :: s


    do concurrent(s=1:this%batch_size)
       this%db(s) = sum(gradient(:,s))
       this%dl(s) = dot_product(input(:,s), gradient(:,s))
       this%dg(s)  = sum(gradient(:,s) * sum(input(:,s), dim=1))

       this%di(:,s) = this%lambda * gradient(:,s) + &
            this%gamma * sum(gradient(:,s), dim=1)
    end do

  end subroutine backward_2d
!!!#############################################################################

end module deepset_layer
!!!#############################################################################
