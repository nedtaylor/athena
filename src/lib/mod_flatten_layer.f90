!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D flattening layer
!!!#############################################################################
module flatten_layer
  use constants, only: real12
  use base_layer, only: flatten_layer_type
  use custom_types, only: &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  implicit none
  
  
  type, extends(base_layer_type) :: flatten_layer_type
     integer :: num_outputs, num_addit_outputs = 0
     !  real(real12), allocatable, dimension(:,:,:) :: di
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_flatten
     procedure, pass(this) :: init => init_flatten
     procedure, pass(this) :: set_batch_size => set_batch_size_flatten
     procedure, pass(this) :: read => read_flatten
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
  end type flatten_layer_type

  interface flatten_layer_type
     module function layer_setup(input_shape, batch_size, num_addit_outputs, &
          verbose ) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: num_addit_outputs
       integer, optional, intent(in) :: verbose
       type(flatten_layer_type) :: layer
     end function layer_setup
  end interface flatten_layer_type

  
  private
  public :: flatten_layer_type
  public :: read_flatten_layer


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select type(output => this%output)
    type is (array2d_type)
       select rank(input); rank(3)
          output%val(:this%num_outputs, :this%batch_size) = &
               reshape(input, [this%num_outputs, this%batch_size])
       end select
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(gradient); rank(2)
       call this%di%allocate( &
            source = reshape(gradient(:this%num_outputs,:), this%di%shape) &
       )
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
         input_shape, batch_size, num_addit_outputs, &
         verbose &
  ) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: num_addit_outputs
    integer, optional, intent(in) :: verbose

    type(flatten_layer_type) :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    if(present(num_addit_outputs)) layer%num_addit_outputs = num_addit_outputs
    call layer%set_hyperparams(num_addit_outputs, verbose_)

    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  subroutine set_hyperparams_flatten(this, num_addit_outputs, &
       input_rank, verbose)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
    integer, intent(in) :: num_addit_outputs
    integer, optional, intent(in) :: input_rank
    integer, optional, intent(in) :: verbose

    this%name = "flatten"
    this%type = "flat"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank
    this%num_addit_outputs = num_addit_outputs

  end subroutine set_hyperparams_flatten
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_flatten(this, input_shape, batch_size, verbose)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
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
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    
    this%num_outputs = product(this%input_shape)
    this%output%shape = [this%num_outputs]


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_flatten
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_flatten(this, batch_size, verbose)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
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
       if(this%output%allocated) call this%output%deallocate()
       this%output = array2d_type()
       call this%output%allocate(shape = [ &
             (this%num_outputs + this%num_addit_outputs), this%batch_size ], &
             source=0._real12 &
       )
       if(allocated(this%di)) deallocate(this%di)
       select case(size(this%input_shape))
       case(1)
          this%input_rank = 1
          this%di = array1d_type()
          call this%di%allocate( shape = [ &
               this%input_shape(1) ], &
               source=0._real12 &
       )
       case(2)
          this%input_rank = 1
          this%di = array2d_type()
          call this%di%allocate( shape = [ &
               this%input_shape(1), this%batch_size ], &
               source=0._real12 &
       )
       case(3)
          this%input_rank = 2
          this%di = array3d_type()
          call this%di%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), this%batch_size ], &
               source=0._real12 &
       )
       case(4)
          this%input_rank = 3
          this%di = array4d_type()
          call this%di%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), this%batch_size ], &
               source=0._real12 &
          )
       case(5)
          this%input_rank = 4
          this%di = array5d_type()
          call this%di%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%input_shape(4), this%batch_size ], &
               source=0._real12 &
          )
       end select
    end if
 
  end subroutine set_batch_size_flatten
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  subroutine read_flatten(this, unit, verbose)
    implicit none
    class(flatten_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    integer :: num_addit_outputs = 0

    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose

    !! loop over tags in layer card
    tag_loop: do

      !! check for end of file
      read(unit,'(A)',iostat=stat) buffer
      if(stat.ne.0)then
         write(0,*) "ERROR: file encountered error (EoF?) before END FLATTEN"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle tag_loop

      !! check for end of convolution card
      if(trim(adjustl(buffer)).eq."END FLATTEN")then
         backspace(unit)
         exit tag_loop
      end if

      tag=trim(adjustl(buffer))
      if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

      !! read parameters from save file
      select case(trim(tag))
      case("INPUT_SHAPE")
         call assign_vec(buffer, input_shape, itmp1)
      case("NUM_ADDIT_OUTPUTS")
         call assign_val(buffer, num_addit_outputs, itmp1)
      case default
         !! don't look for "e" due to scientific notation of numbers
         !! ... i.e. exponent (E+00)
         if(scan(to_lower(trim(adjustl(buffer))),&
              'abcdfghijklmnopqrstuvwxyz').eq.0)then
            cycle tag_loop
         elseif(tag(:3).eq.'END')then
            cycle tag_loop
         end if
         stop "Unrecognised line in input file: "//trim(adjustl(buffer))
      end select
   end do tag_loop

   !! set transfer activation function
   call this%set_hyperparams( &
        num_addit_outputs, &
        verbose = verbose_ &
   )
   call this%init(input_shape = input_shape)

  end subroutine read_flatten
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_flatten_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(flatten_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    call layer%read(unit, verbose=verbose_)

  end function read_flatten_layer
!!!#############################################################################

end module flatten_layer
!!!#############################################################################