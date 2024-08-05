!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D input layer
!!!#############################################################################
module input_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use custom_types, only: &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  implicit none
  
  
  type, extends(base_layer_type) :: input_layer_type
     integer :: num_outputs
     !  real(real12), allocatable, dimension(:,:) :: output
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_input
     procedure, pass(this) :: init => init_input
     procedure, pass(this) :: set_batch_size => set_batch_size_input
     procedure, pass(this) :: read => read_input
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this) :: set => set_input
  end type input_layer_type

  interface input_layer_type
     module function layer_setup(input_shape, batch_size, verbose) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
       type(input_layer_type) :: layer
     end function layer_setup
  end interface input_layer_type

  
  private
  public :: input_layer_type
  public :: read_input_layer


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!! placeholder to satisfy deferred
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(input_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    call this%output%set( input )
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!! placeholder to satisfy deferred
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(input_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient
    return
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
#if defined(GFORTRAN)
  module function layer_setup(input_shape, batch_size, verbose) result(layer)
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    type(input_layer_type) :: layer
#else
  module procedure layer_setup
    implicit none
#endif

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    call layer%set_hyperparams(verbose = verbose_)


    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

#if defined(GFORTRAN)
  end function layer_setup
#else
  end procedure layer_setup
#endif
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  subroutine set_hyperparams_input(this, input_rank, verbose)
    implicit none
    class(input_layer_type), intent(inout) :: this
    integer, optional, intent(in) :: input_rank
    integer, optional, intent(in) :: verbose

    this%name = "input"
    this%type = "inpt"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank

  end subroutine set_hyperparams_input
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_input(this, input_shape, batch_size, verbose)
    implicit none
    class(input_layer_type), intent(inout) :: this
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
    
    this%output%shape = this%input_shape
    this%num_outputs = product(this%input_shape)

    
    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_input
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_input(this, batch_size, verbose)
    implicit none
    class(input_layer_type), intent(inout) :: this
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
       select case(size(this%input_shape))
       case(1)
          this%input_rank = 1
          this%output = array1d_type()
          call this%output%allocate( shape = [ &
               this%input_shape(1) ], &
               source=0._real12 &
       )
       case(2)
          this%input_rank = 1
          this%output = array2d_type()
          call this%output%allocate( shape = [ &
               this%input_shape(1), this%batch_size ], &
               source=0._real12 &
       )
       case(3)
          this%input_rank = 2
          this%output = array3d_type()
          call this%output%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), this%batch_size ], &
               source=0._real12 &
       )
       case(4)
          this%input_rank = 3
          this%output = array4d_type()
          call this%output%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), this%batch_size ], &
               source=0._real12 &
          )
       case(5)
          this%input_rank = 4
          this%output = array5d_type()
          call this%output%allocate( shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%input_shape(4), this%batch_size ], &
               source=0._real12 &
          )
       end select
    end if

  end subroutine set_batch_size_input
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  subroutine read_input(this, unit, verbose)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    class(input_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat, verbose_ = 0
    integer :: itmp1= 0
    integer :: input_rank = 0
    integer, dimension(3) :: input_shape
    character(256) :: buffer, tag

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
      case("INPUT_RANK")
         call assign_val(buffer, input_rank, itmp1)
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
        input_rank = input_rank, &
        verbose = verbose_ &
   )
   call this%init(input_shape = input_shape)

  end subroutine read_input
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_input_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(input_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    call layer%read(unit, verbose=verbose_)

  end function read_input_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set input layer values
!!!#############################################################################
  pure subroutine set_input(this, input)
    implicit none
    class(input_layer_type), intent(inout) :: this
    real(real12), &
         dimension(..), intent(in) :: input
         !dimension(this%batch_size * this%num_outputs), intent(in) :: input


    call this%output%set( input )

  end subroutine set_input
!!!#############################################################################


end module input_layer
!!!#############################################################################
