!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of an activation layer
!!!#############################################################################
module athena__actv_layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: base_layer_type
  use athena__misc_types, only: activation_type, &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  implicit none
  
  
  type, extends(base_layer_type) :: actv_layer_type
     class(activation_type), allocatable :: transfer
   contains
     procedure, pass(this) :: set_hyperparams => set_hyperparams_actv
     procedure, pass(this) :: init => init_actv
     procedure, pass(this) :: set_batch_size => set_batch_size_actv
     procedure, pass(this) :: print => print_actv
     procedure, pass(this) :: read => read_actv
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this), private :: forward_assumed_rank
     procedure, pass(this), private :: backward_assumed_rank
  end type actv_layer_type

  
  interface actv_layer_type
     module function layer_setup( &
          activation_function, activation_scale, &
          input_shape, batch_size, &
          verbose &
     ) result(layer)
       character(*), intent(in) :: activation_function
       real(real32), optional, intent(in) :: activation_scale
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       integer, optional, intent(in) :: verbose
       type(actv_layer_type) :: layer
     end function layer_setup
  end interface actv_layer_type


  private
  public :: actv_layer_type
  public :: read_actv_layer


contains

!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    call this%forward_assumed_rank(input)
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    real(real32), dimension(..), intent(in) :: gradient

    call this%backward_assumed_rank(input, gradient)
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup( &
       activation_function, activation_scale, &
       input_shape, batch_size, &
       verbose &
  ) result(layer)
  use athena__activation,  only: activation_setup
    implicit none
    character(*), intent(in) :: activation_function
    real(real32), optional, intent(in) :: activation_scale
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
    
    type(actv_layer_type) :: layer

    real(real32) :: activation_scale_
    integer :: verbose_


    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    activation_scale_ = 1._real32
    if(present(activation_scale)) activation_scale_ = activation_scale
    call layer%set_hyperparams( &
         activation_function = activation_function, &
         activation_scale = activation_scale_, &
         verbose = verbose_ &
    )


    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init( &
         input_shape=input_shape, &
         verbose=verbose_ &
    )

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  subroutine set_hyperparams_actv( &
       this, &
       activation_function, &
       activation_scale, &
       input_rank, &
       verbose &
  )
    use athena__activation,  only: activation_setup
    use athena__misc_ml, only: set_padding
    implicit none
    class(actv_layer_type), intent(inout) :: this
    integer, optional, intent(in) :: input_rank
    character(*), intent(in) :: activation_function
    real(real32), intent(in) :: activation_scale
    integer, optional, intent(in) :: verbose


    this%name = "actv"
    this%type = "actv"
    this%input_rank = 0
    if(present(input_rank)) this%input_rank = input_rank
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale) &
    )

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("ACTV activation function: ",A)') &
               trim(activation_function)
       end if
    end if

  end subroutine set_hyperparams_actv
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_actv(this, input_shape, batch_size, verbose)
    implicit none
    class(actv_layer_type), intent(inout) :: this
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
    this%input_rank = size(input_shape)
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)

    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    select case(this%input_rank + 1)
    case(1)
       this%output = array1d_type([ this%input_shape, max(1, this%batch_size) ])
    case(2)
       this%output = array2d_type([ this%input_shape, max(1, this%batch_size) ])
    case(3)
       this%output = array3d_type([ this%input_shape, max(1, this%batch_size) ])
    case(4)
       this%output = array4d_type([ this%input_shape, max(1, this%batch_size) ])
    case(5)
       this%output = array5d_type([ this%input_shape, max(1, this%batch_size) ])
    case default
       call stop_program('Activation layer only supports input ranks 1-5')
       return
    end select
    

    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_actv
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_actv(this, batch_size, verbose)
    implicit none
    class(actv_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    verbose_ = 0
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! allocate arrays
    !!--------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(allocated(this%output)) deallocate(this%output)
       select case(size(this%input_shape))
       case(1)
          this%input_rank = 1
          this%output = array2d_type()
          call this%output%allocate( array_shape = [ &
               this%input_shape(1), this%batch_size ], &
               source=0._real32 &
       )
       case(2)
          this%input_rank = 2
          this%output = array3d_type()
          call this%output%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%batch_size ], &
               source=0._real32 &
       )
       case(3)
          this%input_rank = 3
          this%output = array4d_type()
          call this%output%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), this%batch_size ], &
               source=0._real32 &
       )
       case(4)
          this%input_rank = 4
          this%output = array5d_type()
          call this%output%allocate( array_shape = [ &
               this%input_shape(1), &
               this%input_shape(2), &
               this%input_shape(3), &
               this%input_shape(4), this%batch_size ], &
               source=0._real32 &
          )
       end select
    end if

  end subroutine set_batch_size_actv
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_actv(this, file)
    implicit none
    class(actv_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("ACTV")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"ACTIVATION_FUNCTION = ",A)') this%transfer%name
    write(unit,'(3X,"ACTIVATION_SCALE = ",1ES20.10)') this%transfer%scale
    write(unit,'("END ACTV")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_actv
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  subroutine read_actv(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none
    class(actv_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    integer :: stat
    integer :: itmp1
    real(real32) :: activation_scale
    integer, dimension(3) :: input_shape
    character(256) :: buffer, tag, err_msg
    character(20) :: activation_function


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! loop over tags in layer card
    !!--------------------------------------------------------------------------
    tag_loop: do

       !! check for end of file
       !!-----------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       !! check for end of layer card
       !!-----------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END ACTV")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from save file
       !!-----------------------------------------------------------------------
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case default
          !! don't look for "e" due to scientific notation of numbers
          !! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          write(err_msg,'("Unrecognised line in input file: ",A)') &
               trim(adjustl(buffer))
          call stop_program(err_msg)
          return
       end select
    end do tag_loop


    !!--------------------------------------------------------------------------
    !! allocate layer
    !!--------------------------------------------------------------------------
    call this%set_hyperparams( &
         activation_function = activation_function, &
         activation_scale = activation_scale &
    )
    call this%init(input_shape = input_shape)


    !!--------------------------------------------------------------------------
    !! check for end of layer card
    !!--------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END ACTV")then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_actv
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_actv_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=actv_layer_type("none"))
    call layer%read(unit, verbose=verbose_)

 end function read_actv_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
   pure subroutine forward_assumed_rank(this, input)
     implicit none
     class(actv_layer_type), intent(inout) :: this
     real(real32), dimension(..), intent(in), target :: input

     real(real32), pointer :: input_ptr(:,:)

    select rank(input)
    rank(1)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(2)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(3)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(4)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(5)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    end select
    this%output%val(:,:) = this%transfer%activate(input_ptr)

  end subroutine forward_assumed_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_assumed_rank(this, input, gradient)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real32), dimension(..), intent(in), target :: input
    real(real32), dimension(..), intent(in), target :: gradient

    real(real32), pointer :: input_ptr(:,:), gradient_ptr(:,:)

    select rank(gradient)
    rank(1)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(2)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(3)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(4)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    rank(5)
       gradient_ptr(1:product(this%input_shape),1:this%batch_size) => gradient
    end select

    select rank(input)
    rank(1)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(2)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(3)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(4)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    rank(5)
       input_ptr(1:product(this%input_shape),1:this%batch_size) => input
    end select
    this%di%val(:,:) = gradient_ptr * this%transfer%differentiate(input_ptr)

  end subroutine backward_assumed_rank
!!!#############################################################################

end module athena__actv_layer
!!!#############################################################################
