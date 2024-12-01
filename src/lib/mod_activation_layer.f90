!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of an activation layer
!!!#############################################################################
module actv_layer
  use constants, only: real12
  use base_layer, only: base_layer_type
  use custom_types, only: activation_type
  implicit none
  
  
  type, extends(base_layer_type) :: actv_layer_type
     class(activation_type), allocatable :: transfer
     real(real12), allocatable, dimension(:,:) :: output
     real(real12), allocatable, dimension(:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: get_output => get_output_actv
     procedure, pass(this) :: init => init_actv
     procedure, pass(this) :: set_batch_size => set_batch_size_actv
     procedure, pass(this) :: print => print_actv
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
       real(real12), optional, intent(in) :: activation_scale
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
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_actv(this, output)
    implicit none
    class(actv_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(..), intent(out) :: output
  
    select rank(output)
    rank(1)
       output = reshape(this%output, [size(this%output)])
    rank(2)
       output = this%output
    rank(3)
       output = reshape( &
            this%output,  &
            [ this%output_shape(1), this%output_shape(2), this%batch_size ] &
       )
    rank(4)
       output = reshape( &
            this%output,  &
            [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%output_shape(3), this%batch_size ] &
       )
    rank(5)
       output = reshape( &
            this%output,  &
            [ &
                 this%output_shape(1), &
                 this%output_shape(2), &
                 this%output_shape(3), &
                 this%output_shape(4), this%batch_size ] &
       )
    end select
  
  end subroutine get_output_actv
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    call this%forward_assumed_rank(input)
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

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
  use activation,  only: activation_setup
    implicit none
    character(*), intent(in) :: activation_function
    real(real12), optional, intent(in) :: activation_scale
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
    
    type(actv_layer_type) :: layer

    real(real12) :: activation_scale_
    integer :: verbose_


    verbose_ = 0
    if(present(verbose)) verbose_ = verbose
    layer%name = "actv"
    activation_scale_ = 1._real12
    if(present(activation_scale)) activation_scale_ = activation_scale
    if(verbose_.gt.0) write(*,'("ACTV activation function: ",A)') &
         trim(activation_function)
    allocate( &
         layer%transfer, &
         source = activation_setup(activation_function, activation_scale_) &
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


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    allocate(this%output_shape(3))
    this%output_shape = this%input_shape
    

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
    class(actv_layer_type), intent(inout) :: this
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
       allocate(this%output( &
            product(this%output_shape), &
            this%batch_size), &
            source=0._real12 &
       )
       if(allocated(this%di)) deallocate(this%di)
       allocate(this%di( &
            product(this%input_shape), &
            this%batch_size), &
            source=0._real12 &
       )
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
  function read_actv_layer(unit) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit

    class(actv_layer_type), allocatable :: layer

    integer :: stat
    integer :: itmp1
    real(real12) :: activation_scale
    integer, dimension(3) :: input_shape
    character(256) :: buffer, tag
    character(20) :: activation_function


    !! loop over tags in layer card
    tag_loop: do

       !! check for end of file
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(0,*) "ERROR: file encountered error (EoF?) before END ACTV"
          stop "Exiting..."
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       !! check for end of convolution card
       if(trim(adjustl(buffer)).eq."END ACTV")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from save file
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
          stop "Unrecognised line in input file: "//trim(adjustl(buffer))
       end select
    end do tag_loop

    !! set transfer activation function

    layer = actv_layer_type( &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         input_shape=input_shape &
    )

    !! check for end of layer card
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END ACTV")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END ACTV not where expected"
    end if

  end function read_actv_layer
!!!#############################################################################


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
   pure subroutine forward_assumed_rank(this, input)
     implicit none
     class(actv_layer_type), intent(inout) :: this
     real(real12), dimension(..), intent(in), target :: input

     real(real12), pointer :: input_ptr(:,:)

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
    this%output(:,:) = this%transfer%activate(input_ptr)

  end subroutine forward_assumed_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
  pure subroutine backward_assumed_rank(this, input, gradient)
    implicit none
    class(actv_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in), target :: input
    real(real12), dimension(..), intent(in), target :: gradient

    real(real12), pointer :: input_ptr(:,:), gradient_ptr(:,:)

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
    this%di(:,:) = gradient_ptr * this%transfer%differentiate(input_ptr)

  end subroutine backward_assumed_rank
!!!#############################################################################

end module actv_layer
!!!#############################################################################
