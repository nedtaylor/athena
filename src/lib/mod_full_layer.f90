!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a fully connected (dense) layer
!!!#############################################################################
!!! Attribution statement:
!!! The following procedures are based on code from the neural-fortran library
!!! https://github.com/modern-fortran/neural-fortran/blob/main/src/nf/nf_layer.f90
!!! procedures:
!!! - get_num_params*
!!!#############################################################################
module athena__full_layer
  use athena__io_utils, only: stop_program
  use athena__constants, only: real32
  use athena__base_layer, only: learnable_layer_type, base_layer_type
  use athena__misc_types, only: activation_type, initialiser_type, &
       array2d_type
  implicit none
  

!!!-----------------------------------------------------------------------------
!!! fully connected network layer type
!!!-----------------------------------------------------------------------------
  type, extends(learnable_layer_type) :: full_layer_type
     integer :: num_inputs
     integer :: num_outputs
     real(real32), pointer :: weight(:,:) => null()
     real(real32), pointer :: dw(:,:,:) => null() ! weight gradient
     real(real32), allocatable, dimension(:,:) :: z ! activation
   contains
     procedure, pass(this) :: get_num_params => get_num_params_full

     procedure, pass(this) :: print => print_full
     procedure, pass(this) :: read => read_full
     procedure, pass(this) :: set_hyperparams => set_hyperparams_full
     procedure, pass(this), private :: &
          set_ptrs_hyperparams => set_ptrs_hyperparams_full
     procedure, pass(this) :: init => init_full
     procedure, pass(this) :: set_batch_size => set_batch_size_full
     
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_2d
     procedure, private, pass(this) :: backward_2d

     final :: finalise_full
  end type full_layer_type


!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  interface full_layer_type
     module function layer_setup( &
          num_outputs, num_inputs, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser) result(layer)
       integer, intent(in) :: num_outputs
       integer, optional, intent(in) :: num_inputs
       integer, optional, intent(in) :: batch_size
       real(real32), optional, intent(in) :: activation_scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser
       type(full_layer_type) :: layer
     end function layer_setup
  end interface full_layer_type


  private
  public :: full_layer_type
  public :: read_full_layer


contains

!!!#############################################################################
!!! finalise layer
!!!#############################################################################
  subroutine finalise_full(this)
    implicit none
    type(full_layer_type), intent(inout) :: this

    if(associated(this%weight)) nullify(this%weight)
    if(associated(this%dw)) nullify(this%dw)
    if(allocated(this%z)) deallocate(this%z)
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output)) deallocate(this%output)
    if(allocated(this%di)) deallocate(this%di)

  end subroutine finalise_full
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! get number of parameters
!!! procedure modified from neural-fortran library
!!!#############################################################################
  pure function get_num_params_full(this) result(num_params)
    implicit none
    class(full_layer_type), intent(in) :: this
    integer :: num_params

    num_params = ( this%num_inputs + 1 )* this%num_outputs

  end function get_num_params_full
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
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
    class(full_layer_type), intent(inout) :: this
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
       num_outputs, num_inputs, &
       batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, verbose) result(layer)
    implicit none
    integer, intent(in) :: num_outputs
    integer, optional, intent(in) :: num_inputs
    integer, optional, intent(in) :: batch_size
    real(real32), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose
    
    type(full_layer_type) :: layer

    integer :: verbose_ = 0
    real(real32) :: scale = 1._real32
    character(len=10) :: activation_function_ = "none"


    if(present(verbose)) verbose_ = verbose

    !!--------------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!--------------------------------------------------------------------------
    if(present(activation_function)) activation_function_ = activation_function
    if(present(activation_scale)) scale = activation_scale


    !!--------------------------------------------------------------------------
    !! define weights (kernels) and biases initialisers
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser
     

    !!--------------------------------------------------------------------------
    !! set hyperparameters
    !!--------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         activation_function = activation_function_, &
         activation_scale = scale, &
         kernel_initialiser = layer%kernel_initialiser, &
         bias_initialiser = layer%bias_initialiser, &
         verbose = verbose_ &
    )


    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! set hyperparameters
!!!#############################################################################
  subroutine set_hyperparams_full( &
       this, num_outputs, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       verbose )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser
    implicit none
    class(full_layer_type), intent(inout) :: this
    integer, intent(in) :: num_outputs
    character(*), intent(in) :: activation_function
    real(real32), intent(in) :: activation_scale
    character(*), intent(in) :: kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose


    this%name = "full"
    this%type = "full"
    this%input_rank = 1
    this%num_outputs = num_outputs
    allocate(this%transfer, &
         source=activation_setup(activation_function, activation_scale))
    if(trim(kernel_initialiser).eq.'') &
         this%kernel_initialiser = get_default_initialiser(activation_function)
    if(trim(bias_initialiser).eq.'') &
         this%bias_initialiser = get_default_initialiser( &
              activation_function, &
              is_bias=.true. &
         )
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("FULL activation function: ",A)') &
               trim(activation_function)
          write(*,'("FULL kernel initialiser: ",A)') &
               trim(this%kernel_initialiser)
          write(*,'("FULL bias initialiser: ",A)') &
               trim(this%bias_initialiser)
       end if
    end if

  end subroutine set_hyperparams_full
!!!#############################################################################


!!!#############################################################################
!!! set the pointers to hyperparameters
!!!#############################################################################
  subroutine set_ptrs_hyperparams_full(this)
    implicit none
    class(full_layer_type), intent(inout), target :: this

    if(allocated(this%params)) &
         this%weight(1:this%num_outputs,1:this%num_inputs+1) => this%params
    if(allocated(this%dp)) &
         this%dw(1:this%num_outputs,1:this%num_inputs,1:this%batch_size) => &
              this%dp

  end subroutine set_ptrs_hyperparams_full
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_full(this, input_shape, batch_size, verbose)
    use athena__initialiser, only: initialiser_setup
    implicit none
    class(full_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0
    class(initialiser_type), allocatable :: initialiser_


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise number of inputs
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%num_inputs = this%input_shape(1)
    if(allocated(this%output))then
       if(this%output%allocated) call this%output%deallocate()
    end if
    this%output = array2d_type()
    this%output%shape = [this%num_outputs]
    this%num_params = this%get_num_params()


    !!--------------------------------------------------------------------------
    !! allocate weight, weight steps (velocities), output, and activation
    !!--------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_params), source=0._real32)


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise( &
         this%params(:this%num_params-this%num_outputs), &
         fan_in = this%num_inputs + 1, fan_out = this%num_outputs, &
         spacing = [ this%num_outputs ] &
    )
    deallocate(initialiser_)

    !! initialise biases
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%bias_initialiser))
    call initialiser_%initialise( &
         this%params(this%num_params-this%num_outputs+1:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs &
    )
    deallocate(initialiser_)


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_full
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_full(this, batch_size, verbose)
    implicit none
    class(full_layer_type), intent(inout), target :: this
    integer, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: verbose_ = 0


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose
    this%batch_size = batch_size

    !!--------------------------------------------------------------------------
    !! set weights and biases pointers to params array
    !!--------------------------------------------------------------------------
    this%weight(1:this%num_outputs,1:this%num_inputs+1) => this%params


    !!--------------------------------------------------------------------------
    !! allocate arrays
    !!--------------------------------------------------------------------------
    if(allocated(this%input_shape))then
       if(.not.allocated(this%output)) this%output = array2d_type()
       if(this%output%allocated) call this%output%deallocate(keep_shape=.true.)
       call this%output%allocate( &
            [this%num_outputs, this%batch_size], &
            source=0._real32 &
       )
       if(allocated(this%z)) deallocate(this%z)
       select type(output => this%output)
       type is (array2d_type)
          allocate( this%z, source = output%val )
       end select
       if(allocated(this%dp)) deallocate(this%dp)
       allocate(this%dp(this%num_params - this%num_outputs, this%batch_size), source=0._real32)
       this%dw(1:this%num_outputs,1:this%num_inputs,1:this%batch_size) => &
            this%dp
       if(allocated(this%db)) deallocate(this%db)
       allocate(this%db(this%num_outputs, this%batch_size), source=0._real32)
      !  if(allocated(this%dw)) deallocate(this%dw)
      !  allocate(this%dw(this%num_outputs, this%num_inputs+1, this%batch_size), &
      !       source=0._real32)
       if(.not.allocated(this%di)) this%di = array2d_type()
       if(this%di%allocated) call this%di%deallocate()
       call this%di%allocate( &
            [this%num_inputs, this%batch_size], &
            source=0._real32 &
       )
    end if

  end subroutine set_batch_size_full
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_full(this, file)
    implicit none
    class(full_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: i, unit


    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! Write initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("FULL")')
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale

    !! write fully connected weights and biases
    !!--------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do i=1,this%num_inputs+1
       write(unit,'(5(E16.8E2))') this%weight(:,i)
    end do
    write(unit,'("END WEIGHTS")')
    write(unit,'("END FULL")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_full
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  subroutine read_full(this, unit, verbose)
    use athena__tools_infile, only: assign_val, assign_vec
    use athena__misc, only: to_lower, to_upper, icount
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    class(full_layer_type), intent(inout) :: this

    integer :: stat, verbose_ = 0
    integer :: i, j, k, c, itmp1
    integer :: num_inputs, num_outputs
    real(real32) :: activation_scale
    logical :: found_weights = .false.
    character(14) :: kernel_initialiser='', bias_initialiser=''
    character(20) :: activation_function
    character(256) :: buffer, tag, err_msg

    real(real32), allocatable, dimension(:) :: data_list


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
       if(trim(adjustl(buffer)).eq."END FULL")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from file
       !!-----------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("ACTIVATION")
          call assign_val(buffer, activation_function, itmp1)
       case("ACTIVATION_SCALE")
          call assign_val(buffer, activation_scale, itmp1)
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("WEIGHTS")
          found_weights = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
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
         num_outputs = num_outputs, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs])


    !! check if WEIGHTS card was found
    !!--------------------------------------------------------------------------
    if(.not.found_weights)then
      write(0,*) "WARNING: WEIGHTS card in FULL not found"
    else
       do i=1,num_inputs+1
          allocate(data_list((num_outputs)), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_outputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%weight(:,i) = data_list
          deallocate(data_list)
       end do

       !! check for end of weights card
       !!-----------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(0,*) trim(adjustl(buffer))
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if


    !!--------------------------------------------------------------------------
    !! check for end of layer card
    !!--------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END FULL")then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_full
!!!#############################################################################


!!!#############################################################################
!!! read layer from file and return layer
!!!#############################################################################
  function read_full_layer(unit, verbose) result(layer)
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: verbose_ = 0


    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=full_layer_type(num_outputs=0))
    call layer%read(unit, verbose=verbose_)

  end function read_full_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_2d(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input

    integer :: s


    !! generate outputs from weights, biases, and inputs
    do concurrent(s=1:this%batch_size)
       this%z(:,s) = this%weight(:,this%num_inputs+1) + &
            matmul(this%weight(:,:this%num_inputs),input(:,s))
    end do

    !! apply activation function to activation
    this%output%val(:,:) = this%transfer%activate(this%z)

  end subroutine forward_2d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
  pure subroutine backward_2d(this, input, gradient)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real32), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    real(real32), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient

    real(real32), dimension(this%num_outputs, this%batch_size) :: delta

    real(real32), dimension(1) :: bias_diff

    integer :: s, j


    bias_diff = this%transfer%differentiate([1._real32])

    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! delta(l) = g'(a) * dE/dI(l)
    !! delta(l) = differential of activation * error from next layer
    delta = gradient * this%transfer%differentiate(this%z)
    this%db(:,:) = this%db(:,:) + delta * bias_diff(1)

    do concurrent(s=1:this%batch_size)
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       do j = 1, this%num_inputs
          this%dw(:,j,s) = this%dw(:,j,s) + input(j,s) * delta(:,s)
       end do
       !! the errors are summed from the delta of the ...
       !! ... 'child' node * 'child' weight
       !! dE/dI(l-1) = sum(weight(l) * delta(l))
       !! this prepares dE/dI for when it is passed into the previous layer
       this%di%val(:,s) = matmul(delta(:,s), this%weight(:,:this%num_inputs))
    end do

  end subroutine backward_2d
!!!#############################################################################

end module athena__full_layer
!!!#############################################################################
