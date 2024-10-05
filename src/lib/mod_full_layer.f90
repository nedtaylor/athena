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
!!! - get_params*
!!! - set_params*
!!! - get_gradients*
!!! - set_gradients*
!!!#############################################################################
module full_layer
  use constants, only: real12
  use base_layer, only: learnable_layer_type
  use custom_types, only: activation_type, initialiser_type
  implicit none
  

!!!-----------------------------------------------------------------------------
!!! fully connected network layer type
!!!-----------------------------------------------------------------------------
  type, extends(learnable_layer_type) :: full_layer_type
     integer :: num_inputs, num_addit_inputs = 0
     integer :: num_outputs
     real(real12), allocatable, dimension(:,:) :: weight
     real(real12), allocatable, dimension(:,:,:) :: dw ! weight gradient
     real(real12), allocatable, dimension(:,:) :: output, z !output and activation
     real(real12), allocatable, dimension(:,:) :: di ! input gradient (i.e. delta)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_full
     procedure, pass(this) :: get_params => get_params_full
     procedure, pass(this) :: set_params => set_params_full
     procedure, pass(this) :: get_gradients => get_gradients_full
     procedure, pass(this) :: set_gradients => set_gradients_full
     procedure, pass(this) :: get_output => get_output_full

     procedure, pass(this) :: print => print_full
     procedure, pass(this) :: set_shape => set_shape_full
     procedure, pass(this) :: init => init_full
     procedure, pass(this) :: set_batch_size => set_batch_size_full
     
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_2d
     procedure, private, pass(this) :: backward_2d

     procedure, pass(this) :: reduce => layer_reduction
     procedure, pass(this) :: merge => layer_merge
     procedure :: add_t_t => layer_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
  end type full_layer_type


!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  interface full_layer_type
     module function layer_setup( &
          num_outputs, num_inputs, num_addit_inputs, batch_size, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser) result(layer)
       integer, intent(in) :: num_outputs
       integer, optional, intent(in) :: num_inputs, num_addit_inputs
       integer, optional, intent(in) :: batch_size
       real(real12), optional, intent(in) :: activation_scale
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
!!! layer reduction
!!!#############################################################################
  subroutine layer_reduction(this, rhs)
    implicit none
    class(full_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: rhs

    select type(rhs)
    type is(full_layer_type)
       this%dw = this%dw + rhs%dw
    end select

  end subroutine  layer_reduction
!!!#############################################################################


!!!#############################################################################
!!! layer addition
!!!#############################################################################
  function layer_add(a, b) result(output)
    implicit none
    class(full_layer_type), intent(in) :: a, b
    type(full_layer_type) :: output

    output = a
    output%dw = output%dw + b%dw

  end function layer_add
!!!#############################################################################


!!!#############################################################################
!!! layer merge
!!!#############################################################################
  subroutine layer_merge(this, input)
    implicit none
    class(full_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    select type(input)
    class is(full_layer_type)
       this%dw = this%dw + input%dw
    end select

  end subroutine layer_merge
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


!!!#############################################################################
!!! get number of parameters
!!! procedure modified from neural-fortran library
!!!#############################################################################
  pure function get_params_full(this) result(params)
    implicit none
    class(full_layer_type), intent(in) :: this
    real(real12), allocatable, dimension(:) :: params
  
    params = reshape(this%weight, [ (this%num_inputs+1) * this%num_outputs ])
  
  end function get_params_full
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!! procedure modified from neural-fortran library
!!!#############################################################################
  subroutine set_params_full(this, params)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(:), intent(in) :: params
  
    this%weight = reshape(params, [ this%num_inputs+1, this%num_outputs ])
  
  end subroutine set_params_full
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!! procedure modified from neural-fortran library
!!!#############################################################################
  pure function get_gradients_full(this, clip_method) result(gradients)
    use clipper, only: clip_type
    implicit none
    class(full_layer_type), intent(in) :: this
    type(clip_type), optional, intent(in) :: clip_method
    real(real12), allocatable, dimension(:) :: gradients
  
    gradients = reshape(sum(this%dw,dim=3)/this%batch_size, &
         [ (this%num_inputs+1) * this%num_outputs ])

    if(present(clip_method)) call clip_method%apply(size(gradients),gradients)
  
  end function get_gradients_full
!!!#############################################################################


!!!#############################################################################
!!! set gradients
!!! procedure modified from neural-fortran library
!!!#############################################################################
  subroutine set_gradients_full(this, gradients)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: gradients
  
    select rank(gradients)
    rank(0)
       this%dw = gradients
    rank(1)
       this%dw = spread(reshape(gradients, shape(this%dw(:,:,1))), 3, &
            this%batch_size)
    end select
  
  end subroutine set_gradients_full
!!!#############################################################################


!!!#############################################################################
!!! get layer outputs
!!!#############################################################################
  pure subroutine get_output_full(this, output)
  implicit none
  class(full_layer_type), intent(in) :: this
  real(real12), allocatable, dimension(..), intent(out) :: output

  select rank(output)
  rank(1)
     output = reshape(this%output, [size(this%output)])
  rank(2)
     output = this%output
  end select

end subroutine get_output_full
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
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

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
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

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
       num_outputs, num_inputs, num_addit_inputs, &
       batch_size, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser) result(layer)
    use activation,  only: activation_setup
    use initialiser, only: get_default_initialiser
    implicit none
    integer, intent(in) :: num_outputs
    integer, optional, intent(in) :: num_inputs, num_addit_inputs
    integer, optional, intent(in) :: batch_size
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser
    
    type(full_layer_type) :: layer

    real(real12) :: scale
    character(len=10) :: activation_function_


    layer%name = "full"
    layer%input_rank = 1
    !!--------------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!--------------------------------------------------------------------------
    if(present(activation_function))then
       activation_function_ = activation_function
    else
       activation_function_ = "none"
    end if
    if(present(activation_scale))then
       scale = activation_scale
    else
       scale = 1._real12
    end if
    write(*,'("FULL activation function: ",A)') trim(activation_function_)
    allocate(layer%transfer, &
         source=activation_setup(activation_function_, scale))
    

    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! define weights (kernels) and biases initialisers
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser
    if(trim(layer%kernel_initialiser).eq.'') &
         layer%kernel_initialiser=get_default_initialiser(activation_function_)
    write(*,'("FULL kernel initialiser: ",A)') trim(layer%kernel_initialiser)
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser
    if(trim(layer%bias_initialiser).eq.'') &
         layer%bias_initialiser = get_default_initialiser(&
         activation_function_, is_bias=.true.)       
    write(*,'("FULL bias initialiser: ",A)') trim(layer%bias_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    layer%num_outputs = num_outputs
    if(present(num_addit_inputs)) layer%num_addit_inputs = num_addit_inputs
    if(present(num_inputs)) call layer%init(input_shape=[num_inputs])

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! setup input layer shape
!!!#############################################################################
  subroutine set_shape_full(this, input_shape)
   implicit none
   class(full_layer_type), intent(inout) :: this
   integer, dimension(:), intent(in) :: input_shape

   !!--------------------------------------------------------------------------
   !! initialise input shape
   !!--------------------------------------------------------------------------
   if(size(input_shape,dim=1).eq.this%input_rank)then
      this%num_inputs = input_shape(1) + this%num_addit_inputs
   else
      this%num_inputs  = product(input_shape) + this%num_addit_inputs
      !stop "ERROR: invalid size of input_shape in full, expected (1)"
   end if
   this%input_shape = [this%num_inputs]

 end subroutine set_shape_full
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_full(this, input_shape, batch_size, verbose)
    use initialiser, only: initialiser_setup
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
    this%output_shape = [this%num_outputs]


    !!--------------------------------------------------------------------------
    !! allocate weight, weight steps (velocities), output, and activation
    !!--------------------------------------------------------------------------
    allocate(this%weight(this%num_inputs+1,this%num_outputs), source=0._real12)


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%kernel_initialiser))
    call initialiser_%initialise(this%weight(:this%num_inputs,:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs)
    deallocate(initialiser_)

    !! initialise biases
    !!--------------------------------------------------------------------------
    allocate(initialiser_, source=initialiser_setup(this%bias_initialiser))
    call initialiser_%initialise(this%weight(this%num_inputs+1,:), &
         fan_in=this%num_inputs+1, fan_out=this%num_outputs)
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
   class(full_layer_type), intent(inout) :: this
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
      allocate(this%output(this%num_outputs, this%batch_size), source=0._real12)
      if(allocated(this%z)) deallocate(this%z)
      allocate(this%z, source=this%output)
      if(allocated(this%dw)) deallocate(this%dw)
      allocate(this%dw(this%num_inputs+1,this%num_outputs, this%batch_size), &
           source=0._real12)
      if(allocated(this%di)) deallocate(this%di)
      allocate(this%di(this%num_inputs, this%batch_size), source=0._real12)
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

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("FULL")')
    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_inputs
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale

    !! write fully connected weights and biases
    !!--------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do i=1,this%num_outputs
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
  function read_full_layer(unit, verbose) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    class(full_layer_type), allocatable :: layer

    integer :: stat, verbose_ = 0
    integer :: i, j, k, c, itmp1
    integer :: num_inputs, num_outputs
    real(real12) :: activation_scale
    logical :: found_weights = .false.
    character(14) :: kernel_initialiser='', bias_initialiser=''
    character(20) :: activation_function
    character(256) :: buffer, tag

    real(real12), allocatable, dimension(:) :: data_list


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
         write(0,*) "ERROR: file encountered error (EoF?) before END FULL"
          stop "Exiting..."
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
          stop "Unrecognised line in input file: "//trim(adjustl(buffer))
       end select
    end do tag_loop


    !!--------------------------------------------------------------------------
    !! allocate layer
    !!--------------------------------------------------------------------------
    layer = full_layer_type( &
         num_outputs = num_outputs, num_inputs = num_inputs, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser)

    !! check if WEIGHTS card was found
    !!--------------------------------------------------------------------------
    if(.not.found_weights)then
      write(0,*) "WARNING: WEIGHTS card in FULL not found"
    else
       do i=1,num_outputs
          allocate(data_list((num_inputs+1)), source=0._real12)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_inputs+1)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          layer%weight(:,i) = data_list
          deallocate(data_list)
       end do

       !! check for end of weights card
       !!-----------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(*,*) trim(adjustl(buffer))
          stop "ERROR: END WEIGHTS not where expected"
       end if
    end if


    !!--------------------------------------------------------------------------
    !! check for end of layer card
    !!--------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END FULL")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END FULL not where expected"
    end if

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
    real(real12), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input

    integer :: s


    !! generate outputs from weights, biases, and inputs
    do concurrent(s=1:this%batch_size)
       this%z(:,s) = this%weight(this%num_inputs+1,:) + &
            matmul(input(:,s),this%weight(:this%num_inputs,:))
    end do

    !! apply activation function to activation
    this%output = this%transfer%activate(this%z)

  end subroutine forward_2d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
  pure subroutine backward_2d(this, input, gradient)
    implicit none
    class(full_layer_type), intent(inout) :: this
    real(real12), dimension(this%num_inputs, this%batch_size), &
         intent(in) :: input
    real(real12), dimension(this%num_outputs, this%batch_size), &
         intent(in) :: gradient

    real(real12), dimension(this%num_outputs, this%batch_size) :: delta
    real(real12), dimension(&
         this%num_inputs, this%num_outputs, this%batch_size) :: dw

    real(real12), dimension(1) :: bias_diff

    integer :: s


    if(.not.allocated(this%transfer)) return
    bias_diff = this%transfer%differentiate([1._real12])

    !! the delta values are the error multipled by the derivative ...
    !! ... of the transfer function
    !! delta(l) = g'(a) * dE/dI(l)
    !! delta(l) = differential of activation * error from next layer
    delta(:,:) = gradient * this%transfer%differentiate(this%z)

    do concurrent(s=1:this%batch_size)
       !! partial derivatives of error wrt weights
       !! dE/dW = o/p(l-1) * delta
       dw(:,:,s) = matmul(input(:,s:s), transpose(delta(:,s:s)))

       !! the errors are summed from the delta of the ...
       !! ... 'child' node * 'child' weight
       !! dE/dI(l-1) = sum(weight(l) * delta(l))
       !! this prepares dE/dI for when it is passed into the previous layer
       this%di(:,s) = matmul(this%weight(:this%num_inputs,:), delta(:,s))
    end do

    !! sum weights and biases errors to use in batch gradient descent
    delta = delta * bias_diff(1)
    this%dw(:this%num_inputs,:,:)  = this%dw(:this%num_inputs,:,:)  + dw
    this%dw(this%num_inputs+1,:,:) = this%dw(this%num_inputs+1,:,:) + delta(:,:)

  end subroutine backward_2d
!!!#############################################################################

end module full_layer
!!!#############################################################################
