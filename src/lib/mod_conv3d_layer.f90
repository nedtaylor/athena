!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module conv3d_layer
  use constants, only: real12
  use base_layer, only: learnable_layer_type
  use custom_types, only: activation_type, initialiser_type
  implicit none
  
  
  type, extends(learnable_layer_type) :: conv3d_layer_type
     !! knl = kernel
     !! stp = stride (step)
     !! hlf = half
     !! pad = pad
     !! cen = centre
     !! output_shape = dimension (height, width, depth)
     logical :: calc_input_gradients = .true.
     integer, dimension(3) :: knl, stp, hlf, pad, cen
     integer :: num_channels
     integer :: num_filters
     real(real12), allocatable, dimension(:) :: bias, bias_incr
     real(real12), allocatable, dimension(:) :: db ! bias gradient
     real(real12), allocatable, dimension(:,:,:,:,:) :: weight, weight_incr
     real(real12), allocatable, dimension(:,:,:,:,:) :: dw ! weight gradient
     real(real12), allocatable, dimension(:,:,:,:) :: output, z
     real(real12), allocatable, dimension(:,:,:,:) :: di ! input gradient
     class(activation_type), allocatable :: transfer
   contains
     procedure, pass(this) :: init => init_conv3d
     procedure, pass(this) :: print => print_conv3d
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, pass(this) :: update
     procedure, private, pass(this) :: forward_4d
     procedure, private, pass(this) :: backward_4d

     procedure, pass(this) :: reduce => layer_reduction
     procedure, pass(this) :: merge => layer_merge
     procedure :: add_t_t => layer_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
  end type conv3d_layer_type

  
!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  interface conv3d_layer_type
     module function layer_setup( &
          input_shape, &
          num_filters, kernel_size, stride, padding, &
          activation_function, activation_scale, &
          kernel_initialiser, bias_initialiser, &
          calc_input_gradients) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: num_filters
       integer, dimension(..), optional, intent(in) :: kernel_size
       integer, dimension(..), optional, intent(in) :: stride
       real(real12), optional, intent(in) :: activation_scale
       character(*), optional, intent(in) :: activation_function, &
            kernel_initialiser, bias_initialiser, padding
       logical, optional, intent(in) :: calc_input_gradients
       type(conv3d_layer_type) :: layer
     end function layer_setup
  end interface conv3d_layer_type


  private
  public :: conv3d_layer_type
  public :: read_conv3d_layer


contains

!!!#############################################################################
!!! layer reduction
!!!#############################################################################
  subroutine layer_reduction(this, rhs)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: rhs

    select type(rhs)
    type is(conv3d_layer_type)
       this%db = this%db + rhs%db
       this%dw = this%dw + rhs%dw
    end select

  end subroutine  layer_reduction
!!!#############################################################################


!!!#############################################################################
!!! layer addition
!!!#############################################################################
  function layer_add(a, b) result(output)
    implicit none
    class(conv3d_layer_type), intent(in) :: a, b
    type(conv3d_layer_type), allocatable :: output

    output = a
    output%dw = output%dw + b%dw
    output%db = output%db + b%db

  end function layer_add
!!!#############################################################################


!!!#############################################################################
!!! layer merge
!!!#############################################################################
  subroutine layer_merge(this, input)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    select type(input)
    class is(conv3d_layer_type)
       this%dw = this%dw + input%dw
       this%db = this%db + input%db
    end select

  end subroutine layer_merge
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(4)
       call forward_4d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(4)
    select rank(gradient); rank(4)
       call backward_4d(this, input, gradient)
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
       input_shape, &
       num_filters, kernel_size, stride, padding, &
       activation_function, activation_scale, &
       kernel_initialiser, bias_initialiser, &
       calc_input_gradients) result(layer)
    !! add in dilation
    use activation,  only: activation_setup
    use initialiser, only: get_default_initialiser
    use misc_ml, only: set_padding
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: num_filters
    integer, dimension(..), optional, intent(in) :: kernel_size
    integer, dimension(..), optional, intent(in) :: stride
    real(real12), optional, intent(in) :: activation_scale
    character(*), optional, intent(in) :: activation_function, &
         kernel_initialiser, bias_initialiser, padding
    logical, optional, intent(in) :: calc_input_gradients

    type(conv3d_layer_type) :: layer

    integer :: i
    real(real12) :: scale
    character(len=10) :: t_activation_function, initialiser_name
    character(len=20) :: t_padding
    class(initialiser_type), allocatable :: initialiser

    integer, dimension(3) :: end_idx


    !!--------------------------------------------------------------------------
    !! set up number of filters
    !!--------------------------------------------------------------------------
    if(present(calc_input_gradients))then
       layer%calc_input_gradients = calc_input_gradients
       write(*,*) "CONV3D input gradients turned off"
    else
       layer%calc_input_gradients = .true.
    end if


    !!--------------------------------------------------------------------------
    !! set up number of filters
    !!--------------------------------------------------------------------------
    if(present(num_filters))then
       layer%num_filters = num_filters
    else
       layer%num_filters = 32
    end if
    
    
    !!--------------------------------------------------------------------------
    !! set up kernel size
    !!--------------------------------------------------------------------------
    if(present(kernel_size))then
       select rank(kernel_size)
       rank(0)
          layer%knl = kernel_size
       rank(1)
          layer%knl(1) = kernel_size(1)
          if(size(kernel_size,dim=1).eq.1)then
             layer%knl(2:) = kernel_size(1)
          elseif(size(kernel_size,dim=1).eq.3)then
             layer%knl(2:) = kernel_size(2:)
          end if
       end select
    else
       layer%knl = 3
    end if
    !! odd or even kernel/filter size
    !!--------------------------------------------------------------------------
    layer%cen = 2 - mod(layer%knl, 2)
    layer%hlf   = (layer%knl-1)/2

    if(present(padding))then
       t_padding = padding
    else
       t_padding = "valid"
    end if
    do i=1,3
       call set_padding(layer%pad(i), layer%knl(i), t_padding)
    end do


    !!--------------------------------------------------------------------------
    !! set up stride
    !!--------------------------------------------------------------------------
    if(present(stride))then
       select rank(stride)
       rank(0)
          layer%stp = stride
       rank(1)
          layer%stp(1) = stride(1)
          if(size(stride,dim=1).eq.1)then
             layer%stp(2:) = stride(1)
          elseif(size(stride,dim=1).eq.3)then
             layer%stp(2:) = stride(2:)
          end if
       end select
    else
       layer%stp = 1
    end if
    

    !!--------------------------------------------------------------------------
    !! set activation and derivative functions based on input name
    !!--------------------------------------------------------------------------
    if(present(activation_function))then
       t_activation_function = activation_function
    else
       t_activation_function = "none"
    end if
    if(present(activation_scale))then
       scale = activation_scale
    else
       scale = 1._real12
    end if
    write(*,'("CONV3D activation function: ",A)') trim(t_activation_function)
    allocate(layer%transfer, &
         source=activation_setup(t_activation_function, scale))


    !!--------------------------------------------------------------------------
    !! define weights (kernels) and biases initialisers
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser =kernel_initialiser
    if(trim(layer%kernel_initialiser).eq.'') &
         layer%kernel_initialiser=get_default_initialiser(t_activation_function)
    write(*,'("CV kernel initialiser: ",A)') trim(layer%kernel_initialiser)
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser
    if(trim(layer%bias_initialiser).eq.'') &
         layer%bias_initialiser = get_default_initialiser(&
         t_activation_function, is_bias=.true.)
    write(*,'("CV bias initialiser: ",A)') trim(layer%bias_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_conv3d(this, input_shape, verbose)
    use initialiser, only: initialiser_setup
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: t_verb
    integer, dimension(3) :: end_idx
    class(initialiser_type), allocatable :: initialiser


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(size(input_shape,dim=1).eq.4)then
       this%input_shape = input_shape
       this%num_channels = input_shape(4)
    else
       stop "ERROR: invalid size of input_shape in conv3d, expected (4)"
    end if


    !!--------------------------------------------------------------------------
    !! allocate output, activation, bias, and weight shapes
    !!--------------------------------------------------------------------------
    !! NOTE: INPUT SHAPE DOES NOT INCLUDE PADDING WIDTH
    !! THIS IS HANDLED AUTOMATICALLY BY THE CODE
    !! ... provide the initial input data shape and let us deal with the padding
    allocate(this%output_shape(4))
    this%output_shape(4) = this%num_filters
    this%output_shape(:3) = floor(&
         (input_shape(:3) + 2.0 * this%pad - this%knl)/real(this%stp) ) + 1

    allocate(this%output(&
         this%output_shape(1),this%output_shape(2),this%output_shape(3),&
         this%num_filters))
    allocate(this%z, mold=this%output)
    this%z = 0._real12

    allocate(this%bias(this%num_filters))

    end_idx   = this%hlf + (this%cen - 1)
    allocate(this%weight(&
         -this%hlf(1):end_idx(1),&
         -this%hlf(2):end_idx(2),&
         -this%hlf(3):end_idx(3),&
         this%num_channels,this%num_filters))


    !!--------------------------------------------------------------------------
    !! initialise weights and biases steps (velocities)
    !!--------------------------------------------------------------------------
    allocate(this%bias_incr, mold=this%bias)
    allocate(this%weight_incr, mold=this%weight)
    this%bias_incr = 0._real12
    this%weight_incr = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise gradients
    !!--------------------------------------------------------------------------
    allocate(this%di(&
         input_shape(1), input_shape(2), input_shape(3), input_shape(4)), &
         source=0._real12)
    allocate(this%dw, mold=this%weight)
    allocate(this%db, mold=this%bias)
    this%di = 0._real12
    this%dw = 0._real12
    this%db = 0._real12


    !!--------------------------------------------------------------------------
    !! initialise weights (kernels)
    !!--------------------------------------------------------------------------
    allocate(initialiser, source=initialiser_setup(this%kernel_initialiser))
    call initialiser%initialise(this%weight, &
         fan_in=product(this%knl)+1, fan_out=1)
    deallocate(initialiser)

    !! initialise biases
    !!--------------------------------------------------------------------------
    allocate(initialiser, source=initialiser_setup(this%bias_initialiser))
    call initialiser%initialise(this%bias, &
         fan_in=product(this%knl)+1, fan_out=1)
    deallocate(initialiser)

  end subroutine init_conv3d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_conv3d(this, file)
    implicit none
    class(conv3d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: l, i, itmp1, idx
    integer :: unit
    character(:), allocatable :: padding_type


    !! handle different width kernels for x, y, z
    !!--------------------------------------------------------------------------
    itmp1 = -1
    do i=1,3
       if(this%pad(i).gt.itmp1)then
          itmp1 = this%pad(i)
          idx = i
       end if
    end do

    !! determine padding method
    !!--------------------------------------------------------------------------
    padding_type = ""
    if(this%pad(idx).eq.this%knl(idx)-1)then
       padding_type = "full"
    elseif(this%pad(idx).eq.0)then
       padding_type = "valid"
    else
       padding_type = "same"
    end if

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("CONV3D")')
    write(unit,'(3X,"INPUT_SHAPE = ",4(1X,I0))') this%input_shape
    write(unit,'(3X,"NUM_FILTERS = ",I0)') this%num_filters
    if(all(this%knl.eq.this%knl(1)))then
       write(unit,'(3X,"KERNEL_SIZE =",1X,I0)') this%knl(1)
    else
       write(unit,'(3X,"KERNEL_SIZE =",3(1X,I0))') this%knl
    end if
    if(all(this%stp.eq.this%stp(1)))then
       write(unit,'(3X,"STRIDE =",1X,I0)') this%stp(1)
    else
       write(unit,'(3X,"STRIDE =",3(1X,I0))') this%stp
    end if
    write(unit,'(3X,"PADDING = ",A)') padding_type

    write(unit,'(3X,"ACTIVATION = ",A)') trim(this%transfer%name)
    write(unit,'(3X,"ACTIVATION_SCALE = ",F0.9)') this%transfer%scale

    !! write convolution weights and biases
    !!--------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do l=1,this%num_filters
       write(unit,'(5(E16.8E2))', advance="no") this%weight(:,:,:,:,l)
       if(mod(size(this%weight(:,:,:,:,l)),5).eq.0) write(unit,*)
       write(unit,'(E16.8E2)') this%bias(l)
    end do
    write(unit,'("END WEIGHTS")')
    write(unit,'("END CONV3D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_conv3d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_conv3d_layer(unit, verbose) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    class(conv3d_layer_type), allocatable :: layer

    integer :: stat, t_verb
    integer :: j, k, l, c, itmp1
    integer :: num_filters, num_inputs
    real(real12) :: activation_scale
    logical :: found_weights
    character(14) :: kernel_initialiser='', bias_initialiser=''
    character(20) :: padding, activation_function
    character(256) :: buffer, tag

    integer, dimension(3) :: kernel_size, stride
    integer, dimension(4) :: input_shape
    integer, allocatable, dimension(:) :: itmp_list
    real(real12), allocatable, dimension(:) :: data_list


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if


    !!--------------------------------------------------------------------------
    !! loop over tags in layer card
    !!--------------------------------------------------------------------------
    found_weights = .false.
    tag_loop: do

       !! check for end of file
       !!-----------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
         write(0,*) "ERROR: file encountered error (EoF?) before END CONV3D"
          stop "Exiting..."
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       !! check for end of layer card
       !!-----------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END CONV3D")then
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
       case("NUM_FILTERS")
          call assign_val(buffer, num_filters, itmp1)
       case("KERNEL_SIZE")
          call assign_vec(buffer, kernel_size, itmp1)
       case("STRIDE")
          call assign_vec(buffer, stride, itmp1)
       case("PADDING")
          call assign_val(buffer, padding, itmp1)
          padding = to_lower(padding)
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
    layer = conv3d_layer_type( &
         input_shape = input_shape, &
         num_filters = num_filters, &
         kernel_size = kernel_size, stride = stride, &
         padding = padding, &
         activation_function = activation_function, &
         activation_scale = activation_scale, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser)

    !! check if WEIGHTS card was found
    !!--------------------------------------------------------------------------
    if(.not.found_weights)then
      write(0,*) "WARNING: WEIGHTS card in CONV3D not found"
   else
       do l=1,num_filters
          num_inputs = product(layer%knl) + 1 !+1 for bias
          allocate(data_list(num_inputs), source=0._real12)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.num_inputs)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          layer%weight(:,:,:,:,l) = &
                reshape(&
                data_list(1:num_inputs-1),&
                shape(layer%weight(:,:,:,:,l)))
          layer%bias(l) = data_list(num_inputs)
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
    if(trim(adjustl(buffer)).ne."END CONV3D")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END CONV3D not where expected"
    end if

  end function read_conv3d_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_4d(this, input)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    real(real12), &
         dimension(-this%pad(1)+1:,-this%pad(2)+1:,-this%pad(3)+1:,:), &
         intent(in) :: input

    integer :: i, j, k, l
    integer, dimension(3) :: stp_idx, start_idx, end_idx


    !! perform the convolution operation
    !!--------------------------------------------------------------------------
    do concurrent(&
         i=1:this%output_shape(1):1, &
         j=1:this%output_shape(2):1, &
         k=1:this%output_shape(3):1)
       stp_idx = ([i,j,k]-1)*this%stp + 1 + (this%hlf - this%pad)
       start_idx  = stp_idx - this%hlf
       end_idx    = start_idx + this%knl - 1

       this%z(i,j,k,:) = this%bias(:)

       do concurrent(l=1:this%num_filters)
          this%z(i,j,k,l) = this%z(i,j,k,l) + &
               sum( &
               input(&
               start_idx(1):end_idx(1),&
               start_idx(2):end_idx(2),&
               start_idx(3):end_idx(3),:) * &
               this%weight(:,:,:,:,l) &
               )
       end do
    end do
    
    !! apply activation function to activation values (z)
    !!--------------------------------------------------------------------------
    this%output = this%transfer%activate(this%z) 

  end subroutine forward_4d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!! method : gradient descent
!!!#############################################################################
  pure subroutine backward_4d(this, input, gradient)
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    real(real12), &
         dimension(-this%pad(1)+1:,-this%pad(2)+1:,-this%pad(3)+1:,:), &
         intent(in) :: input
    real(real12), &
         dimension(&
         this%output_shape(1),&
         this%output_shape(2),&
         this%output_shape(3),this%num_filters), &
         intent(in) :: gradient


    !! NOTE: lim_w is rotated by 180, so first idx is end, then start
    !! ... i.e. lim_w(1,:) = ends
    !! ... i.e. lim_w(2,:) = starts
    integer :: l, m, i, j, k, x, y, z
    integer, dimension(3) :: stp_idx, offset, adjust, end_idx, n_stp
    integer, dimension(2,3) :: lim, lim_w, lim_g
    real(real12), &
         dimension(&
         this%output_shape(1),&
         this%output_shape(2),&
         this%output_shape(3),this%num_filters) :: grad_dz


    real(real12), dimension(1) :: bias_diff
    bias_diff = this%transfer%differentiate([1._real12])


    !! get size of the input and output feature maps
    !!--------------------------------------------------------------------------
    end_idx = this%hlf + (this%cen - 1)
    offset  = 1 + this%hlf - this%pad
    adjust  = 2 * max(this%pad, this%hlf)


    !! get gradient multiplied by differential of Z
    !!--------------------------------------------------------------------------
    grad_dz = 0._real12
    grad_dz(&
         1:this%output_shape(1),&
         1:this%output_shape(2),&
         1:this%output_shape(3),:) = gradient * &
         this%transfer%differentiate(this%z)
    do concurrent(l=1:this%num_filters)
       this%db(l) = this%db(l) + sum(grad_dz(:,:,:,l)) * bias_diff(1)
    end do

    
    !! apply convolution to compute weight gradients
    !! offset applied as centre of kernel is 0 ...
    !! ... whilst the starting index for input is 1
    !! -1 needed in input upper bound indices to account for size, i.e. ...
    !! ...   x + offset : x + offset + size(input)
    !! ... = 0 + 1      : 0 + 1      + 1 
    !! ... = 1          : 2
    !! ... which would be wrong, as size = 1, so it should be 1:1
    !! ... ergo, we need to subtract 1 from upper index
    !!--------------------------------------------------------------------------
    do concurrent( &
         z=-this%hlf(3):end_idx(3):1, &
         y=-this%hlf(2):end_idx(2):1, &
         x=-this%hlf(1):end_idx(1):1, &
         m=1:this%num_channels, &
         l=1:this%num_filters &
         )
       this%dw(x,y,z,m,l) = this%dw(x,y,z,m,l) + &
            sum(grad_dz(:,:,:,l) * &
            input(&
            x+offset(1):x+offset(1)-1+size(input,1)-adjust(1):this%stp(1), &
            y+offset(2):y+offset(2)-1+size(input,2)-adjust(2):this%stp(2), &
            z+offset(3):z+offset(3)-1+size(input,3)-adjust(3):this%stp(3),m))
    end do


    !! apply strided convolution to obtain input gradients
    !!--------------------------------------------------------------------------
    if(this%calc_input_gradients)then
       lim(1,:) = this%knl - 1
       lim(2,:) = (this%output_shape - 1) * this%stp + 1 + end_idx
       n_stp = this%output_shape * this%stp
       this%di = 0._real12
       !! all elements of the output are separated by stride_x (stride_y)
       do concurrent( &
            i=1:size(this%di,dim=1):1, &
            j=1:size(this%di,dim=2):1, &
            k=1:size(this%di,dim=3):1, &
            m=1:this%num_channels, &
            l=1:this%num_filters &
            )

          !! set weight bounds
          stp_idx = ([i,j,k]-offset)/this%stp + 1
          !! max( ...
          !! ... 1. offset of 1st o/p idx from centre of knl     (lim)
          !! ... 2. lwst o/p idx overlap with <<- knl idx (rpt. pattern)
          !! ...)
          lim_w(2,:) = max(lim(1,:)-[i,j,k],  -this%hlf + &
               mod(n_stp+this%knl-[i,j,k],this%stp))
          !! min( ...
          !! ... 1. offset of last o/p idx from centre of knl    (lim)
          !! ... 2. hghst o/p idx overlap with ->> knl idx (rpt. pattern)
          !! ...)
          lim_w(1,:) = min(lim(2,:)-[i,j,k], end_idx - &
               mod(n_stp-1+[i,j,k],this%stp))
          if(any(lim_w(2,:).gt.lim_w(1,:))) cycle

          !! set gradient bounds
          lim_g(1,:) = max(1,                 stp_idx)
          lim_g(2,:) = min(this%output_shape, stp_idx + ([i,j,k]-1)/this%stp )

          !! apply full convolution to compute input gradients
          !! https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
          this%di(i,j,k,m) = &
               this%di(i,j,k,m) + &
               sum( &
               grad_dz(&
               lim_g(1,1):lim_g(2,1),&
               lim_g(1,2):lim_g(2,2),&
               lim_g(1,3):lim_g(2,3),l) * &
               this%weight(&
               lim_w(1,1):lim_w(2,1):-this%stp(1),&
               lim_w(1,2):lim_w(2,2):-this%stp(2),&
               lim_w(1,3):lim_w(2,3):-this%stp(3),m,l) )

       end do
    end if

  end subroutine backward_4d
!!!#############################################################################


!!!#############################################################################
!!! update the weights based on how much error the node is responsible for
!!!#############################################################################
  pure subroutine update(this, optimiser, batch_size)
    use optimiser, only: optimiser_type
    implicit none
    class(conv3d_layer_type), intent(inout) :: this
    type(optimiser_type), intent(in) :: optimiser
    integer, optional, intent(in) :: batch_size

    
    !! normalise by number of samples
    if(present(batch_size))then
       this%dw = this%dw/batch_size
       this%db = this%db/batch_size
    end if

    !! apply gradient clipping
    call optimiser%clip(size(this%dw),this%dw,this%db)

    !! update the convolution layer weights using gradient descent
    call optimiser%optimise(&
         this%weight,&
         this%weight_incr, &
         this%dw)
    !! update the convolution layer bias using gradient descent
    call optimiser%optimise(&
         this%bias,&
         this%bias_incr, &
         this%db)

    !! reset gradients
    this%di = 0._real12
    this%dw = 0._real12
    this%db = 0._real12

  end subroutine update
!!!#############################################################################


end module conv3d_layer
!!!#############################################################################
