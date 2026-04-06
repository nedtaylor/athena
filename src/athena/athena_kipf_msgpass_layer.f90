module athena__kipf_msgpass_layer
  !! Module implementing Kipf & Welling Graph Convolutional Network (GCN)
  !!
  !! This module implements the graph convolutional layer from Kipf & Welling
  !! (2017) with symmetric degree normalisation for semi-supervised learning.
  !!
  !! Mathematical operation:
  !! \[ H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right) \]
  !!
  !! where:
  !! * \( \tilde{A} = A + I \) (adjacency matrix with added self-loops)
  !! * \( \tilde{D} \) is the degree matrix of \( \tilde{A} \)
  !! * \( H^{(l)} \) is the node feature matrix at layer l
  !! * \( W^{(l)} \) is a learnable weight matrix
  !! * \( \sigma \) is the activation function
  !!
  !! The normalisation \( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \) ensures
  !! proper scaling by degree.
  !! Preserves graph structure, producing node-level (not graph-level) outputs.
  !!
  !! Reference: Kipf & Welling (2017), ICLR
  use coreutils, only: real32, stop_program
  use graphstruc, only: graph_type
  use athena__misc_types, only: base_actv_type, base_init_type, &
       onnx_attribute_type, onnx_node_type, onnx_initialiser_type, &
       onnx_tensor_type
  use diffstruc, only: array_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  use athena__diffstruc_extd, only: kipf_propagate, kipf_update
  use diffstruc, only: matmul
  implicit none


  private

  public :: kipf_msgpass_layer_type
  public :: read_kipf_msgpass_layer


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: kipf_msgpass_layer_type

     ! this is for chen 2021 et al
     !  type(array2d_type), dimension(:), allocatable :: edge_weight
     !  !! Weights for the edges
     !  type(array2d_type), dimension(:), allocatable :: vertex_weight
     !  !! Weights for the vertices

   contains
     procedure, pass(this) :: get_num_params => get_num_params_kipf
     !! Get the number of parameters for the message passing layer
     procedure, pass(this) :: set_hyperparams => set_hyperparams_kipf
     !! Set the hyperparameters for the message passing layer
     procedure, pass(this) :: init => init_kipf
     !! Initialise the message passing layer
     procedure, pass(this) :: print_to_unit => print_to_unit_kipf
     !! Print the message passing layer
     procedure, pass(this) :: read => read_kipf
     !! Read the message passing layer

     procedure, pass(this) :: update_message => update_message_kipf
     !! Update the message

     procedure, pass(this) :: update_readout => update_readout_kipf
     !! Update the readout

     procedure, pass(this) :: get_attributes => get_attributes_kipf
     !! Get the attributes of the layer (for ONNX export)
     procedure, pass(this) :: emit_onnx_nodes => emit_onnx_nodes_kipf
     !! Emit ONNX JSON nodes for Kipf GCN layer
  end type kipf_msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  interface kipf_msgpass_layer_type
     !! Interface for setting up the MPNN layer
     module function layer_setup( &
          num_vertex_features, num_time_steps, &
          activation, &
          kernel_initialiser, &
          verbose &
     ) result(layer)
       !! Set up the message passing layer
       integer, dimension(:), intent(in) :: num_vertex_features
       !! Number of features
       integer, intent(in) :: num_time_steps
       !! Number of time steps
       class(*), optional, intent(in) :: activation, kernel_initialiser
       !! Activation function and kernel initialiser
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(kipf_msgpass_layer_type) :: layer
       !! Instance of the message passing layer
     end function layer_setup
  end interface kipf_msgpass_layer_type

contains


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure function get_num_params_kipf(this) result(num_params)
    !! Get the number of parameters for the message passing layer
    !!
    !! This function calculates the number of parameters for the message passing
    !! layer.
    !! This procedure is based on code from the neural-fortran library
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer :: num_params
    !! Number of parameters

    ! Local variables
    integer :: t
    !! Loop index

    num_params = 0
    do t = 1, this%num_time_steps
       num_params = num_params + &
            this%num_vertex_features(t-1) * this%num_vertex_features(t)
    end do

  end function get_num_params_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function layer_setup( &
       num_vertex_features, num_time_steps, &
       activation, &
       kernel_initialiser, &
       verbose &
  ) result(layer)
    !! Set up the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    class(*), optional, intent(in) :: activation, kernel_initialiser
    !! Activation function and kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    type(kipf_msgpass_layer_type) :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level
    class(base_actv_type), allocatable :: activation_
    !! Activation function object
    class(base_init_type), allocatable :: kernel_initialiser_
    !! Kernel initialisers

    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Set activation functions based on input name
    !---------------------------------------------------------------------------
    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if


    !---------------------------------------------------------------------------
    ! Define weights (kernels) and biases initialisers
    !---------------------------------------------------------------------------
    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if


    !---------------------------------------------------------------------------
    ! Set hyperparameters
    !---------------------------------------------------------------------------
    call layer%set_hyperparams( &
         num_vertex_features = num_vertex_features, &
         num_time_steps = num_time_steps, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         verbose = verbose_ &
    )


    !---------------------------------------------------------------------------
    ! Initialise layer shape
    !---------------------------------------------------------------------------
    call layer%init(input_shape=[layer%num_vertex_features(0), 0])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_kipf( &
       this, &
       num_vertex_features, &
       num_time_steps, &
       activation, &
       kernel_initialiser, &
       verbose &
  )
    !! Set the hyperparameters for the message passing layer
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, dimension(:), intent(in) :: num_vertex_features
    !! Number of vertex features
    integer, intent(in) :: num_time_steps
    !! Number of time steps
    class(base_actv_type), allocatable, intent(in) :: activation
    !! Activation function
    class(base_init_type), allocatable, intent(in) :: kernel_initialiser
    !! Kernel initialiser
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    character(len=256) :: buffer


    this%name = 'kipf'
    this%type = 'msgp'
    this%input_rank = 2
    this%output_rank = 2
    this%use_graph_output = .true.
    this%num_time_steps = num_time_steps
    if(allocated(this%num_vertex_features)) &
         deallocate(this%num_vertex_features)
    if(allocated(this%num_edge_features)) &
         deallocate(this%num_edge_features)
    if(size(num_vertex_features, 1) .eq. 1) then
       allocate( &
            this%num_vertex_features(0:num_time_steps), &
            source = num_vertex_features(1) &
       )
    elseif(size(num_vertex_features, 1) .eq. num_time_steps + 1) then
       allocate( &
            this%num_vertex_features(0:this%num_time_steps), &
            source = num_vertex_features &
       )
    else
       call stop_program( &
            "Error: num_vertex_features must be a scalar or a vector of length &
            &num_time_steps + 1" &
       )
    end if
    allocate( this%num_edge_features(0:this%num_time_steps), source = 0 )
    this%use_graph_input = .true.
    if(allocated(this%activation)) deallocate(this%activation)
    if(.not.allocated(activation))then
       this%activation = activation_setup("none")
    else
       allocate(this%activation, source=activation)
    end if
    if(allocated(this%kernel_init)) deallocate(this%kernel_init)
    if(.not.allocated(kernel_initialiser))then
       buffer = get_default_initialiser(this%activation%name)
       this%kernel_init = initialiser_setup(buffer)
    else
       allocate(this%kernel_init, source=kernel_initialiser)
    end if
    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("KIPF activation function: ",A)') &
               trim(this%activation%name)
          write(*,'("KIPF kernel initialiser: ",A)') &
               trim(this%kernel_init%name)
       end if
    end if
    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1:this%num_time_steps))
    do t = 1, this%num_time_steps
       this%num_params_msg(t) = &
            this%num_vertex_features(t-1) * this%num_vertex_features(t)
    end do
    if(allocated(this%input_shape)) deallocate(this%input_shape)
    if(allocated(this%output_shape)) deallocate(this%output_shape)

  end subroutine set_hyperparams_kipf
!###############################################################################


!###############################################################################
  subroutine init_kipf(this, input_shape, verbose)
    !! Initialise the message passing layer
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the fully connected layer
    integer, dimension(:), intent(in) :: input_shape
    !! Input shape
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: t
    !! Loop index
    integer :: verbose_ = 0
    !! Verbosity level


    !---------------------------------------------------------------------------
    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    !---------------------------------------------------------------------------
    ! Initialise number of inputs
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)
    this%output_shape = [this%num_vertex_features(this%num_time_steps), 0]
    this%num_params = this%get_num_params()
    if(allocated(this%weight_shape)) deallocate(this%weight_shape)
    if(allocated(this%bias_shape)) deallocate(this%bias_shape)
    allocate(this%weight_shape(2,this%num_time_steps))
    do t = 1, this%num_time_steps
       this%weight_shape(:,t) = &
            [ this%num_vertex_features(t), this%num_vertex_features(t-1) ]
    end do


    !---------------------------------------------------------------------------
    ! Allocate weight, weight steps (velocities), output, and activation
    !---------------------------------------------------------------------------
    if(allocated(this%params)) deallocate(this%params)
    allocate(this%params(this%num_time_steps))
    do t = 1, this%num_time_steps
       call this%params(t)%allocate( &
            array_shape = [ this%weight_shape(:,t), 1 ] &
       )
       call this%params(t)%set_requires_grad(.true.)
       this%params(t)%is_sample_dependent = .false.
       this%params(t)%is_temporary = .false.
       this%params(t)%fix_pointer = .true.
    end do


    !---------------------------------------------------------------------------
    ! Initialise weights (kernels)
    !---------------------------------------------------------------------------
    do t = 1, this%num_time_steps
       call this%kernel_init%initialise( &
            this%params(t)%val(:,1), &
            fan_in = this%num_vertex_features(t-1), &
            fan_out = this%num_vertex_features(t), &
            spacing = [ this%num_vertex_features(t) ] &
       )
    end do


    !---------------------------------------------------------------------------
    ! Allocate arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)

  end subroutine init_kipf
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  subroutine print_to_unit_kipf(this, unit)
    !! Print kipf message passing layer to unit
    use coreutils, only: to_upper
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(in) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! File unit

    ! Local variables
    integer :: t
    !! Loop index
    character(100) :: fmt
    !! Format string


    ! Write initial parameters
    !---------------------------------------------------------------------------
    write(unit,'(3X,"NUM_TIME_STEPS = ",I0)') this%num_time_steps
    write(fmt,'("(3X,""NUM_VERTEX_FEATURES ="",",I0,"(1X,I0))")') &
         this%num_time_steps + 1
    write(unit,fmt) this%num_vertex_features

    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if


    ! Write learned parameters
    !---------------------------------------------------------------------------
    write(unit,'("WEIGHTS")')
    do t = 1, this%num_time_steps, 1
       write(unit,'(5(E16.8E2))') this%params(t)%val
    end do
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_kipf
!###############################################################################


!###############################################################################
  subroutine read_kipf(this, unit, verbose)
    !! Read the message passing layer
    use athena__tools_infile, only: assign_val, assign_vec, get_val, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Unit to read from
    integer, optional, intent(in) :: verbose
    !! Verbosity level

    ! Local variables
    integer :: stat
    !! Status of read
    integer :: verbose_ = 0
    !! Verbosity level
    integer :: t, j, k, c, itmp1, iline
    !! Loop variables and temporary integer
    integer :: num_time_steps = 0
    !! Number of time steps
    character(14) :: kernel_initialiser_name=''
    !! Initialisers
    character(20) :: activation_name=''
    !! Activation function name
    class(base_actv_type), allocatable :: activation
    !! Activation function
    class(base_init_type), allocatable :: kernel_initialiser
    !! Initialisers
    integer, dimension(:), allocatable :: num_vertex_features
    !! Number of vertex and edge features
    character(256) :: buffer, tag, err_msg
    !! Buffer, tag, and error message
    real(real32), allocatable, dimension(:) :: data_list
    !! Data list
    integer :: param_line, final_line
    !! Parameter line number


    ! Initialise optional arguments
    !---------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose


    ! Loop over tags in layer card
    !---------------------------------------------------------------------------
    iline = 0
    param_line = 0
    final_line = 0
    tag_loop: do

       ! Check for end of file
       !------------------------------------------------------------------------
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       ! Check for end of layer card
       !------------------------------------------------------------------------
       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       ! Read parameters from file
       !------------------------------------------------------------------------
       select case(trim(tag))
       case("NUM_TIME_STEPS")
          call assign_val(buffer, num_time_steps, itmp1)
       case("NUM_VERTEX_FEATURES")
          itmp1 = icount(get_val(buffer))
          allocate(num_vertex_features(itmp1), source=0)
          call assign_vec(buffer, num_vertex_features, itmp1)
       case("ACTIVATION")
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALisER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          param_line = iline
       case default
          ! Don't look for "e" due to scientific notation of numbers
          ! ... i.e. exponent (E+00)
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
    kernel_initialiser = initialiser_setup(kernel_initialiser_name)


    ! Set hyperparameters and initialise layer
    !---------------------------------------------------------------------------
    if(num_time_steps.gt.0 .and. num_time_steps.ne.size(num_vertex_features,1)-1)then
       write(err_msg,'("NUM_TIME_STEPS = ",I0," does not match length of "// &
            &"NUM_VERTEX_FEATURES = ",I0)') num_time_steps, &
            size(num_vertex_features,1)-1
       call stop_program(err_msg)
       return
    end if
    call this%set_hyperparams( &
         num_time_steps = num_time_steps, &
         num_vertex_features = num_vertex_features, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[this%num_vertex_features(0), 0])


    ! Check if WEIGHTS card was found
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in "//to_upper(trim(this%name))//" not found"
    else
       call move(unit, param_line - iline, iostat=stat)
       do t = 1, this%num_time_steps
          allocate(data_list(this%num_params_msg(t)), source=0._real32)
          c = 1
          k = 1
          data_concat_loop: do while(c.le.this%num_params_msg(t))
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit data_concat_loop
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
             c = c + k
          end do data_concat_loop
          this%params(t)%val(:,1) = data_list(1:this%num_params_msg(t))
          deallocate(data_list)
       end do

       ! Check for end of weights card
       !------------------------------------------------------------------------
       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          write(0,*) trim(adjustl(buffer))
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if


    !---------------------------------------------------------------------------
    ! Check for end of layer card
    !---------------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(0,*) trim(adjustl(buffer))
       write(err_msg,'("END ",A," not where expected")') to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_kipf
!###############################################################################


!###############################################################################
  function read_kipf_msgpass_layer(unit, verbose) result(layer)
    !! Read kipf message passing layer from file and return layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the message passing layer

    ! Local variables
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source = kipf_msgpass_layer_type( &
         num_time_steps = 1, &
         num_vertex_features = [ 0, 0 ] &
    ))
    call layer%read(unit, verbose=verbose_)

  end function read_kipf_msgpass_layer
!###############################################################################


!###############################################################################
  function get_attributes_kipf(this) result(attributes)
    !! Get the attributes of the Kipf GCN layer (for ONNX export)
    implicit none
    class(kipf_msgpass_layer_type), intent(in) :: this
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes

    integer :: t
    character(256) :: buf

    allocate(attributes(3))

    write(buf, '(I0)') this%num_time_steps
    attributes(1) = onnx_attribute_type( &
         name='num_time_steps', type='int', val=trim(buf))

    buf = ''
    do t = 0, this%num_time_steps
       if(t .eq. 0)then
          write(buf, '(I0)') this%num_vertex_features(t)
       else
          write(buf, '(A," ",I0)') trim(buf), this%num_vertex_features(t)
       end if
    end do
    attributes(2) = onnx_attribute_type( &
         name='num_vertex_features', type='ints', val=trim(buf))

    attributes(3) = onnx_attribute_type( &
         name='message_activation', type='string', &
         val=trim(this%activation%name))

  end function get_attributes_kipf
!###############################################################################


!###############################################################################
  subroutine emit_onnx_nodes_kipf( &
       this, prefix, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits &
  )
    !! Emit ONNX JSON nodes for Kipf GCN layer
    !!
    !! Decomposes the Kipf message passing layer into standard ONNX ops:
    !!   Gather, ScatterElements, Mul, Pow, MatMul, activation
    !!
    !! Kipf GCN: H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l))
    !! Decomposed per timestep:
    !!   1. Extract source/target indices from edge_index
    !!   2. Gather source vertex features
    !!   3. Compute normalisation coeff = (deg_src * deg_tgt)^(-0.5)
    !!   4. Scale source features by coefficient
    !!   5. Scatter-add to target vertices
    !!   6. MatMul with weight W (transposed)
    !!   7. Apply activation
    use athena__onnx_utils, only: emit_node, emit_squeeze_node, &
         emit_constant_int64, emit_constant_float, &
         emit_constant_of_shape_float, emit_activation_node, &
         col_to_row_major_2d
    implicit none

    class(kipf_msgpass_layer_type), intent(in) :: this
    character(*), intent(in) :: prefix
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    integer, intent(in) :: max_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    integer, intent(in) :: max_inits

    integer :: t
    character(128) :: cur_vertex_name
    character(:), allocatable :: suffix

    do t = 1, this%num_time_steps
       call emit_kipf_timestep_impl( &
            prefix, t, &
            this%num_vertex_features(t-1), &
            this%num_vertex_features(t), &
            this%params(t)%val(:,1), &
            this%activation%name, &
            nodes, num_nodes, max_nodes, &
            inits, num_inits, max_inits, &
            cur_vertex_name &
       )
    end do

    ! Kipf produces node-level output (no readout)
    ! Rename final timestep output to match expected naming convention
    suffix = '_output'
    if(this%activation%name .ne. "none")then
       suffix = '_' // trim(adjustl(this%activation%name)) // '_output'
    end if
    num_nodes = num_nodes + 1
    nodes(num_nodes)%name = trim(prefix) // '_identity'
    nodes(num_nodes)%op_type = 'Identity'
    allocate(nodes(num_nodes)%inputs(1))
    nodes(num_nodes)%inputs(1) = trim(cur_vertex_name)
    allocate(nodes(num_nodes)%outputs(1))
    nodes(num_nodes)%outputs(1) = trim(prefix) // trim(suffix)
    nodes(num_nodes)%attributes_json = ''

  end subroutine emit_onnx_nodes_kipf
!###############################################################################


!###############################################################################
  subroutine emit_kipf_timestep_impl( &
       prefix, t, &
       nv_in, nv_out, &
       weight_data, activation_name, &
       nodes, num_nodes, max_nodes, &
       inits, num_inits, max_inits, &
       vertex_out &
  )
    !! Emit ONNX nodes for one Kipf GCN time step
    !!
    !! Steps:
    !!   1. Extract source (row 0) and target (row 2) from edge_index
    !!   2. Gather source features → Scale by (deg_src*deg_tgt)^(-0.5)
    !!   3. Scatter-add normalised messages to target vertices
    !!   4. MatMul with weight W^T
    !!   5. Apply activation
    use athena__onnx_utils, only: emit_node, emit_squeeze_node, &
         emit_constant_int64, emit_constant_float, &
         emit_constant_of_shape_float, emit_activation_node, &
         col_to_row_major_2d
    implicit none
    character(*), intent(in) :: prefix
    integer, intent(in) :: t
    integer, intent(in) :: nv_in, nv_out
    real(real32), intent(in) :: weight_data(:)
    character(*), intent(in) :: activation_name
    type(onnx_node_type), intent(inout), dimension(:) :: nodes
    integer, intent(inout) :: num_nodes
    integer, intent(in) :: max_nodes
    type(onnx_initialiser_type), intent(inout), dimension(:) :: inits
    integer, intent(inout) :: num_inits
    integer, intent(in) :: max_inits
    character(128), intent(out) :: vertex_out

    character(128) :: tp
    character(128) :: tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7
    character(128) :: vertex_in, edge_index_in, degree_in

    write(tp, '(A,"_t",I0)') trim(prefix), t

    ! Input tensor names follow convention set during write_onnx
    write(vertex_in, '(A,"_vertex_in")') trim(prefix)
    write(edge_index_in, '(A,"_edge_index_in")') trim(prefix)
    write(degree_in, '(A,"_degree_in")') trim(prefix)

    ! If timestep > 1, vertex input is previous timestep output
    if(t .gt. 1) then
       if(trim(activation_name) .ne. "none")then
          write(vertex_in, '(A,"_t",I0,"_",A,"_output")') &
               trim(prefix), t-1, trim(adjustl(activation_name))
       else
          write(vertex_in, '(A,"_t",I0,"_mm_out")') trim(prefix), t-1
       end if
    end if

    ! --- Step 1: Extract source and target indices from edge_index ---

    ! Constant: index 0 (row 0 = source vertices)
    write(tmp1, '(A,"_idx0")') trim(tp)
    call emit_constant_int64(trim(tmp1), [0], [1], &
         nodes, num_nodes, inits, num_inits)

    ! Gather row 0 → source indices [num_csr]
    write(tmp2, '(A,"_src_raw")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_src', &
         trim(tmp2), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(edge_index_in), in2=trim(tmp1))

    ! Squeeze → [num_csr]
    write(tmp3, '(A,"_src")') trim(tp)
    call emit_squeeze_node(trim(tp)//'_sq_src', &
         trim(tmp2), trim(tmp1), trim(tmp3), &
         nodes, num_nodes)

    ! Constant: index 2 (row 2 = target vertices)
    write(tmp1, '(A,"_idx2")') trim(tp)
    call emit_constant_int64(trim(tmp1), [2], [1], &
         nodes, num_nodes, inits, num_inits)

    ! Gather row 2 → target indices [num_csr]
    write(tmp4, '(A,"_tgt_raw")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_tgt', &
         trim(tmp4), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(edge_index_in), in2=trim(tmp1))

    write(tmp5, '(A,"_tgt")') trim(tp)
    write(tmp6, '(A,"_idx0b")') trim(tp)
    call emit_constant_int64(trim(tmp6), [0], [1], &
         nodes, num_nodes, inits, num_inits)
    call emit_squeeze_node(trim(tp)//'_sq_tgt', &
         trim(tmp4), trim(tmp6), trim(tmp5), &
         nodes, num_nodes)

    ! --- Step 2: Gather source features and compute normalisation ---

    ! Gather source vertex features: [num_csr, nv_in]
    write(tmp1, '(A,"_src_feat")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_vfeat', &
         trim(tmp1), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(vertex_in), in2=trim(tmp3))

    ! Cast degree to float
    write(tmp2, '(A,"_deg_f")') trim(tp)
    call emit_node('Cast', trim(tp)//'_cast_deg', &
         trim(tmp2), &
         '        "attribute": [{"name": "to", "i": "1", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(degree_in))

    ! Gather source degrees: deg[src] → [num_csr]
    write(tmp4, '(A,"_deg_src")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_deg_src', &
         trim(tmp4), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(tmp2), in2=trim(tmp3))

    ! Gather target degrees: deg[tgt] → [num_csr]
    write(tmp6, '(A,"_deg_tgt")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_deg_tgt', &
         trim(tmp6), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(tmp2), in2=trim(tmp5))

    ! coeff = (deg_src * deg_tgt)^(-0.5)
    write(tmp7, '(A,"_deg_prod")') trim(tp)
    call emit_node('Mul', trim(tp)//'_mul_deg', &
         trim(tmp7), '', nodes, num_nodes, &
         in1=trim(tmp4), in2=trim(tmp6))

    write(tmp2, '(A,"_neg_half")') trim(tp)
    call emit_constant_float(trim(tmp2), [-0.5_real32], [1], &
         nodes, num_nodes, inits, num_inits)

    write(tmp3, '(A,"_coeff")') trim(tp)
    call emit_node('Pow', trim(tp)//'_pow_coeff', &
         trim(tmp3), '', nodes, num_nodes, &
         in1=trim(tmp7), in2=trim(tmp2))

    ! Unsqueeze coeff for broadcasting: [num_csr] → [num_csr, 1]
    write(tmp4, '(A,"_coeff_us")') trim(tp)
    write(tmp6, '(A,"_us_ax1")') trim(tp)
    call emit_constant_int64(trim(tmp6), [1], [1], &
         nodes, num_nodes, inits, num_inits)
    call emit_node('Unsqueeze', trim(tp)//'_us_coeff', &
         trim(tmp4), '', nodes, num_nodes, &
         in1=trim(tmp3), in2=trim(tmp6))

    ! Scale source features by coefficient: [num_csr, nv_in]
    write(tmp2, '(A,"_scaled_feat")') trim(tp)
    call emit_node('Mul', trim(tp)//'_mul_coeff', &
         trim(tmp2), '', nodes, num_nodes, &
         in1=trim(tmp1), in2=trim(tmp4))

    ! --- Step 3: Scatter-add normalised messages to target vertices ---

    ! Get num_nodes from shape of vertex_in
    write(tmp1, '(A,"_vshape")') trim(tp)
    call emit_node('Shape', trim(tp)//'_shape_v', &
         trim(tmp1), '', nodes, num_nodes, &
         in1=trim(vertex_in))

    write(tmp3, '(A,"_nnodes_idx")') trim(tp)
    call emit_constant_int64(trim(tmp3), [0], [1], &
         nodes, num_nodes, inits, num_inits)

    write(tmp4, '(A,"_nnodes")') trim(tp)
    call emit_node('Gather', trim(tp)//'_gather_nn', &
         trim(tmp4), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(tmp1), in2=trim(tmp3))

    ! Constant: [nv_in] as int64
    write(tmp6, '(A,"_feat_dim")') trim(tp)
    call emit_constant_int64(trim(tmp6), [nv_in], [1], &
         nodes, num_nodes, inits, num_inits)

    ! Concat [num_nodes, nv_in] → shape for zeros
    write(tmp7, '(A,"_aggr_shape")') trim(tp)
    call emit_node('Concat', trim(tp)//'_cat_shape', &
         trim(tmp7), &
         '        "attribute": [{"name": "axis", "i": "0", "type": "INT"}]', &
         nodes, num_nodes, &
         in1=trim(tmp4), in2=trim(tmp6))

    ! ConstantOfShape → zeros [num_nodes, nv_in]
    write(tmp1, '(A,"_zeros")') trim(tp)
    call emit_constant_of_shape_float(trim(tp)//'_zeros', &
         trim(tmp7), 0.0_real32, trim(tmp1), &
         nodes, num_nodes, inits, num_inits)

    ! Unsqueeze target [num_csr] → [num_csr, 1]
    write(tmp3, '(A,"_tgt_us")') trim(tp)
    write(tmp6, '(A,"_us_ax1b")') trim(tp)
    call emit_constant_int64(trim(tmp6), [1], [1], &
         nodes, num_nodes, inits, num_inits)
    call emit_node('Unsqueeze', trim(tp)//'_us_tgt', &
         trim(tmp3), '', nodes, num_nodes, &
         in1=trim(tp)//'_tgt', in2=trim(tmp6))

    ! Get message shape for Expand
    write(tmp4, '(A,"_msg_shape")') trim(tp)
    call emit_node('Shape', trim(tp)//'_shape_msg', &
         trim(tmp4), '', nodes, num_nodes, &
         in1=trim(tmp2))

    write(tmp6, '(A,"_tgt_exp")') trim(tp)
    call emit_node('Expand', trim(tp)//'_expand_tgt', &
         trim(tmp6), '', nodes, num_nodes, &
         in1=trim(tmp3), in2=trim(tmp4))

    ! ScatterElements(zeros, expanded_target, scaled_features, axis=0, reduction=add)
    write(tmp7, '(A,"_aggr")') trim(tp)
    call emit_node('ScatterElements', trim(tp)//'_scatter_add', &
         trim(tmp7), &
         '        "attribute": [' // &
         '{"name": "axis", "i": "0", "type": "INT"}, ' // &
         '{"name": "reduction", "s": "YWRk", "type": "STRING"}]', &
         nodes, num_nodes, &
         in1=trim(tmp1), in2=trim(tmp6), in3=trim(tmp2))

    ! --- Step 4: MatMul with weight W ---
    ! Store W as initialiser: [nv_out, nv_in]
    write(tmp1, '(A,"_W")') trim(tp)
    num_inits = num_inits + 1
    inits(num_inits)%name = trim(tmp1)
    inits(num_inits)%data_type = 1
    allocate(inits(num_inits)%dims(2))
    inits(num_inits)%dims = [nv_out, nv_in]
    allocate(inits(num_inits)%data(size(weight_data)))
    call col_to_row_major_2d( &
         weight_data, inits(num_inits)%data, nv_out, nv_in)

    ! Transpose W: [nv_out, nv_in] → [nv_in, nv_out]
    write(tmp2, '(A,"_Wt")') trim(tp)
    call emit_node('Transpose', trim(tp)//'_transpose_W', &
         trim(tmp2), &
         '        "attribute": [{"name": "perm", "ints": ["1", "0"], ' // &
         '"type": "INTS"}]', &
         nodes, num_nodes, in1=trim(tmp1))

    ! MatMul: aggr @ W^T = [num_nodes, nv_in] @ [nv_in, nv_out] → [num_nodes, nv_out]
    write(tmp3, '(A,"_mm_out")') trim(tp)
    call emit_node('MatMul', trim(tp)//'_matmul', &
         trim(tmp3), '', nodes, num_nodes, &
         in1=trim(tmp7), in2=trim(tmp2))

    ! --- Step 5: Activation ---
    if(trim(activation_name) .ne. "none")then
       call emit_activation_node( &
            activation_name, &
            trim(tp), trim(tmp3), &
            nodes, num_nodes, max_nodes)
       write(vertex_out, '(A)') &
            trim(nodes(num_nodes)%outputs(1))
    else
       vertex_out = trim(tmp3)
    end if

  end subroutine emit_kipf_timestep_impl
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!##############################################################################!
  subroutine update_message_kipf(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in), target :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, t
    !! Batch index, time step
    type(array_type), pointer :: ptr1, ptr2, ptr3
    !! Pointers to arrays

    if(allocated(this%output))then
       if(size(this%output,2).ne.size(input,2))then
          deallocate(this%output)
          allocate(this%output(1,size(input,2)))
       end if
    else
       allocate(this%output(1,size(input,2)))
    end if

    do s = 1, size(input,2)
       ptr1 => input(1,s)
       do t = 1, this%num_time_steps
          ptr2 => kipf_propagate( &
               ptr1, &
               this%graph(s)%adj_ia, this%graph(s)%adj_ja &
          )

          ! this%z(t,s) = kipf_update( &
          !      this%message(t,s), this%params(t), this%graph(s)%adj_ia &
          ! )
          ptr3 => matmul( this%params(t), ptr2 )
          ptr1 => this%activation%apply( ptr3 )
       end do
       call this%output(1,s)%zero_grad()
       call this%output(1,s)%assign_and_deallocate_source(ptr1)
       this%output(1,s)%is_temporary = .false.
    end do

  end subroutine update_message_kipf
!###############################################################################


!###############################################################################
  subroutine update_readout_kipf(this)
    !! Update the readout (empty for node-level output)
    implicit none
    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout), target :: this
    !! Instance of the message passing layer
  end subroutine update_readout_kipf
!###############################################################################

end module athena__kipf_msgpass_layer
