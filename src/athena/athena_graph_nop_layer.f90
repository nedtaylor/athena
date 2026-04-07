module athena__graph_nop_layer
  !! Module containing implementation of a Graph Neural Operator (GNO) layer
  !!
  !! This module implements a Graph Neural Operator layer that learns a
  !! continuous kernel on graph edges.  It combines a learnable kernel
  !! network (small MLP) evaluated on relative coordinates with a linear
  !! transform of the node features:
  !!
  !! \[ h_i^{(l+1)} = \sigma\!\left(
  !!    \mathbf{W}\,h_i^{(l)}
  !!  + \sum_{j \in \mathcal{N}(i)}
  !!      \kappa_\theta(x_i - x_j)\,h_j^{(l)}
  !! \right) \]
  !!
  !! where:
  !!   - \(h_i^{(l)} \in \mathbb{R}^{F_{in}}\) is the node feature at layer l
  !!   - \(x_i \in \mathbb{R}^{d}\) is the node coordinate / attribute
  !!   - \(\kappa_\theta \colon \mathbb{R}^d \to \mathbb{R}^{F_{out} \times F_{in}}\)
  !!     is a learnable kernel MLP
  !!   - \(\mathbf{W} \in \mathbb{R}^{F_{out} \times F_{in}}\) is a learnable
  !!     linear (bypass) transform
  !!   - \(\sigma\) is the activation function
  !!
  !! The kernel MLP has one hidden layer:
  !!   \(\kappa_\theta(\Delta x) = V\,\text{relu}(U\,\Delta x + b_u) + b_v\)
  !! where \(U \in \mathbb{R}^{H \times d}\),
  !!       \(V \in \mathbb{R}^{(F_{out} F_{in}) \times H}\),
  !!       and \(H\) is the hidden width of the kernel network.
  !!
  !! Input layout:
  !!   input(1,s) = node features                    [F_in x num_vertices]
  !!   input(2,s) = edge geometry / relative coords [d x num_edges]
  !!
  !! Number of learnable parameters:
  !!   Kernel MLP:  \(H d + H + (F_{out} F_{in}) H + F_{out} F_{in}\)
  !!   Linear:      \(F_{out} F_{in}\)
  !!   Bias:        \(F_{out}\)  (optional)
  !!
  !! This layer extends the message passing layer type and uses the diffstruc
  !! autodiff framework to support physics-informed neural networks (PINNs).
  !! The forward pass builds a computation graph through two differentiable
  !! operations: `gno_kernel_eval` (kernel MLP evaluation on every edge) and
  !! `gno_aggregate` (neighbour message aggregation), followed by the standard
  !! `matmul`, `add_bias`, and activation operations.
  use coreutils, only: real32, stop_program
  use graphstruc, only: graph_type
  use athena__base_layer, only: base_layer_type
  use athena__msgpass_layer, only: msgpass_layer_type
  use athena__misc_types, only: base_actv_type, base_init_type
  use diffstruc, only: array_type, matmul, operator(+)
  use athena__diffstruc_extd, only: add_bias, gno_kernel_eval, gno_aggregate
  implicit none


  private

  public :: graph_nop_layer_type
  public :: read_graph_nop_layer


  type, extends(msgpass_layer_type) :: graph_nop_layer_type
     !! Type for a Graph Neural Operator layer
     integer :: coord_dim = 0
     !! Dimensionality of edge geometric features (d)
     integer :: kernel_hidden = 0
     !! Hidden width of the kernel MLP (H)
   contains
     procedure, pass(this) :: get_num_params => get_num_params_gno
     procedure, pass(this) :: set_hyperparams => set_hyperparams_gno
     procedure, pass(this) :: init => init_gno
     procedure, pass(this) :: print_to_unit => print_to_unit_gno
     procedure, pass(this) :: read => read_gno

     procedure, pass(this) :: update_message => update_message_gno
     procedure, pass(this) :: update_readout => update_readout_gno
  end type graph_nop_layer_type

  interface graph_nop_layer_type
     module function layer_setup( &
          num_outputs, coord_dim, kernel_hidden, &
          num_inputs, use_bias, &
          activation, &
          kernel_initialiser, bias_initialiser, &
          verbose &
     ) result(layer)
       integer, intent(in) :: num_outputs
       !! Number of output node features
       integer, intent(in) :: coord_dim
       !! Dimensionality of edge geometric features
       integer, optional, intent(in) :: kernel_hidden
       !! Hidden width of kernel MLP (default: num_outputs)
       integer, optional, intent(in) :: num_inputs
       !! Number of input node features (deferred if absent)
       logical, optional, intent(in) :: use_bias
       !! Whether to use bias (default: .true.)
       class(*), optional, intent(in) :: activation
       !! Activation function
       class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
       !! Parameter initialisers
       integer, optional, intent(in) :: verbose
       !! Verbosity level
       type(graph_nop_layer_type) :: layer
     end function layer_setup
  end interface graph_nop_layer_type


contains


!###############################################################################
  pure function get_num_params_gno(this) result(num_params)
    !! Get the number of learnable parameters
    !!
    !! Parameters:
    !!   params(1): packed kernel MLP [H*d + H + F*H + F, 1]
    !!              where F = F_out * F_in
    !!              Layout: U [H*d] | b_u [H] | V [F*H] | b_v [F]
    !!   params(2): W   - linear bypass weights  [F_out * F_in, 1]
    !!   params(3): b   - output bias            [F_out, 1]  (optional)
    implicit none
    class(graph_nop_layer_type), intent(in) :: this
    integer :: num_params

    integer :: F_in, F_out, d, H, F

    F_in  = this%num_vertex_features(0)
    F_out = this%num_vertex_features(1)
    d     = this%coord_dim
    H     = this%kernel_hidden
    F     = F_out * F_in

    num_params = &
         H * d + H + F * H + F + &   ! kernel MLP (U, b_u, V, b_v)
         F_out * F_in                  ! W (linear bypass)
    if(this%use_bias) num_params = num_params + F_out  ! b

  end function get_num_params_gno
!###############################################################################


!###############################################################################
  module function layer_setup( &
       num_outputs, coord_dim, kernel_hidden, &
       num_inputs, use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  ) result(layer)
    use athena__activation, only: activation_setup
    use athena__initialiser, only: initialiser_setup
    implicit none

    integer, intent(in) :: num_outputs
    integer, intent(in) :: coord_dim
    integer, optional, intent(in) :: kernel_hidden
    integer, optional, intent(in) :: num_inputs
    logical, optional, intent(in) :: use_bias
    class(*), optional, intent(in) :: activation
    class(*), optional, intent(in) :: kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    type(graph_nop_layer_type) :: layer

    integer :: verbose_ = 0
    logical :: use_bias_ = .true.
    integer :: kernel_hidden_
    class(base_actv_type), allocatable :: activation_
    class(base_init_type), allocatable :: kernel_initialiser_, bias_initialiser_

    if(present(verbose)) verbose_ = verbose
    if(present(use_bias)) use_bias_ = use_bias
    kernel_hidden_ = num_outputs
    if(present(kernel_hidden)) kernel_hidden_ = kernel_hidden

    if(present(activation))then
       activation_ = activation_setup(activation)
    else
       activation_ = activation_setup("none")
    end if

    if(present(kernel_initialiser))then
       kernel_initialiser_ = initialiser_setup(kernel_initialiser)
    end if
    if(present(bias_initialiser))then
       bias_initialiser_ = initialiser_setup(bias_initialiser)
    end if

    call layer%set_hyperparams( &
         num_outputs = num_outputs, &
         coord_dim = coord_dim, &
         kernel_hidden = kernel_hidden_, &
         use_bias = use_bias_, &
         activation = activation_, &
         kernel_initialiser = kernel_initialiser_, &
         bias_initialiser = bias_initialiser_, &
         verbose = verbose_ &
    )

    if(present(num_inputs)) call layer%init(input_shape=[num_inputs, 0])

  end function layer_setup
!###############################################################################


!###############################################################################
  subroutine set_hyperparams_gno( &
       this, num_outputs, coord_dim, kernel_hidden, &
       use_bias, &
       activation, &
       kernel_initialiser, bias_initialiser, &
       verbose &
  )
    use athena__activation, only: activation_setup
    use athena__initialiser, only: get_default_initialiser, initialiser_setup
    implicit none

    class(graph_nop_layer_type), intent(inout) :: this
    integer, intent(in) :: num_outputs
    integer, intent(in) :: coord_dim
    integer, intent(in) :: kernel_hidden
    logical, intent(in) :: use_bias
    class(base_actv_type), allocatable, intent(in) :: activation
    class(base_init_type), allocatable, intent(in) :: &
         kernel_initialiser, bias_initialiser
    integer, optional, intent(in) :: verbose

    character(len=256) :: buffer

    this%name = "graph_nop"
    this%type = "gnop"
    this%input_rank = 2
    this%output_rank = 2
    this%use_graph_input = .true.
    this%use_graph_output = .true.
    this%use_bias = use_bias
    this%num_outputs = num_outputs
    this%coord_dim = coord_dim
    this%kernel_hidden = kernel_hidden
    this%num_time_steps = 1

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
    if(allocated(this%bias_init)) deallocate(this%bias_init)
    if(.not.allocated(bias_initialiser))then
       buffer = get_default_initialiser( &
            this%activation%name, &
            is_bias=.true. &
       )
       this%bias_init = initialiser_setup(buffer)
    else
       allocate(this%bias_init, source=bias_initialiser)
    end if

    if(present(verbose))then
       if(abs(verbose).gt.0)then
          write(*,'("GRAPH_NOP activation: ",A)') &
               trim(this%activation%name)
       end if
    end if

  end subroutine set_hyperparams_gno
!###############################################################################


!###############################################################################
  subroutine init_gno(this, input_shape, verbose)
    !! Initialise the Graph Neural Operator layer
    !!
    !! input_shape(1) = num_inputs (F_in)
    !! input_shape(2) = num_vertices (set to 0 if variable)
    implicit none

    class(graph_nop_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: verbose

    integer :: num_inputs, H, F_out, F_in, d, F
    integer :: kernel_size, off_U, off_bu, off_V, off_bv
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose

    !---------------------------------------------------------------------------
    ! Set shapes
    !---------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)

    F_in  = input_shape(1)
    F_out = this%num_outputs
    d     = this%coord_dim
    H     = this%kernel_hidden
    F     = F_out * F_in

    !---------------------------------------------------------------------------
    ! Set msgpass fields
    !---------------------------------------------------------------------------
    if(allocated(this%num_vertex_features)) deallocate(this%num_vertex_features)
    allocate(this%num_vertex_features(0:1))
    this%num_vertex_features(0) = F_in
    this%num_vertex_features(1) = F_out

    if(allocated(this%num_edge_features)) deallocate(this%num_edge_features)
    allocate(this%num_edge_features(0:1), source=0)

    kernel_size = H * d + H + F * H + F

    if(allocated(this%num_params_msg)) deallocate(this%num_params_msg)
    allocate(this%num_params_msg(1))
    this%num_params_msg(1) = kernel_size + F_out * F_in
    if(this%use_bias) this%num_params_msg(1) = this%num_params_msg(1) + F_out
    this%num_params_readout = 0

    this%output_shape = [this%num_outputs, 0]
    this%num_params = this%get_num_params()

    !---------------------------------------------------------------------------
    ! Allocate learnable parameters
    !
    ! params(1): packed kernel MLP  [kernel_size, 1]
    !            Layout: U [H*d] | b_u [H] | V [F*H] | b_v [F]
    ! params(2): W   [F_out*F_in, 1]  - linear bypass weights
    ! params(3): b   [F_out, 1]       - output bias (optional)
    !---------------------------------------------------------------------------
    if(allocated(this%weight_shape)) deallocate(this%weight_shape)
    if(allocated(this%params)) deallocate(this%params)
    if(this%use_bias)then
       if(allocated(this%bias_shape)) deallocate(this%bias_shape)
       this%bias_shape = [ F_out ]
       allocate(this%weight_shape(2, 3))
       this%weight_shape(:,3) = [ F_out, 1 ]
       allocate(this%params(3))
    else
       allocate(this%weight_shape(2, 2))
       allocate(this%params(2))
    end if
    this%weight_shape(:,1) = [ kernel_size, 1 ]
    this%weight_shape(:,2) = [ F_out, F_in ]

    ! params(1): packed kernel MLP params
    call this%params(1)%allocate([kernel_size, 1])
    call this%params(1)%set_requires_grad(.true.)
    this%params(1)%fix_pointer = .true.
    this%params(1)%is_sample_dependent = .false.
    this%params(1)%is_temporary = .false.

    ! params(2): W [F_out x F_in]
    call this%params(2)%allocate([F_out, F_in, 1])
    call this%params(2)%set_requires_grad(.true.)
    this%params(2)%fix_pointer = .true.
    this%params(2)%is_sample_dependent = .false.
    this%params(2)%is_temporary = .false.

    if(this%use_bias)then
       ! params(3): b [F_out]
       call this%params(3)%allocate([F_out, 1])
       call this%params(3)%set_requires_grad(.true.)
       this%params(3)%fix_pointer = .true.
       this%params(3)%is_sample_dependent = .false.
       this%params(3)%is_temporary = .false.
    end if


    !---------------------------------------------------------------------------
    ! Initialise learnable parameters
    !---------------------------------------------------------------------------
    off_U  = 0
    off_bu = H * d
    off_V  = off_bu + H
    off_bv = off_V + F * H

    ! U [H x d] — kernel first-layer weights
    call this%kernel_init%initialise( &
         this%params(1)%val(off_U+1:off_bu, 1), &
         fan_in = d, fan_out = H, &
         spacing = [ H ] &
    )
    ! b_u [H] — kernel first-layer bias
    call this%bias_init%initialise( &
         this%params(1)%val(off_bu+1:off_V, 1), &
         fan_in = d, fan_out = H &
    )
    ! V [F x H] — kernel second-layer weights
    call this%kernel_init%initialise( &
         this%params(1)%val(off_V+1:off_bv, 1), &
         fan_in = H, fan_out = F, &
         spacing = [ F ] &
    )
    ! b_v [F] — kernel second-layer bias
    call this%bias_init%initialise( &
         this%params(1)%val(off_bv+1:, 1), &
         fan_in = H, fan_out = F &
    )
    ! W [F_out x F_in] — linear bypass
    num_inputs = F_in
    if(this%use_bias) num_inputs = F_in + 1
    call this%kernel_init%initialise( &
         this%params(2)%val(:,1), &
         fan_in = num_inputs, fan_out = F_out, &
         spacing = [ F_out ] &
    )
    if(this%use_bias)then
       call this%bias_init%initialise( &
            this%params(3)%val(:,1), &
            fan_in = num_inputs, fan_out = F_out &
       )
    end if


    !---------------------------------------------------------------------------
    ! Allocate output arrays
    !---------------------------------------------------------------------------
    if(allocated(this%output)) deallocate(this%output)

  end subroutine init_gno
!###############################################################################


!###############################################################################
  subroutine print_to_unit_gno(this, unit)
    use coreutils, only: to_upper
    implicit none

    class(graph_nop_layer_type), intent(in) :: this
    integer, intent(in) :: unit

    integer :: p

    write(unit,'(3X,"NUM_INPUTS = ",I0)') this%num_vertex_features(0)
    write(unit,'(3X,"NUM_OUTPUTS = ",I0)') this%num_outputs
    write(unit,'(3X,"COORD_DIM = ",I0)') this%coord_dim
    write(unit,'(3X,"KERNEL_HIDDEN = ",I0)') this%kernel_hidden
    write(unit,'(3X,"USE_BIAS = ",L1)') this%use_bias
    if(this%activation%name .ne. 'none')then
       call this%activation%print_to_unit(unit)
    end if

    write(unit,'("WEIGHTS")')
    do p = 1, size(this%params)
       write(unit,'(5(E16.8E2))') this%params(p)%val(:,1)
    end do
    write(unit,'("END WEIGHTS")')

  end subroutine print_to_unit_gno
!###############################################################################


!###############################################################################
  subroutine read_gno(this, unit, verbose)
    use athena__tools_infile, only: assign_val, move
    use coreutils, only: to_lower, to_upper, icount
    use athena__activation, only: read_activation
    use athena__initialiser, only: initialiser_setup
    implicit none

    class(graph_nop_layer_type), intent(inout) :: this
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose

    integer :: stat, verbose_ = 0
    integer :: j, k, c, itmp1, iline
    integer :: num_inputs, num_outputs, coord_dim, kernel_hidden
    logical :: use_bias = .true.
    character(14) :: kernel_initialiser_name='', bias_initialiser_name=''
    class(base_actv_type), allocatable :: activation
    class(base_init_type), allocatable :: kernel_initialiser, bias_initialiser
    character(256) :: buffer, tag, err_msg
    real(real32), allocatable, dimension(:) :: data_list
    integer :: param_line, final_line, num_vals, p

    if(present(verbose)) verbose_ = verbose

    iline = 0
    param_line = 0
    final_line = 0
    tag_loop: do
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg, &
               '("file encountered error (EoF?) before END ",A)') &
               to_upper(this%name)
          call stop_program(err_msg)
          return
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop

       if(trim(adjustl(buffer)).eq."END "//to_upper(trim(this%name)))then
          final_line = iline
          backspace(unit)
          exit tag_loop
       end if
       iline = iline + 1

       tag = trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag = trim(tag(:scan(tag,"=")-1))

       select case(trim(tag))
       case("NUM_INPUTS")
          call assign_val(buffer, num_inputs, itmp1)
       case("NUM_OUTPUTS")
          call assign_val(buffer, num_outputs, itmp1)
       case("COORD_DIM")
          call assign_val(buffer, coord_dim, itmp1)
       case("KERNEL_HIDDEN")
          call assign_val(buffer, kernel_hidden, itmp1)
       case("USE_BIAS")
          call assign_val(buffer, use_bias, itmp1)
       case("ACTIVATION")
          iline = iline - 1
          backspace(unit)
          activation = read_activation(unit, iline)
       case("KERNEL_INITIALISER", "KERNEL_INIT", "KERNEL_INITIALIZER")
          call assign_val(buffer, kernel_initialiser_name, itmp1)
       case("BIAS_INITIALISER", "BIAS_INIT", "BIAS_INITIALIZER")
          call assign_val(buffer, bias_initialiser_name, itmp1)
       case("WEIGHTS")
          kernel_initialiser_name = 'zeros'
          bias_initialiser_name   = 'zeros'
          param_line = iline
       case default
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
    bias_initialiser = initialiser_setup(bias_initialiser_name)

    call this%set_hyperparams( &
         num_outputs = num_outputs, &
         coord_dim = coord_dim, &
         kernel_hidden = kernel_hidden, &
         use_bias = use_bias, &
         activation = activation, &
         kernel_initialiser = kernel_initialiser, &
         bias_initialiser = bias_initialiser, &
         verbose = verbose_ &
    )
    call this%init(input_shape=[num_inputs, 0])


    ! Read weights
    !---------------------------------------------------------------------------
    if(param_line.eq.0)then
       write(0,*) "WARNING: WEIGHTS card in GRAPH_NOP not found"
    else
       call move(unit, param_line - iline, iostat=stat)

       do p = 1, size(this%params)
          num_vals = size(this%params(p)%val(:,1))
          allocate(data_list(num_vals), source=0._real32)
          c = 1
          do while(c .le. num_vals)
             read(unit,'(A)',iostat=stat) buffer
             if(stat.ne.0) exit
             k = icount(buffer)
             read(buffer,*,iostat=stat) (data_list(j), j=c, c+k-1)
             c = c + k
          end do
          this%params(p)%val(:,1) = data_list
          deallocate(data_list)
       end do

       read(unit,'(A)') buffer
       if(trim(adjustl(buffer)).ne."END WEIGHTS")then
          call stop_program("END WEIGHTS not where expected")
          return
       end if
    end if

    call move(unit, final_line - iline, iostat=stat)
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END "//to_upper(trim(this%name)))then
       write(err_msg,'("END ",A," not where expected")') &
            to_upper(this%name)
       call stop_program(err_msg)
       return
    end if

  end subroutine read_gno
!###############################################################################


!###############################################################################
  function read_graph_nop_layer(unit, verbose) result(layer)
    !! Read a graph NOP layer from file and return
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer
    integer :: verbose_ = 0

    if(present(verbose)) verbose_ = verbose
    allocate(layer, source=graph_nop_layer_type( &
         num_outputs=0, coord_dim=1))
    call layer%read(unit, verbose=verbose_)

  end function read_graph_nop_layer
!###############################################################################


!###############################################################################
  subroutine update_message_gno(this, input)
    !! Update message for the Graph Neural Operator layer
    !!
    !! Builds a differentiable computation graph through the diffstruc
    !! autodiff framework:
    !!
    !! input(1,s) : node features                    [F_in x num_vertices]
    !! input(2,s) : edge geometry / relative coords [coord_dim x num_edges]
    !!
    !! Pointer chain per sample s:
    !!   1. edge_kernels = gno_kernel_eval(coords, kernel_params, adj)
    !!   2. agg          = gno_aggregate(features, edge_kernels, adj)
    !!   3. bypass       = matmul(W, features)
    !!   4. z            = agg + bypass
    !!   5. z            = add_bias(z, b)          (if use_bias)
    !!   6. output       = activation(z)
    implicit none

    class(graph_nop_layer_type), intent(inout), target :: this
    class(array_type), dimension(:,:), intent(in), target :: input

    integer :: s, F_in, F_out
    type(array_type), pointer :: ptr1, ptr2, ptr3, ptr4

    F_in  = this%num_vertex_features(0)
    F_out = this%num_vertex_features(1)

    ! Allocate output array
    if(size(input, 1) .lt. 2)then
       call stop_program( &
            'graph_nop layer expects vertex and edge feature inputs' &
       )
       return
    end if

    if(allocated(this%output))then
       if(any(shape(this%output).ne.[2, size(input,2)]))then
          deallocate(this%output)
          allocate(this%output(2, size(input,2)))
       end if
    else
       allocate(this%output(2, size(input,2)))
    end if

    do s = 1, size(input, 2)

       ! Step 1: Evaluate kernel MLP on every edge
       ptr1 => gno_kernel_eval( &
            input(2,s), &               ! edge features [d, num_edges]
            this%params(1), &            ! packed kernel params
            this%graph(s)%adj_ia, &
            this%graph(s)%adj_ja, &
            this%coord_dim, this%kernel_hidden, F_in, F_out &
       )

       ! Step 2: Aggregate neighbour messages using per-edge kernels
       ptr2 => gno_aggregate( &
            input(1,s), &               ! features [F_in, num_v]
            ptr1, &                      ! edge kernels [F*F_in, num_edges]
            this%graph(s)%adj_ia, &
            this%graph(s)%adj_ja, &
            F_in, F_out &
       )

       ! Step 3: Linear bypass — W @ features
       ptr3 => matmul(this%params(2), input(1,s))

       ! Step 4: Combine aggregation with bypass
       ptr4 => ptr2 + ptr3

       ! Step 5: Add bias (if used)
       if(this%use_bias)then
          ptr4 => add_bias( &
               ptr4, this%params(3), dim=1, dim_act_on_shape=.true. &
          )
       end if

       ! Step 6: Apply activation
       ptr4 => this%activation%apply(ptr4)

       ! Store output
       call this%output(1,s)%zero_grad()
       call this%output(1,s)%assign_and_deallocate_source(ptr4)
       this%output(1,s)%is_temporary = .false.

       if(this%output(2,s)%allocated) call this%output(2,s)%deallocate()
       call this%output(2,s)%allocate(source=input(2,s)%val)
       call this%output(2,s)%zero_grad()
       call this%output(2,s)%set_requires_grad(.false.)
       this%output(2,s)%is_temporary = .false.
    end do

  end subroutine update_message_gno
!###############################################################################


!###############################################################################
  subroutine update_readout_gno(this)
    !! No graph-level readout needed — GNO produces node-level output
    implicit none
    class(graph_nop_layer_type), intent(inout), target :: this
  end subroutine update_readout_gno
!###############################################################################

end module athena__graph_nop_layer
