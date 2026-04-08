module athena__onnx_creators
  !! Module containing ONNX layer creator functions
  use coreutils, only: real32, stop_program, icount
  use athena__base_layer, only: base_layer_type
  use athena__avgpool1d_layer, only: avgpool1d_layer_type
  use athena__avgpool2d_layer, only: avgpool2d_layer_type
  use athena__avgpool3d_layer, only: avgpool3d_layer_type
  use athena__batchnorm1d_layer, only: batchnorm1d_layer_type
  use athena__batchnorm2d_layer, only: batchnorm2d_layer_type
  use athena__batchnorm3d_layer, only: batchnorm3d_layer_type
  use athena__conv1d_layer, only: conv1d_layer_type
  use athena__conv2d_layer, only: conv2d_layer_type
  use athena__conv3d_layer, only: conv3d_layer_type
  use athena__pad1d_layer, only: pad1d_layer_type
  use athena__pad2d_layer, only: pad2d_layer_type
  use athena__pad3d_layer, only: pad3d_layer_type
  use athena__maxpool1d_layer, only: maxpool1d_layer_type
  use athena__maxpool2d_layer, only: maxpool2d_layer_type
  use athena__maxpool3d_layer, only: maxpool3d_layer_type

  use athena__misc_types, only: &
       onnx_node_type, onnx_initialiser_type, onnx_tensor_type
  use athena__onnx_nop_utils, only: parse_nop_metadata, extract_nop_prefix, &
       load_nop_param_from_inits, find_onnx_expanded_node_by_suffix, &
       find_node_initialiser_index, detect_onnx_expanded_nop_activation, &
       load_onnx_expanded_matrix_param
  use athena__onnx_utils, only: row_to_col_major_2d, &
       parse_space_separated_ints
  implicit none


  private

  public :: create_from_onnx_avgpool_layer
  public :: create_from_onnx_batchnorm_layer
  public :: create_from_onnx_conv_layer
  public :: create_from_onnx_maxpool_layer
  public :: create_from_onnx_pad_layer
  public :: create_from_onnx_duvenaud_layer
  public :: create_from_onnx_kipf_layer
  public :: create_from_onnx_dynamic_lno_layer
  public :: create_from_onnx_fixed_lno_layer
  public :: create_from_onnx_neural_operator_layer
  public :: create_from_onnx_orthogonal_nop_layer
  public :: create_from_onnx_orthogonal_attention_layer
  public :: classify_dynamic_lno_onnx_expanded_nop
  public :: build_dynamic_lno_onnx_expanded_nop
  public :: classify_fixed_lno_onnx_expanded_nop
  public :: build_fixed_lno_onnx_expanded_nop
  public :: classify_neural_operator_onnx_expanded_nop
  public :: build_neural_operator_onnx_expanded_nop
  public :: classify_spectral_filter_onnx_expanded_nop
  public :: build_spectral_filter_onnx_expanded_nop



contains

!###############################################################################
  function create_from_onnx_avgpool_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build avgpool layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: i, dim
    !! Loop variable and data rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    dim = size(value_info(1)%dims) - 2

    select case(dim)
    case(1)
       allocate(layer, source=avgpool1d_layer_type())
    case(2)
       allocate(layer, source=avgpool2d_layer_type())
    case(3)
       allocate(layer, source=avgpool3d_layer_type())
    case default
       call stop_program("create_from_onnx_avgpool_layer: " // &
            "unsupported pooling dimension")
    end select
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_avgpool_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_batchnorm_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build batchnorm layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the batch normalisation layer

    ! Local variables
    integer :: i, dim
    !! Loop variable and data rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    dim = size(value_info(1)%dims) - 2

    select case(dim)
    case(0)
       allocate(layer, source=batchnorm1d_layer_type())
    case(2)
       allocate(layer, source=batchnorm2d_layer_type())
    case(3)
       allocate(layer, source=batchnorm3d_layer_type())
    case default
       call stop_program("create_from_onnx_batchnorm_layer: " // &
            "unsupported batchnorm dimension")
    end select
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_batchnorm_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_conv_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build conv layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: i, dim
    !! Loop variable and data rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    dim = size(value_info(1)%dims) - 2

    select case(dim)
    case(1)
       allocate(layer, source=conv1d_layer_type())
    case(2)
       allocate(layer, source=conv2d_layer_type())
    case(3)
       allocate(layer, source=conv3d_layer_type())
    case default
       call stop_program("create_from_onnx_conv_layer: " // &
            "unsupported convolution dimension")
    end select
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_conv_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_maxpool_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build maxpool layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the 2D convolutional layer

    ! Local variables
    integer :: i, dim
    !! Loop variable and data rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    dim = size(value_info(1)%dims) - 2

    select case(dim)
    case(1)
       allocate(layer, source=maxpool1d_layer_type())
    case(2)
       allocate(layer, source=maxpool2d_layer_type())
    case(3)
       allocate(layer, source=maxpool3d_layer_type())
    case default
       call stop_program("create_from_onnx_maxpool_layer: " // &
            "unsupported pooling dimension")
    end select
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_maxpool_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_pad_layer( &
       node, initialisers, value_info, verbose &
  ) result(layer)
    !! Build pad layer from attributes and return layer
    implicit none

    ! Arguments
    type(onnx_node_type), intent(in) :: node
    !! ONNX node information
    type(onnx_initialiser_type), dimension(:), intent(in) :: initialisers
    !! ONNX initialiser information
    type(onnx_tensor_type), dimension(:), intent(in) :: value_info
    !! ONNX value info information
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Instance of the pad layer

    ! Local variables
    integer :: i, dim
    !! Loop variable and data rank
    integer :: verbose_ = 0
    !! Verbosity level

    if(present(verbose)) verbose_ = verbose
    dim = size(value_info(1)%dims) - 2

    select case(dim)
    case(1)
       allocate(layer, source=pad1d_layer_type(padding=[0], method="valid"))
    case(2)
       allocate(layer, source=pad2d_layer_type(padding=[0], method="valid"))
    case(3)
       allocate(layer, source=pad3d_layer_type(padding=[0], method="valid"))
    case default
       call stop_program("create_from_onnx_pad_layer: " // &
            "unsupported pad dimension")
    end select
    call layer%build_from_onnx(node, initialisers, value_info, verbose=verbose_)

  end function create_from_onnx_pad_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_duvenaud_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build Duvenaud message-passing layer from ONNX metadata and return layer
    use athena__duvenaud_msgpass_layer, only: duvenaud_msgpass_layer_type
    use athena__onnx_utils, only: row_to_col_major_2d, &
         parse_space_separated_ints
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key
    !! GNN metadata key (e.g. "athena_gnn_node_1")
    character(*), intent(in) :: meta_value
    !! Semicolon-separated GNN metadata value string
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers (valid entries only)
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed Duvenaud message-passing layer

    ! Local variables
    integer :: nts, n_out, min_deg, max_deg, num_deg
    integer, allocatable :: nv_arr(:), ne_arr(:)
    character(64) :: msg_activation
    character(128) :: gnn_prefix
    integer :: t, k, pos, pos2, verbose_
    character(256) :: meta_str, token, key, val
    character(128) :: init_prefix
    real(real32), allocatable :: col_data(:)

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    ! Parse hyperparameters from the metadata value string
    meta_str = meta_value
    nts = 1
    min_deg = 1
    max_deg = 10
    n_out = 1
    msg_activation = 'sigmoid'

    pos = 1
    do while(pos .le. len_trim(meta_str))
       pos2 = index(meta_str(pos:), ';')
       if(pos2 .eq. 0)then
          token = meta_str(pos:len_trim(meta_str))
          pos = len_trim(meta_str) + 1
       else
          token = meta_str(pos:pos+pos2-2)
          pos = pos + pos2
       end if
       k = index(token, '=')
       if(k .eq. 0) cycle
       key = trim(adjustl(token(1:k-1)))
       val = trim(adjustl(token(k+1:)))
       select case(trim(key))
       case('num_time_steps')
          read(val, *) nts
       case('min_vertex_degree')
          read(val, *) min_deg
       case('max_vertex_degree')
          read(val, *) max_deg
       case('num_vertex_features')
          call parse_space_separated_ints(val, nv_arr)
       case('num_edge_features')
          call parse_space_separated_ints(val, ne_arr)
       case('num_outputs')
          read(val, *) n_out
       case('message_activation')
          msg_activation = trim(val)
       end select
    end do

    if(.not.allocated(nv_arr)) allocate(nv_arr(1), source=1)
    if(.not.allocated(ne_arr)) allocate(ne_arr(1), source=0)
    num_deg = max_deg - min_deg + 1

    ! Derive initialiser name prefix from the metadata key
    gnn_prefix = trim(meta_key)
    pos = index(gnn_prefix, 'athena_gnn_')
    if(pos .gt. 0) gnn_prefix = gnn_prefix(pos+11:)

    block
      type(duvenaud_msgpass_layer_type) :: duvenaud_layer

      duvenaud_layer = duvenaud_msgpass_layer_type( &
           num_vertex_features = nv_arr, &
           num_edge_features = ne_arr, &
           num_time_steps = nts, &
           num_outputs = n_out, &
           min_vertex_degree = min_deg, &
           max_vertex_degree = max_deg, &
           message_activation = msg_activation &
      )

      do t = 1, nts
         ! Message weight: node_X_t{t}_W
         write(init_prefix, '(A,"_t",I0,"_W")') trim(gnn_prefix), t
         do k = 1, size(inits)
            if(trim(inits(k)%name) .eq. trim(init_prefix))then
               if(allocated(inits(k)%data) .and. &
                    allocated(duvenaud_layer%params))then
                  allocate(col_data(size(inits(k)%data)))
                  block
                    integer :: d, slice_size
                    slice_size = nv_arr(t+1) * (nv_arr(t) + ne_arr(1))
                    do d = 1, num_deg
                       call row_to_col_major_2d( &
                            inits(k)%data((d-1)*slice_size+1:d*slice_size), &
                            col_data((d-1)*slice_size+1:d*slice_size), &
                            nv_arr(t+1), nv_arr(t) + ne_arr(1))
                    end do
                  end block
                  duvenaud_layer%params(t)%val(:,1) = col_data
                  deallocate(col_data)
               end if
               exit
            end if
         end do

         ! Readout weight: node_X_ro_t{t}_R
         write(init_prefix, '(A,"_ro_t",I0,"_R")') trim(gnn_prefix), t
         do k = 1, size(inits)
            if(trim(inits(k)%name) .eq. trim(init_prefix))then
               if(allocated(inits(k)%data) .and. &
                    allocated(duvenaud_layer%params))then
                  allocate(col_data(size(inits(k)%data)))
                  call row_to_col_major_2d( &
                       inits(k)%data, col_data, n_out, nv_arr(t+1))
                  duvenaud_layer%params(nts + t)%val(:,1) = col_data
                  deallocate(col_data)
               end if
               exit
            end if
         end do
      end do

      allocate(layer, source=duvenaud_layer)
    end block

  end function create_from_onnx_duvenaud_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_kipf_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build Kipf GCN layer from ONNX metadata and return layer
    use athena__kipf_msgpass_layer, only: kipf_msgpass_layer_type
    use athena__onnx_utils, only: row_to_col_major_2d, &
         parse_space_separated_ints
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key
    !! GNN metadata key (e.g. "athena_gnn_node_1")
    character(*), intent(in) :: meta_value
    !! Semicolon-separated GNN metadata value string
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers (valid entries only)
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed Kipf GCN layer

    ! Local variables
    integer :: nts
    integer, allocatable :: nv_arr(:)
    character(64) :: msg_activation
    character(128) :: gnn_prefix
    integer :: t, k, pos, pos2, verbose_
    character(256) :: meta_str, token, key, val
    character(128) :: init_prefix
    real(real32), allocatable :: col_data(:)

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    ! Parse hyperparameters from the metadata value string
    meta_str = meta_value
    nts = 1
    msg_activation = 'sigmoid'

    pos = 1
    do while(pos .le. len_trim(meta_str))
       pos2 = index(meta_str(pos:), ';')
       if(pos2 .eq. 0)then
          token = meta_str(pos:len_trim(meta_str))
          pos = len_trim(meta_str) + 1
       else
          token = meta_str(pos:pos+pos2-2)
          pos = pos + pos2
       end if
       k = index(token, '=')
       if(k .eq. 0) cycle
       key = trim(adjustl(token(1:k-1)))
       val = trim(adjustl(token(k+1:)))
       select case(trim(key))
       case('num_time_steps')
          read(val, *) nts
       case('num_vertex_features')
          call parse_space_separated_ints(val, nv_arr)
       case('message_activation')
          msg_activation = trim(val)
       end select
    end do

    if(.not.allocated(nv_arr)) allocate(nv_arr(1), source=1)

    ! Derive initialiser name prefix from the metadata key
    gnn_prefix = trim(meta_key)
    pos = index(gnn_prefix, 'athena_gnn_')
    if(pos .gt. 0) gnn_prefix = gnn_prefix(pos+11:)

    block
      type(kipf_msgpass_layer_type) :: kipf_layer

      kipf_layer = kipf_msgpass_layer_type( &
           num_vertex_features = nv_arr, &
           num_time_steps = nts, &
           activation = trim(msg_activation) &
      )

      do t = 1, nts
         ! Message weight: node_X_t{t}_W
         write(init_prefix, '(A,"_t",I0,"_W")') trim(gnn_prefix), t
         do k = 1, size(inits)
            if(trim(inits(k)%name) .eq. trim(init_prefix))then
               if(allocated(inits(k)%data) .and. &
                    allocated(kipf_layer%params))then
                  allocate(col_data(size(inits(k)%data)))
                  call row_to_col_major_2d( &
                       inits(k)%data, col_data, nv_arr(t+1), nv_arr(t))
                  kipf_layer%params(t)%val(:,1) = col_data
                  deallocate(col_data)
               end if
               exit
            end if
         end do
      end do

      allocate(layer, source=kipf_layer)
    end block

  end function create_from_onnx_kipf_layer
!###############################################################################


!###############################################################################
! NOP ONNX creators
!###############################################################################


!###############################################################################
  function create_from_onnx_dynamic_lno_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build dynamic LNO layer from ONNX metadata and return layer
    use athena__dynamic_lno_layer, only: dynamic_lno_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key, meta_value
    !! NOP metadata key/value pair
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers containing parameter tensors
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed dynamic LNO layer

    ! Local variables
    integer :: num_inputs, num_outputs, num_modes, verbose_
    !! Parsed layer dimensions and effective verbosity level
    logical :: use_bias
    !! Whether the imported layer uses bias
    character(64) :: activation_name, nop_prefix

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    num_inputs = 0
    num_outputs = 0
    num_modes = 0
    use_bias = .true.
    activation_name = 'none'

    call parse_nop_metadata(meta_value, &
         num_inputs, num_outputs, num_modes, use_bias, activation_name)

    nop_prefix = extract_nop_prefix(meta_key)

    block
      type(dynamic_lno_layer_type) :: lno_layer

      lno_layer = dynamic_lno_layer_type( &
           num_outputs = num_outputs, &
           num_modes = num_modes, &
           num_inputs = num_inputs, &
           use_bias = use_bias, &
           activation = trim(activation_name) &
      )

      ! Load params: (1) mu, (2) beta, (3) W, (4) b
      call load_nop_param_from_inits( &
           lno_layer%params(1), nop_prefix, '_param1', &
           inits, size(inits), [num_modes, 1])
      call load_nop_param_from_inits( &
           lno_layer%params(2), nop_prefix, '_param2', &
           inits, size(inits), [num_modes, 1])
      call load_nop_param_from_inits( &
           lno_layer%params(3), nop_prefix, '_param3', &
           inits, size(inits), [num_outputs, num_inputs])
      if(use_bias)then
         call load_nop_param_from_inits( &
              lno_layer%params(4), nop_prefix, '_param4', &
              inits, size(inits), [num_outputs, 1])
      end if

      allocate(layer, source=lno_layer)
    end block

  end function create_from_onnx_dynamic_lno_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_fixed_lno_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build fixed LNO layer from ONNX metadata and return layer
    use athena__fixed_lno_layer, only: fixed_lno_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key, meta_value
    !! NOP metadata key/value pair
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers containing parameter tensors
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed fixed LNO layer

    ! Local variables
    integer :: num_inputs, num_outputs, num_modes, verbose_
    !! Parsed layer dimensions and effective verbosity level
    logical :: use_bias
    !! Whether the imported layer uses bias
    character(64) :: activation_name, nop_prefix

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    num_inputs = 0; num_outputs = 0; num_modes = 0
    use_bias = .true.; activation_name = 'none'

    call parse_nop_metadata(meta_value, &
         num_inputs, num_outputs, num_modes, use_bias, activation_name)

    nop_prefix = extract_nop_prefix(meta_key)

    block
      type(fixed_lno_layer_type) :: lno_layer

      lno_layer = fixed_lno_layer_type( &
           num_outputs = num_outputs, &
           num_modes = num_modes, &
           num_inputs = num_inputs, &
           use_bias = use_bias, &
           activation = trim(activation_name) &
      )

      ! params: (1) R [modes x modes], (2) W [n_out x n_in], (3) b [n_out]
      call load_nop_param_from_inits( &
           lno_layer%params(1), nop_prefix, '_param1', &
           inits, size(inits), [num_modes, num_modes])
      call load_nop_param_from_inits( &
           lno_layer%params(2), nop_prefix, '_param2', &
           inits, size(inits), [num_outputs, num_inputs])
      if(use_bias)then
         call load_nop_param_from_inits( &
              lno_layer%params(3), nop_prefix, '_param3', &
              inits, size(inits), [num_outputs, 1])
      end if

      allocate(layer, source=lno_layer)
    end block

  end function create_from_onnx_fixed_lno_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_neural_operator_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build neural_operator layer from ONNX metadata and return layer
    use athena__neural_operator_layer, only: neural_operator_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key, meta_value
    !! NOP metadata key/value pair
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers containing parameter tensors
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed neural operator layer

    ! Local variables
    integer :: num_inputs, num_outputs, num_modes, verbose_
    !! Parsed layer dimensions and effective verbosity level
    logical :: use_bias
    !! Whether the imported layer uses bias
    character(64) :: activation_name, nop_prefix

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    num_inputs = 0; num_outputs = 0; num_modes = 0
    use_bias = .true.; activation_name = 'none'

    call parse_nop_metadata(meta_value, &
         num_inputs, num_outputs, num_modes, use_bias, activation_name)

    nop_prefix = extract_nop_prefix(meta_key)

    block
      type(neural_operator_layer_type) :: nop_layer

      nop_layer = neural_operator_layer_type( &
           num_outputs = num_outputs, &
           num_inputs = num_inputs, &
           use_bias = use_bias, &
           activation = trim(activation_name) &
      )

      ! params: (1) W [n_out x n_in], (2) W_k [n_out], (3) b [n_out]
      call load_nop_param_from_inits( &
           nop_layer%params(1), nop_prefix, '_param1', &
           inits, size(inits), [num_outputs, num_inputs])
      call load_nop_param_from_inits( &
           nop_layer%params(2), nop_prefix, '_param2', &
           inits, size(inits), [num_outputs, 1])
      if(use_bias)then
         call load_nop_param_from_inits( &
              nop_layer%params(3), nop_prefix, '_param3', &
              inits, size(inits), [num_outputs, 1])
      end if

      allocate(layer, source=nop_layer)
    end block

  end function create_from_onnx_neural_operator_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_orthogonal_nop_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build orthogonal NOP block from ONNX metadata and return layer
    use athena__orthogonal_nop_block, only: orthogonal_nop_block_type
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key, meta_value
    !! NOP metadata key/value pair
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers containing parameter tensors
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed orthogonal NOP layer

    ! Local variables
    integer :: num_inputs, num_outputs, num_modes, verbose_
    !! Parsed layer dimensions and effective verbosity level
    logical :: use_bias
    !! Whether the imported layer uses bias
    character(64) :: activation_name, nop_prefix

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    num_inputs = 0; num_outputs = 0; num_modes = 0
    use_bias = .true.; activation_name = 'none'

    call parse_nop_metadata(meta_value, &
         num_inputs, num_outputs, num_modes, use_bias, activation_name)

    nop_prefix = extract_nop_prefix(meta_key)

    block
      type(orthogonal_nop_block_type) :: ono_layer

      ono_layer = orthogonal_nop_block_type( &
           num_outputs = num_outputs, &
           num_basis = num_modes, &
           num_inputs = num_inputs, &
           use_bias = use_bias, &
           activation = trim(activation_name) &
      )

      ! params: (1) R [nb x nb], (2) B [n_in x nb], (3) W [n_out x n_in],
      !         (4) b [n_out]
      call load_nop_param_from_inits( &
           ono_layer%params(1), nop_prefix, '_param1', &
           inits, size(inits), [num_modes, num_modes])
      call load_nop_param_from_inits( &
           ono_layer%params(2), nop_prefix, '_param2', &
           inits, size(inits), [num_inputs, num_modes])
      call load_nop_param_from_inits( &
           ono_layer%params(3), nop_prefix, '_param3', &
           inits, size(inits), [num_outputs, num_inputs])
      if(use_bias)then
         call load_nop_param_from_inits( &
              ono_layer%params(4), nop_prefix, '_param4', &
              inits, size(inits), [num_outputs, 1])
      end if

      allocate(layer, source=ono_layer)
    end block

  end function create_from_onnx_orthogonal_nop_layer
!###############################################################################


!###############################################################################
  function create_from_onnx_orthogonal_attention_layer( &
       meta_key, meta_value, inits, verbose &
  ) result(layer)
    !! Build orthogonal attention layer from ONNX metadata and return layer
    use athena__orthogonal_attention_layer, only: &
         orthogonal_attention_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: meta_key, meta_value
    !! NOP metadata key/value pair
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    !! ONNX initialisers containing parameter tensors
    integer, optional, intent(in) :: verbose
    !! Verbosity level
    class(base_layer_type), allocatable :: layer
    !! Constructed orthogonal attention layer

    ! Local variables
    integer :: num_inputs, num_outputs, num_modes, key_dim, verbose_
    !! Parsed layer dimensions, key dimension and effective verbosity level
    logical :: use_bias
    !! Whether the imported layer uses bias
    character(64) :: activation_name, nop_prefix
    integer :: k, pos, pos2
    !! Parsing indices
    character(256) :: token, key, val

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    num_inputs = 0; num_outputs = 0; num_modes = 0; key_dim = 0
    use_bias = .true.; activation_name = 'none'

    call parse_nop_metadata(meta_value, &
         num_inputs, num_outputs, num_modes, use_bias, activation_name)

    ! Also parse key_dim
    pos = 1
    do while(pos .le. len_trim(meta_value))
       pos2 = index(meta_value(pos:), ';')
       if(pos2 .eq. 0)then
          token = meta_value(pos:len_trim(meta_value))
          pos = len_trim(meta_value) + 1
       else
          token = meta_value(pos:pos+pos2-2)
          pos = pos + pos2
       end if
       k = index(token, '=')
       if(k .eq. 0) cycle
       key = trim(adjustl(token(1:k-1)))
       val = trim(adjustl(token(k+1:)))
       if(trim(key) .eq. 'key_dim') read(val, *) key_dim
    end do

    nop_prefix = extract_nop_prefix(meta_key)

    block
      type(orthogonal_attention_layer_type) :: attn_layer

      attn_layer = orthogonal_attention_layer_type( &
           num_outputs = num_outputs, &
           num_basis = num_modes, &
           key_dim = key_dim, &
           num_inputs = num_inputs, &
           use_bias = use_bias, &
           activation = trim(activation_name) &
      )

      ! params: (1) W_Q, (2) W_K, (3) W_V, (4) B, (5) W, (6) b
      call load_nop_param_from_inits( &
           attn_layer%params(1), nop_prefix, '_param1', &
           inits, size(inits), [key_dim, num_inputs])
      call load_nop_param_from_inits( &
           attn_layer%params(2), nop_prefix, '_param2', &
           inits, size(inits), [key_dim, num_inputs])
      call load_nop_param_from_inits( &
           attn_layer%params(3), nop_prefix, '_param3', &
           inits, size(inits), [num_outputs, num_inputs])
      call load_nop_param_from_inits( &
           attn_layer%params(4), nop_prefix, '_param4', &
           inits, size(inits), [num_inputs, num_modes])
      call load_nop_param_from_inits( &
           attn_layer%params(5), nop_prefix, '_param5', &
           inits, size(inits), [num_outputs, num_inputs])
      if(use_bias)then
         call load_nop_param_from_inits( &
              attn_layer%params(6), nop_prefix, '_param6', &
              inits, size(inits), [num_outputs, 1])
      end if

      allocate(layer, source=attn_layer)
    end block

  end function create_from_onnx_orthogonal_attention_layer
!###############################################################################


!###############################################################################
! Expanded-ONNX NOP layer classifiers and builders
!###############################################################################


!###############################################################################
  logical function classify_dynamic_lno_onnx_expanded_nop(prefix, nodes, &
       num_nodes)
    !! Return true when the expanded-ONNX node cluster
    !! for prefix is a dynamic LNO.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Expanded-ONNX layer prefix (e.g. "layer1")
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries

    classify_dynamic_lno_onnx_expanded_nop = &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'Exp') .gt. 0 &
         .and. &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'Exp_1') .gt. 0

  end function classify_dynamic_lno_onnx_expanded_nop
!###############################################################################


!###############################################################################
  function build_dynamic_lno_onnx_expanded_nop( &
       prefix, nodes, num_nodes, inits, num_inits) result(layer)
    !! Build one dynamic LNO layer from an expanded-ONNX node cluster.
    use athena__dynamic_lno_layer, only: dynamic_lno_layer_type
    use athena__onnx_nop_utils, only: infer_dynamic_lno_poles
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix (e.g. layer1)
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid initialiser entries
    class(base_layer_type), allocatable :: layer
    !! Constructed dynamic LNO layer

    ! Local variables
    type(dynamic_lno_layer_type) :: typed_layer
    !! Concrete layer object before up-casting
    integer :: exp_idx, exp1_idx, mul_idx, matmul2_idx, add1_idx
    !! Node indices for the dynamic LNO decomposition
    integer :: e_idx, d_idx, beta_idx, w_idx, b_idx
    !! Initialiser indices used to populate the layer parameters
    integer :: num_inputs, num_outputs, num_modes
    !! Reconstructed layer dimensions
    logical :: use_bias
    !! Whether the graph includes a bias add
    character(64) :: activation_name
    !! Activation reconstructed from the tail of the graph
    real(real32), allocatable :: poles(:)
    !! Dynamic poles reconstructed from exported encoder/decoder arguments

    exp_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Exp')
    exp1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Exp_1')
    mul_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Mul')
    matmul2_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'MatMul_2')
    add1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Add_1')

    if(exp_idx .le. 0 .or. exp1_idx .le. 0 .or. mul_idx .le. 0 .or. &
         matmul2_idx .le. 0)then
       call stop_program('Dynamic LNO ONNX cluster is incomplete for ' // &
            trim(prefix))
    end if

    e_idx = find_node_initialiser_index(nodes(exp_idx), inits, num_inits)
    d_idx = find_node_initialiser_index(nodes(exp1_idx), inits, num_inits)
    beta_idx = find_node_initialiser_index(nodes(mul_idx), inits, num_inits)
    w_idx = find_node_initialiser_index(nodes(matmul2_idx), inits, num_inits)

    if(min(e_idx, d_idx, beta_idx, w_idx) .le. 0)then
       call stop_program('Dynamic LNO ONNX parameters are missing for ' // &
            trim(prefix))
    end if

    num_modes = inits(beta_idx)%dims(1)
    num_outputs = inits(w_idx)%dims(1)
    num_inputs = inits(w_idx)%dims(2)
    use_bias = add1_idx .gt. 0
    activation_name = detect_onnx_expanded_nop_activation( &
         prefix, nodes, num_nodes)

    typed_layer = dynamic_lno_layer_type( &
         num_outputs=num_outputs, num_modes=num_modes, &
         num_inputs=num_inputs, use_bias=use_bias, &
         activation=trim(activation_name))

    allocate(poles(num_modes))
    call infer_dynamic_lno_poles( &
         inits(e_idx), inits(d_idx), num_inputs, num_outputs, poles)
    typed_layer%params(1)%val(:,1) = poles
    typed_layer%params(2)%val(:,1) = inits(beta_idx)%data
    call load_onnx_expanded_matrix_param( &
         typed_layer%params(3), inits(w_idx), num_outputs, num_inputs)

    if(use_bias)then
       b_idx = find_node_initialiser_index(nodes(add1_idx), inits, num_inits)
       if(b_idx .le. 0)then
          call stop_program('Dynamic LNO bias initialiser missing for ' // &
               trim(prefix))
       end if
       typed_layer%params(4)%val(:,1) = inits(b_idx)%data
    end if

    allocate(layer, source=typed_layer)

  end function build_dynamic_lno_onnx_expanded_nop
!###############################################################################


!###############################################################################
  logical function classify_fixed_lno_onnx_expanded_nop(prefix, nodes, &
       num_nodes)
    !! Return true when the expanded-ONNX node cluster
    !! for prefix is a fixed LNO.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Expanded-ONNX layer prefix (e.g. "layer2")
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries

    !! Fixed LNO has MatMul_3 but not the Exp/Exp_1 pair of dynamic LNO
    classify_fixed_lno_onnx_expanded_nop = &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'MatMul_3') .gt. 0 &
         .and. &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'Exp') .le. 0

  end function classify_fixed_lno_onnx_expanded_nop
!###############################################################################


!###############################################################################
  function build_fixed_lno_onnx_expanded_nop( &
       prefix, nodes, num_nodes, inits, num_inits) result(layer)
    !! Build one fixed LNO layer from an expanded-ONNX node cluster.
    use athena__fixed_lno_layer, only: fixed_lno_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix (e.g. layer2)
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid initialiser entries
    class(base_layer_type), allocatable :: layer
    !! Constructed fixed LNO layer

    ! Local variables
    type(fixed_lno_layer_type) :: typed_layer
    !! Concrete layer object before up-casting
    integer :: matmul1_idx, matmul3_idx, add1_idx
    !! Node indices for learnable parameters in the fixed LNO decomposition
    integer :: r_idx, w_idx, b_idx
    !! Initialiser indices used to populate the layer parameters
    integer :: num_inputs, num_outputs, num_modes
    !! Reconstructed layer dimensions
    logical :: use_bias
    !! Whether the graph includes a bias add
    character(64) :: activation_name
    !! Activation reconstructed from the tail of the graph

    matmul1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'MatMul_1')
    matmul3_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'MatMul_3')
    add1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Add_1')

    if(matmul1_idx .le. 0 .or. matmul3_idx .le. 0)then
       call stop_program('Fixed LNO ONNX cluster is incomplete for ' // &
            trim(prefix))
    end if

    r_idx = find_node_initialiser_index(nodes(matmul1_idx), inits, num_inits)
    w_idx = find_node_initialiser_index(nodes(matmul3_idx), inits, num_inits)

    if(min(r_idx, w_idx) .le. 0)then
       call stop_program('Fixed LNO ONNX parameters are missing for ' // &
            trim(prefix))
    end if

    num_modes = inits(r_idx)%dims(1)
    num_outputs = inits(w_idx)%dims(1)
    num_inputs = inits(w_idx)%dims(2)
    use_bias = add1_idx .gt. 0
    activation_name = detect_onnx_expanded_nop_activation( &
         prefix, nodes, num_nodes)

    typed_layer = fixed_lno_layer_type( &
         num_outputs=num_outputs, num_modes=num_modes, &
         num_inputs=num_inputs, use_bias=use_bias, &
         activation=trim(activation_name))

    call load_onnx_expanded_matrix_param( &
         typed_layer%params(1), inits(r_idx), num_modes, num_modes)
    call load_onnx_expanded_matrix_param( &
         typed_layer%params(2), inits(w_idx), num_outputs, num_inputs)

    if(use_bias)then
       b_idx = find_node_initialiser_index(nodes(add1_idx), inits, num_inits)
       if(b_idx .le. 0)then
          call stop_program('Fixed LNO bias initialiser missing for ' // &
               trim(prefix))
       end if
       typed_layer%params(3)%val(:,1) = inits(b_idx)%data
    end if

    allocate(layer, source=typed_layer)

  end function build_fixed_lno_onnx_expanded_nop
!###############################################################################


!###############################################################################
  logical function classify_neural_operator_onnx_expanded_nop(prefix, nodes, &
       num_nodes)
    !! Return true when the expanded-ONNX node cluster
    !! for prefix is a neural operator.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Expanded-ONNX layer prefix (e.g. "layer3")
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries

    !! Neural operator has ReduceMean but not Exp/Exp_1 or MatMul_3
    classify_neural_operator_onnx_expanded_nop = &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'ReduceMean') .gt. 0

  end function classify_neural_operator_onnx_expanded_nop
!###############################################################################


!###############################################################################
  function build_neural_operator_onnx_expanded_nop( &
       prefix, nodes, num_nodes, inits, num_inits) result(layer)
    !! Build one neural operator layer from an expanded-ONNX node cluster.
    use athena__neural_operator_layer, only: neural_operator_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix (e.g. layer3)
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid initialiser entries
    class(base_layer_type), allocatable :: layer
    !! Constructed neural operator layer

    ! Local variables
    type(neural_operator_layer_type) :: typed_layer
    !! Concrete layer object before up-casting
    integer :: matmul_idx, mul_idx, add1_idx
    !! Node indices for the neural operator decomposition
    integer :: w_idx, wk_idx, b_idx
    !! Initialiser indices used to populate the layer parameters
    integer :: num_inputs, num_outputs
    !! Reconstructed layer dimensions
    logical :: use_bias
    !! Whether the graph includes a bias add
    character(64) :: activation_name
    !! Activation reconstructed from the tail of the graph

    matmul_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'MatMul')
    mul_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Mul')
    add1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Add_1')

    if(matmul_idx .le. 0 .or. mul_idx .le. 0)then
       call stop_program('Neural operator ONNX cluster is incomplete for ' // &
            trim(prefix))
    end if

    w_idx = find_node_initialiser_index(nodes(matmul_idx), inits, num_inits)
    wk_idx = find_node_initialiser_index(nodes(mul_idx), inits, num_inits)

    if(min(w_idx, wk_idx) .le. 0)then
       call stop_program('Neural operator ONNX parameters are missing for ' // &
            trim(prefix))
    end if

    num_outputs = inits(w_idx)%dims(1)
    num_inputs = inits(w_idx)%dims(2)
    use_bias = add1_idx .gt. 0
    activation_name = detect_onnx_expanded_nop_activation( &
         prefix, nodes, num_nodes)

    typed_layer = neural_operator_layer_type( &
         num_outputs=num_outputs, num_inputs=num_inputs, &
         use_bias=use_bias, activation=trim(activation_name))

    call load_onnx_expanded_matrix_param( &
         typed_layer%params(1), inits(w_idx), num_outputs, num_inputs)
    typed_layer%params(2)%val(:,1) = inits(wk_idx)%data

    if(use_bias)then
       b_idx = find_node_initialiser_index(nodes(add1_idx), inits, num_inits)
       if(b_idx .le. 0)then
          call stop_program('Neural operator bias initialiser missing for ' // &
               trim(prefix))
       end if
       typed_layer%params(3)%val(:,1) = inits(b_idx)%data
    end if

    allocate(layer, source=typed_layer)

  end function build_neural_operator_onnx_expanded_nop
!###############################################################################


!###############################################################################
  logical function classify_spectral_filter_onnx_expanded_nop(prefix, nodes, &
       num_nodes)
    !! Return true when the expanded-ONNX node cluster
    !! for prefix is a spectral filter.
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Expanded-ONNX layer prefix (e.g. "layer4")
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries

    !! Spectral filter has Mul but not Exp/Exp_1 or ReduceMean or MatMul_3
    classify_spectral_filter_onnx_expanded_nop = &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'Mul') .gt. 0 &
         .and. &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'Exp') .le. 0 &
         .and. &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'ReduceMean') .le. 0 &
         .and. &
         find_onnx_expanded_node_by_suffix( &
              nodes, num_nodes, prefix, 'MatMul_3') .le. 0

  end function classify_spectral_filter_onnx_expanded_nop
!###############################################################################


!###############################################################################
  function build_spectral_filter_onnx_expanded_nop( &
       prefix, nodes, num_nodes, inits, num_inits) result(layer)
    !! Build one spectral filter layer from an expanded-ONNX node cluster.
    use athena__spectral_filter_layer, only: spectral_filter_layer_type
    implicit none

    ! Arguments
    character(*), intent(in) :: prefix
    !! Layer node prefix (e.g. layer4)
    type(onnx_node_type), intent(in) :: nodes(:)
    !! Parsed ONNX nodes
    integer, intent(in) :: num_nodes
    !! Number of valid node entries
    type(onnx_initialiser_type), intent(in) :: inits(:)
    !! Parsed ONNX initialisers
    integer, intent(in) :: num_inits
    !! Number of valid initialiser entries
    class(base_layer_type), allocatable :: layer
    !! Constructed spectral filter layer

    ! Local variables
    type(spectral_filter_layer_type) :: typed_layer
    !! Concrete layer object before up-casting
    integer :: mul_idx, matmul_idx, add1_idx
    !! Node indices for the spectral filter decomposition
    integer :: ws_idx, w_idx, b_idx
    !! Initialiser indices used to populate the layer parameters
    integer :: num_inputs, num_outputs, num_modes
    !! Reconstructed layer dimensions
    logical :: use_bias
    !! Whether the graph includes a bias add
    character(64) :: activation_name
    !! Activation reconstructed from the tail of the graph

    mul_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Mul')
    matmul_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'MatMul')
    add1_idx = find_onnx_expanded_node_by_suffix(nodes, num_nodes, prefix, &
         'Add_1')

    if(mul_idx .le. 0 .or. matmul_idx .le. 0)then
       call stop_program('Spectral filter ONNX cluster is incomplete for ' // &
            trim(prefix))
    end if

    ws_idx = find_node_initialiser_index(nodes(mul_idx), inits, num_inits)
    w_idx = find_node_initialiser_index(nodes(matmul_idx), inits, num_inits)

    if(min(ws_idx, w_idx) .le. 0)then
       call stop_program('Spectral filter ONNX parameters are missing for ' // &
            trim(prefix))
    end if

    num_modes = inits(ws_idx)%dims(1)
    num_outputs = inits(w_idx)%dims(1)
    num_inputs = inits(w_idx)%dims(2)
    use_bias = add1_idx .gt. 0
    activation_name = detect_onnx_expanded_nop_activation( &
         prefix, nodes, num_nodes)

    typed_layer = spectral_filter_layer_type( &
         num_outputs=num_outputs, num_modes=num_modes, &
         num_inputs=num_inputs, use_bias=use_bias, &
         activation=trim(activation_name))

    typed_layer%params(1)%val(:,1) = inits(ws_idx)%data
    call load_onnx_expanded_matrix_param( &
         typed_layer%params(2), inits(w_idx), num_outputs, num_inputs)

    if(use_bias)then
       b_idx = find_node_initialiser_index(nodes(add1_idx), inits, num_inits)
       if(b_idx .le. 0)then
          call stop_program('Spectral filter bias initialiser missing for ' // &
               trim(prefix))
       end if
       typed_layer%params(3)%val(:,1) = inits(b_idx)%data
    end if

    allocate(layer, source=typed_layer)

  end function build_spectral_filter_onnx_expanded_nop
!###############################################################################


end module athena__onnx_creators
