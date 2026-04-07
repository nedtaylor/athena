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
    use athena__initialiser_data, only: data_init_type
    implicit none

    character(*), intent(in) :: meta_key, meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: num_inputs, num_outputs, num_modes, verbose_
    logical :: use_bias
    character(64) :: activation_name, nop_prefix
    character(128) :: init_prefix
    integer :: k, pos, pos2, stat
    character(256) :: token, key, val
    real(real32), allocatable :: col_data(:)

    verbose_ = 0
    if(present(verbose)) verbose_ = verbose

    ! Defaults
    num_inputs = 0
    num_outputs = 0
    num_modes = 0
    use_bias = .true.
    activation_name = 'none'

    ! Parse hyperparameters from metadata value
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
       select case(trim(key))
       case('num_inputs')
          read(val, *) num_inputs
       case('num_outputs')
          read(val, *) num_outputs
       case('num_modes')
          read(val, *) num_modes
       case('use_bias')
          use_bias = read_logical_from_string(val, stat)
          if(stat .ne. 0)then
             call stop_program("create_from_onnx_dynamic_lno_layer: " // &
                  "invalid logical value for use_bias")
          end if
       case('activation')
          activation_name = trim(val)
       end select
    end do

    ! Derive initialiser name prefix from metadata key
    nop_prefix = trim(meta_key)
    pos = index(nop_prefix, 'athena_nop_')
    if(pos .gt. 0) nop_prefix = nop_prefix(pos+11:)

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

    character(*), intent(in) :: meta_key, meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: num_inputs, num_outputs, num_modes, verbose_
    logical :: use_bias
    character(64) :: activation_name, nop_prefix
    integer :: k, pos, pos2
    character(256) :: token, key, val

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

    character(*), intent(in) :: meta_key, meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: num_inputs, num_outputs, num_modes, verbose_
    logical :: use_bias
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

    character(*), intent(in) :: meta_key, meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: num_inputs, num_outputs, num_modes, verbose_
    logical :: use_bias
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

    character(*), intent(in) :: meta_key, meta_value
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, optional, intent(in) :: verbose
    class(base_layer_type), allocatable :: layer

    integer :: num_inputs, num_outputs, num_modes, key_dim, verbose_
    logical :: use_bias
    character(64) :: activation_name, nop_prefix
    integer :: k, pos, pos2
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


! =============================================================================
! NOP shared helper routines
! =============================================================================


!###############################################################################
  subroutine parse_nop_metadata(meta_value, &
       num_inputs, num_outputs, num_modes, use_bias, activation_name)
    !! Parse common NOP hyperparameters from metadata value string.
    implicit none

    character(*), intent(in) :: meta_value
    integer, intent(inout) :: num_inputs, num_outputs, num_modes
    logical, intent(inout) :: use_bias
    character(64), intent(inout) :: activation_name

    integer :: k, pos, pos2, stat
    character(256) :: token, key, val

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
       select case(trim(key))
       case('num_inputs')
          read(val, *) num_inputs
       case('num_outputs')
          read(val, *) num_outputs
       case('num_modes', 'num_basis')
          read(val, *) num_modes
       case('use_bias')
          use_bias = read_logical_from_string(val, stat)
          if(stat .ne. 0)then
             call stop_program("parse_nop_metadata: " // &
                   "invalid logical value for use_bias")
          end if
       case('activation')
          activation_name = trim(val)
       end select
    end do

  end subroutine parse_nop_metadata
!###############################################################################


!###############################################################################
  function extract_nop_prefix(meta_key) result(prefix)
    !! Extract the node prefix from an athena_nop_node_X metadata key.
    implicit none

    character(*), intent(in) :: meta_key
    character(64) :: prefix

    integer :: pos

    prefix = trim(meta_key)
    pos = index(prefix, 'athena_nop_')
    if(pos .gt. 0) prefix = prefix(pos+11:)

  end function extract_nop_prefix
!###############################################################################


!###############################################################################
  subroutine load_nop_param_from_inits( &
       param, prefix, suffix, inits, num_inits, dims)
    !! Load a parameter from ONNX initialisers into a diffstruc array.
    use diffstruc, only: array_type
    implicit none

    type(array_type), intent(inout) :: param
    character(*), intent(in) :: prefix, suffix
    type(onnx_initialiser_type), dimension(:), intent(in) :: inits
    integer, intent(in) :: num_inits
    integer, dimension(2), intent(in) :: dims

    integer :: k
    character(128) :: target_name
    real(real32), allocatable :: col_data(:)

    write(target_name, '(A,A)') trim(prefix), suffix

    do k = 1, num_inits
       if(trim(inits(k)%name) .ne. trim(target_name)) cycle
       if(.not.allocated(inits(k)%data)) cycle

       if(dims(2) .gt. 1)then
          ! 2D parameter — convert row-major to column-major
          allocate(col_data(size(inits(k)%data)))
          call row_to_col_major_2d( &
               inits(k)%data, col_data, dims(1), dims(2))
          param%val(:,1) = col_data
          deallocate(col_data)
       else
          ! 1D parameter
          param%val(:,1) = inits(k)%data
       end if
       return
    end do

  end subroutine load_nop_param_from_inits
!###############################################################################


!###############################################################################
   function read_logical_from_string(val, stat) result(logical_val)
      !! Convert a string to a logical value
      !!
      !! Acceptable true values: "true", "1" (case-insensitive)
      !! Acceptable false values: "false", "0" (case-insensitive)
      use coreutils, only: to_lower
      implicit none

      ! Arguments
      character(*), intent(in) :: val
      !! Input string to convert
      integer, intent(out) :: stat
      !! Status code: 0 for success, non-zero for invalid input

      logical :: logical_val
      !! Local variable for the resulting logical value

      stat = 0
      if(to_lower(trim(adjustl(val))) .eq. 'true' .or. trim(adjustl(val)) .eq. '1') then
          logical_val = .true.
      else if(to_lower(trim(adjustl(val))) .eq. 'false' .or. trim(adjustl(val)) .eq. '0') then
          logical_val = .false.
      else
          stat = 1
          logical_val = .false.  ! Default value in case of error
      end if

   end function read_logical_from_string
!###############################################################################

end module athena__onnx_creators
