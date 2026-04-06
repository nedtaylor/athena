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
  implicit none


  private

  public :: create_from_onnx_avgpool_layer
  public :: create_from_onnx_batchnorm_layer
  public :: create_from_onnx_conv_layer
  public :: create_from_onnx_maxpool_layer
  public :: create_from_onnx_pad_layer
  public :: create_from_onnx_duvenaud_layer
  public :: create_from_onnx_kipf_layer



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

end module athena__onnx_creators
