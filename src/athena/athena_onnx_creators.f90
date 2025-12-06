module athena__onnx_creators
  !! Module containing ONNX layer creator functions
  use coreutils, only: stop_program, icount
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
    !! Instance of the batch normalization layer

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

end module athena__onnx_creators
