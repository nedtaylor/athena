module athena__onnx
  !! Module containing the types and interfaces for ONNX operations.
  !! Supports both text-based ONNX format and binary protobuf .onnx format.
  use athena__network, only: network_type
  use athena__onnx_binary, only: write_onnx_binary, read_onnx_binary
  implicit none


  private

  public :: write_onnx
  public :: read_onnx
  public :: write_onnx_binary
  public :: read_onnx_binary
  public :: save_onnx
  public :: load_onnx


  interface
     module subroutine write_onnx(file, network, format)
       class(network_type), intent(in) :: network
       !! Instance of network
       character(*), intent(in) :: file
       !! File to export the network to
       class(*), optional, intent(in) :: format
       !! Export format: 'athena_abstract' (default) or 'onnx_expanded'
     end subroutine write_onnx

     module function read_onnx(file, verbose) result(network)
       character(*), intent(in) :: file
       !! File to import the network from (text format)
       integer, optional, intent(in) :: verbose
       !! Verbosity level (0=quiet, 1=normal, 2=debug)
       type(network_type) :: network
       !! Network instance
     end function read_onnx

     module subroutine save_onnx(file, network)
       class(network_type), intent(in) :: network
       !! Instance of network
       character(*), intent(in) :: file
       !! File to export the network to.
       !! Format is auto-detected from file extension:
       !!   .onnx -> binary protobuf format
       !!   anything else -> text format
     end subroutine save_onnx

     module function load_onnx(file, verbose) result(network)
       character(*), intent(in) :: file
       !! File to import the network from.
       !! Format is auto-detected from file extension:
       !!   .onnx -> binary protobuf format
       !!   anything else -> text format
       integer, optional, intent(in) :: verbose
       !! Verbosity level (0=quiet, 1=normal, 2=debug)
       type(network_type) :: network
       !! Network instance
     end function load_onnx
  end interface

end module athena__onnx
