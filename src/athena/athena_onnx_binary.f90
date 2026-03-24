module athena__onnx_binary
  !! Module providing interfaces for binary ONNX (.onnx protobuf) I/O
  use athena__network, only: network_type
  implicit none

  private

  public :: write_onnx_binary
  public :: read_onnx_binary


  interface
     module subroutine write_onnx_binary(file, network)
       class(network_type), intent(in) :: network
       !! Instance of network
       character(*), intent(in) :: file
       !! File to export the network to (binary .onnx format)
     end subroutine write_onnx_binary

     module function read_onnx_binary(file, verbose) result(network)
       character(*), intent(in) :: file
       !! File to import the network from (binary .onnx format)
       integer, optional, intent(in) :: verbose
       !! Verbosity level (0=quiet, 1=normal, 2=debug)
       type(network_type) :: network
       !! Network instance
     end function read_onnx_binary
  end interface

end module athena__onnx_binary
