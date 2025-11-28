module athena__onnx
  !! Module containing the types and interfaces for ONNX operations
  use athena__network, only: network_type
  implicit none


  private

  public :: write_onnx
  public :: read_onnx


  interface
     module subroutine write_onnx(file, network)
       class(network_type), intent(in) :: network
       !! Instance of network
       character(*), intent(in) :: file
       !! File to export the network to
     end subroutine write_onnx

    module function read_onnx(file, verbose) result(network)
       character(*), intent(in) :: file
       !! File to import the network from
       integer, optional, intent(in) :: verbose
       !! Verbosity level (0=quiet, 1=normal, 2=debug)
       type(network_type) :: network
       !! Network instance
     end function read_onnx
  end interface

end module athena__onnx
