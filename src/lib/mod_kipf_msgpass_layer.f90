module athena__kipf_msgpass_layer
  !! Module containing the types and interfacees of a message passing layer
  use athena__constants, only: real32
  use graphstruc, only: graph_type
  use athena__misc_types, only: array_type, array2d_type
  use athena__msgpass_layer, only: msgpass_layer_type
  implicit none


  private

  public :: kipf_msgpass_layer_type


!-------------------------------------------------------------------------------
! Message passing layer
!-------------------------------------------------------------------------------
  type, extends(msgpass_layer_type) :: kipf_msgpass_layer_type

     ! this is for chen 2021 et al
     !  type(array2d_type), dimension(:), allocatable :: edge_weight
     !  !! Weights for the edges
     !  type(array2d_type), dimension(:), allocatable :: vertex_weight
     !  !! Weights for the vertices

     real(real32), dimension(:,:,:), allocatable :: weight
     !! Weights for the message passing layer

   contains

     procedure, pass(this) :: read => read_kipf
     !! Read the message passing layer

     procedure, pass(this) :: forward => forward_rank
     !! Forward pass for message passing layer
     procedure, pass(this) :: backward => backward_rank
     !! Backward pass for message passing layer

     procedure, pass(this) :: update_message => update_message_kipf
     !! Update the message
     procedure, pass(this) :: backward_message => backward_message_kipf
     !! Backward pass for the message phase

     procedure, pass(this) :: update_readout => update_readout_kipf
     !! Update the readout
     procedure, pass(this) :: backward_readout => backward_readout_kipf
     !! Backward pass for the readout phase
  end type kipf_msgpass_layer_type

  ! Interface for setting up the MPNN layer
  !-----------------------------------------------------------------------------
  ! interface kipf_msgpass_layer_type
  !    !! Interface for setting up the MPNN layer
  !    module function layer_setup( &
  !         num_features, num_time_steps, num_outputs, batch_size, &
  !         verbose &
  !    ) result(layer)
  !      !! Set up the MPNN layer
  !      !!! MAKE THESE ASSUMED RANK
  !      integer, dimension(2), intent(in) :: num_features
  !      !! Number of features
  !      integer, intent(in) :: num_time_steps
  !      !! Number of time steps
  !      integer, intent(in) :: num_outputs
  !      !! Number of outputs
  !      integer, optional, intent(in) :: batch_size
  !      !! Batch size
  !      integer, optional, intent(in) :: verbose
  !      !! Verbosity level
  !      type(kipf_msgpass_layer_type) :: layer
  !      !! Instance of the message passing layer
  !    end function layer_setup
  ! end interface kipf_msgpass_layer_type

contains

  subroutine read_kipf(this, unit, verbose)
    !! Read the message passing layer
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    integer, intent(in) :: unit
    !! Unit to read from
    integer, optional, intent(in) :: verbose
    !! Verbosity level
  end subroutine read_kipf


  pure subroutine forward_rank(this, input)
    !! Forward pass for message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    real(real32), dimension(..), intent(in) :: input
    !! Input to the message passing layer

  end subroutine forward_rank

  pure subroutine backward_rank(this, input, gradient)
    !! Backward pass for message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    real(real32), dimension(..), intent(in) :: input
    !! Input to the message passing layer
    real(real32), dimension(..), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_rank



  pure subroutine update_message_kipf(this, input)
    !! Update the message
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input to the message passing layer

    ! Local variables
    integer :: s, v, e, t
    !! Batch index, vertex index, edge index, time step
    real(real32) :: c
    !! Normalisation constant for the message passing

    ! real(real32), dimension(:,:), allocatable :: xe


    do s = 1, this%batch_size
       this%vertex_features(0,s)%val = input(1,s)%val
       this%edge_features(0,s)%val = input(2,s)%val
    end do

    do t = 1, this%num_time_steps
       do concurrent (s = 1: this%batch_size)
          do v = 1, this%graph(s)%num_vertices
             !  allocate( xe(2 * this%num_vertex_features + this%num_edge_features, this%graph(s)%vertex(v)%degree) )
             do e = this%graph(s)%adj_ia(v), this%graph(s)%adj_ia(v+1) - 1

                ! this is from Chen 2021 et al
                ! xe(:,e-this%graph(s)%adj_ia(v)+1) = [ &
                !      ( this%vertex_features(t-1,s)%val(:,v) + this%vertex_features(t-1,s)%val(:,this%graph(s)%adj_ja(1,e)) ) / 2._real32, &
                !      abs( this%vertex_features(t-1,s)%val(:,v) - this%vertex_features(t-1,s)%val(:,this%graph(s)%adj_ja(1,e)) ) / 2._real32, &
                !      this%edge_features(t-1,s)%val(:,e) &
                ! ]

                ! xe(:,e-this%graph(s)%adj_ia(v)+1) = matmul( &
                !      this%edge_weight(t)%val(:,:), &
                !      xe(:,e-this%graph(s)%adj_ia(v)+1) &
                ! )



                if( this%graph(s)%adj_ja(2,e) .eq. 0 )then
                   c = 1._real32
                else
                   c = this%graph(s)%edge_weights(this%graph(s)%adj_ja(2,e))
                end if
                ! fix this for lower memory case, where we don't store the vertices as derived types
                c = c * ( &
                     ( this%graph(s)%vertex(v)%degree + 1 ) * &
                     ( &
                          this%graph(s)%vertex( &
                               this%graph(s)%adj_ja(1,e) &
                          )%degree + 1 &
                     ) &
                ) ** ( -0.5_real32 )
                this%message(t,s)%val(:,v) = &
                     this%message(t,s)%val(:,v) + &
                     c * [ &
                          this%vertex_features(t-1,s)%val( &
                               :, &
                               this%graph(s)%adj_ja(1,e) &
                          ) &
                     ]
             end do
             this%z(t,s)%val(:,v) = matmul( &
                  this%message(t,s)%val(:,v), &
                  this%weight(:,:,t) &
             )
          end do
          this%vertex_features(t,s)%val(:,:) = &
               this%transfer%activate( this%z(t,s)%val(:,:) )
       end do
    end do

  end subroutine update_message_kipf


  pure subroutine update_readout_kipf(this)
    !! Update the readout
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer

    ! Local variables
    integer :: s, v
    !! Loop indices


    do s = 1, this%batch_size
       this%output(1,s)%val = this%vertex_features(this%num_time_steps,s)%val
       this%output(2,s)%val = this%edge_features(this%num_time_steps,s)%val
    end do

  end subroutine update_readout_kipf


  pure subroutine backward_message_kipf(this, input, gradient)
    !! Backward pass for the message phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: input
    !! Input data (i.e. vertex and edge features)
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_message_kipf



  pure subroutine backward_readout_kipf(this, gradient)
    !! Backward pass for the readout phase
    implicit none

    ! Arguments
    class(kipf_msgpass_layer_type), intent(inout) :: this
    !! Instance of the message passing layer
    class(array_type), dimension(:,:), intent(in) :: gradient
    !! Gradient of the loss with respect to the output of the layer

  end subroutine backward_readout_kipf




end module athena__kipf_msgpass_layer
