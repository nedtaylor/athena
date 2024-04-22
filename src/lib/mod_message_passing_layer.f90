!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! module contains implementation of a 1D convolutional layer
!!!#############################################################################
module mpnn_module
  use constants, only: real12
  use custom_types, only: initialiser_type, graph_type
  implicit none
  

  private

  public :: mpnn_type

  type :: mpnn_type
     integer :: num_vertices
     integer :: num_time_steps
     !! hidden features has dimensions (vertex, time step)
     type(feature_type), dimension(:,:), allocatable :: hidden
!     type(feature_type), dimension(:,:), allocatable :: hidden_vertex
!     type(feature_type), dimension(:,:), allocatable :: hidden_edge
     !! message has dimensions (vertex, time step)
     type(feature_type), dimension(:,:), allocatable :: message
   contains
     procedure(message_update), pass(this), pointer :: message_update => null()
     procedure(state_update), pass(this), pointer :: state_update => null()
     procedure, pass(this) :: readout
     procedure, pass(this) :: forward
  end type mpnn_type

  type :: feature_type
     real(real12), dimension(:), allocatable :: feature
  end type feature_type

  
!!!-----------------------------------------------------------------------------
!!! interface for layer set up
!!!-----------------------------------------------------------------------------
  ! interface mpnn_type
  !    module function layer_setup( &
  !         input_shape, batch_size, &
  !         num_filters, kernel_size, stride, padding, &
  !         activation_function, activation_scale, &
  !         kernel_initialiser, bias_initialiser, &
  !         calc_input_gradients) result(layer)
  !      integer, dimension(:), optional, intent(in) :: input_shape
  !      integer, optional, intent(in) :: batch_size
  !      integer, optional, intent(in) :: num_filters
  !      integer, dimension(..), optional, intent(in) :: kernel_size
  !      integer, dimension(..), optional, intent(in) :: stride
  !      real(real12), optional, intent(in) :: activation_scale
  !      character(*), optional, intent(in) :: activation_function, &
  !           kernel_initialiser, bias_initialiser, padding
  !      logical, optional, intent(in) :: calc_input_gradients
  !      type(conv1d_layer_type) :: layer
  !    end function layer_setup
  ! end interface mpnn_type


  abstract interface
     pure subroutine message_update(this, graph)
       implicit none
       class(mpnn_type), intent(in) :: this
       type(graph_type), intent(in) :: graph
     end subroutine message_update

     pure subroutine state_update(this, graph)
       implicit none
       class(mpnn_type), intent(in) :: this
       type(graph_type), intent(in) :: graph
     end subroutine state_update
  end interface
  


contains

!!!#############################################################################
!!! message function
!!!#############################################################################
  pure subroutine convolutional_message_update(this, graph, time_step)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), intent(in) :: graph
    integer, intent(in) :: time_step

    integer :: i, j
    integer :: num_features

    !! assume all hidden vertices for one time_step have the same number of features
    num_features = size(this%hidden(1,time_step)%feature)
    do i = 1, graph%num_vertices
       allocate(this%message(i,time_step+1)%feature(num_features), source=0._real12)
       do j = 1, graph%num_vertices
          this%message(i,time_step+1)%feature = &
               this%message(i,time_step+1)%feature + &
               [ this%hidden(j,time_step), graph%edge(i,j)%feature ]
               ![ graph%vertex(j)%feature, graph%edge(i,j)%feature ]
               ! at time step 1, set hidden to graph vertex features
       end do
    end do

  end subroutine convolutional_message_update

  pure subroutine convolutional_state_update(this, graph, time_step)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), intent(in) :: graph
    integer, intent(in) :: time_step

    integer :: i

    do i = 1, graph%num_vertices
       this%hidden(:,i,time_step+1) = sigmoid( &
            matmul( this%update_matrix(graph%get_degree(i),time_step), &
                    this%message(i,time_step+1) ) )
    end do

  end subroutine convolutional_state_update

  pure function convolutional_readout(this) result(output)
    implicit none
    class(mpnn_type), intent(in) :: this

    real(real12), dimension(:), allocatable :: output

    integer :: i, t

    allocate(output(size(this%hidden(1,num_time_steps)%feature)), source=0._real12)

    do i = 1, this%num_vertices
       do t = 1, this%num_time_steps
          output = output + softmax( matmul( &
               this%readout_matrix(:,:,t), this%hidden(i,t)%feature ) )
       end do
    end do

  end function convolutional_readout

!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward(this, graph)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), intent(in) :: graph

    this%hidden(:,1)%feature = graph%vertex(:)%feature

    do t = 1, this%num_time_steps
       call this%message(graph)
       call this%update(graph)
    end do

    call this%readout()

  end subroutine forward
!!!#############################################################################


!!!#############################################################################
!!! backpropagation
!!!#############################################################################
  pure subroutine backward(this, graph, gradient)
    implicit none
    class(mpnn_type), intent(inout) :: this
    type(graph_type), intent(in) :: graph

    !df/dv_c = h(M_c) * df/dM_y

    ! M_y = sum_c v_c * h(M_c)     message for output y
    ! h()                          hidden function

    !this%dw(:,this%num_time_steps)%feature = this%derivative(this%hidden(:,this%num_time_steps)%feature) * this%readout_matrix(:,:,this%num_time_steps)
    this%v(:,this%num_time_steps)%feature = gradient
    this%dw(:,this%num_time_steps) = this%hidden(:,this%num_time_steps) * this%v(:,this%num_time_steps)
    do t = this%num_time_steps-1, 1, -1
       this%v(:,t)%feature = this%derivative(this%hidden(:,t)%feature) * sum( this%weight * this%v(:,t+1)%feature )
       this%dw = this%hidden(:,t) * this%v(:,t)
    end do



  end subroutine backward
!!!#############################################################################

end module mpnn_module
!!!#############################################################################
