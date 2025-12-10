program rnn_timeseries_example
  !! Recurrent Neural Network for time series prediction
  !!
  !! This example demonstrates using RNNs to learn temporal patterns and
  !! make predictions on sequential data, specifically learning to predict
  !! sine wave patterns.
  !!
  !! ## Recurrent Neural Networks
  !!
  !! RNNs process sequences by maintaining a hidden state that captures
  !! information about previous inputs:
  !!
  !! $$\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$
  !! $$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$
  !!
  !! where:
  !! - \( \mathbf{x}_t \) is the input at time \( t \)
  !! - \( \mathbf{h}_t \) is the hidden state at time \( t \)
  !! - \( \mathbf{y}_t \) is the output at time \( t \)
  !! - \( \mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy} \) are weight matrices
  !!
  !! ## Time Series Prediction
  !!
  !! Given a sequence \( [x_1, x_2, ..., x_t] \), predict \( x_{t+1} \):
  !! $$\hat{x}_{t+1} = f_{\theta}(x_t, x_{t-1}, ..., x_{t-k+1})$$
  !!
  !! The RNN learns to capture temporal dependencies and patterns.
  !!
  !! ## Sine Wave Task
  !!
  !! Learn the pattern: \( x_t = \sin(\omega t + \phi) \)
  !! - Input: Sequence of past values \( [x_{t-k}, ..., x_{t-1}, x_t] \)
  !! - Output: Next value \( x_{t+1} \)
  !!
  !! ## Training
  !!
  !! Uses backpropagation through time (BPTT) to compute gradients:
  !! - Unroll the network across time steps
  !! - Compute loss at each step or final step
  !! - Backpropagate gradients through the temporal chain
  !!
  !! ## Applications
  !!
  !! - Stock price prediction
  !! - Weather forecasting
  !! - Signal processing
  !! - Natural language modeling
  use athena
  use coreutils, only: real32
  use constants_mnist, only: pi

  implicit none

  type(network_type), target :: network
  real(real32), dimension(1,1) :: x, y
  type(array_type) :: x_array(1,1), y_array(1,1)
  type(array_type), pointer :: loss, output_ptr, output(:,:)

  integer, parameter :: num_iterations = 3000
  integer, parameter :: num_time_steps = 10   ! length of input sequence

  integer :: seed_size
  integer, allocatable, dimension(:) :: seed

  integer :: i, n, t
  real(real32) :: r
  real(real32), dimension(num_time_steps) :: x_seq, y_seq


  !-----------------------------------------------------------------------------
  ! set random seed for reproducibility
  !-----------------------------------------------------------------------------
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 42
  call random_seed(put=seed)

  write(*,*) "Time Series Pattern Learning using Recurrent Neural Network"
  write(*,*) "============================================================"
  write(*,*) "Task: Learn sine wave pattern for function approximation"
  write(*,*)


  !-----------------------------------------------------------------------------
  ! create network with recurrent layer
  !-----------------------------------------------------------------------------
  write(*,*) "Building network with RNN layer..."
  call network%add(recurrent_layer_type( &
       input_size=1, &
       hidden_size=4, &
       activation="tanh" &

  ))
  call network%add(full_layer_type( &
       num_outputs=1, &
       activation="sigmoid" &
  ))

  call network%compile( &
       optimiser = sgd_optimiser_type(learning_rate=0.05_real32), &
       loss_method="mse", &
       metrics=["loss"], &
       verbose=1 &
  )
  call network%set_batch_size(1)
  write(*,*) "Network structure:"
  call network%print_summary()
  write(*,*)


  !-----------------------------------------------------------------------------
  ! generate toy example train data
  !-----------------------------------------------------------------------------
  do i = 1, num_time_steps
     call random_number(r)
     x_seq(i) = merge(1._real32, 0._real32, r .gt. 0.5_real32)
  end do

  y_seq(1) = x_seq(num_time_steps)
  do i = 2, num_time_steps
     y_seq(i) = x_seq(i-1)
  end do
  call x_array(1,1)%allocate(array_shape=[1,1])
  call y_array(1,1)%allocate(array_shape=[1,1])


  !-----------------------------------------------------------------------------
  ! train network
  !-----------------------------------------------------------------------------
  write(*,*) "Training network to learn sine wave pattern..."
  write(*,*) "Iteration, MSE"

  ! run training
  do n = 1, num_iterations
     call network%reset_state()   ! important for proper recurrence
     ! ---- Load into array_type wrappers ----
     do t = 1, num_time_steps
        x_array(1,1)%val = reshape([ x_seq(t) ], [1,1])   ! (features, batch)
        y_array(1,1)%val = reshape([ y_seq(t) ], [1,1])
        x_array(1,1)%is_temporary = .false.
        y_array(1,1)%is_temporary = .false.
        output => network%forward_eval(x_array)
        output_ptr => output(1,1)%duplicate_graph()
        if(t.eq.1)then
           loss => (output_ptr - y_array(1,1))**2
        else
           loss => loss + &
                (output_ptr - y_array(1,1))**2
        end if
     end do
     if(n.eq.1 .or. mod(n, 200) == 0) write(*,*) n, loss%val(1,1)
     call loss%grad_reverse()
     call network%update()
     loss => null()
     call network%reset_state()   ! important for proper recurrence
  end do
  call network%reset_state()   ! important for proper recurrence

  write(*,*)
  write(*,*) "Training complete!"
  write(*,*)


  !-----------------------------------------------------------------------------
  ! test the network
  !-----------------------------------------------------------------------------
  write(*,*) "Testing network on sample points:"
  write(*,*) "  i    Input      Expected   Predicted    Error"
  write(*,*) "----  ---------  ---------  ---------  ---------"
  do t = 1, num_time_steps
     x_array(1,1)%val = reshape([ x_seq(t) ], [1,1])   ! (features, batch)
     y_array = network%predict(x_array)
     write(*,'(I4,2X,F9.5,2X,F9.5,2X,F9.5,2X,F9.5)') t, x_seq(t), y_seq(t), &
          y_array(1,1)%val(1,1), abs(y_seq(t) - y_array(1,1)%val(1,1))
  end do


  !-----------------------------------------------------------------------------
  ! save network to file
  !-----------------------------------------------------------------------------
  write(*,*)
  write(*,*) "Saving network to 'rnn_timeseries_model.txt'..."
  call network%print(file='rnn_timeseries_model.txt')
  write(*,*) "Network saved"
  write(*,*)
  write(*,*) "RNN example complete - network learned sine wave pattern!"

end program rnn_timeseries_example
