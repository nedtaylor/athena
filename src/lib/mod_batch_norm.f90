module batch_norm
  use constants, only: real12
  implicit none


  private

  public :: initialise
  public :: forward
  public :: backward
  public :: update


contains

!!!########################################################################
!!! 
!!!########################################################################
    subroutine initialise(mean, variance, gamma, beta)
        real(real12), dimension(:), intent(out) :: mean, variance
        real(real12), dimension(:), optional, intent(out) :: gamma, beta

        mean = 0._real12
        variance = 0._real12

        if(present(gamma)) gamma = 1._real12
        if(present(beta)) beta = 0._real12
        

      end subroutine initialise
!!!########################################################################


!!!########################################################################
!!! batch normalisation
!!!########################################################################
!!! https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
!!! https://d2l.ai/chapter_convolutional-modern/batch-norm.html
!!! https://stackoverflow.com/questions/45799926/why-batch-normalization-over-channels-only-in-cnn
!!! https://stackoverflow.com/questions/65613694/calculation-of-mean-and-variance-in-batch-normalization-in-convolutional-neural
  subroutine forward(input, mean, variance, &
       momentum, gamma, beta, epsilon, input_norm)
    implicit none
    real(real12), dimension(:,:,:,:), intent(inout) :: input
    real(real12), dimension(:), intent(inout) :: mean, variance
    real(real12), optional, intent(in) :: momentum !! for exponential moving average, inference moving average, for single sample training
    real(real12), dimension(:), optional, intent(in) :: gamma, beta
    real(real12), optional, intent(in) :: epsilon
    real(real12), dimension(:,:,:,:), optional, intent(out) :: input_norm

    integer :: k, n, input_size
    integer :: num_samples, num_channels, num_features
    real(real12) :: t_epsilon
    real(real12), dimension(size(mean,1)) :: t_mean, t_variance
    real(real12), dimension(size(mean,1)) :: inv_std_dev


    if(present(epsilon))then
       t_epsilon = epsilon
    else
       t_epsilon = 1.E-8_real12
    end if



    num_samples = size(input,dim=4)
    num_channels = size(input,dim=3)
    input_size = size(input,dim=1)
    num_features = size(input(:,:,1,1))

    t_mean = 0._real12
    t_variance = 0._real12
    !! apply batch normalization
    do k=1,num_channels

       !! separate mean and variance for each channel       
       t_mean(k) = sum(input(:,:,k,:))/real(num_samples,real12)
       t_variance(k) = sum(input(:,:,k,n) - t_mean(k))/&
            real(num_samples,real12)
       inv_std_dev = 1._real12 / sqrt(t_variance(k) + t_epsilon)
       
       do n = 1, num_samples
          input(:,:,k,n) = (input(:,:,k,n) - t_mean(k)) * inv_std_dev(k)
       end do

       if(present(input_norm))then
          input_norm(:,:,k,:) = input(:,:,k,:)
       end if

       if(present(gamma).and.present(beta))then
          input(:,:,k,:) = gamma(k) * input(:,:,k,:) + beta(k)
       end if
    end do

    if(present(momentum))then
       mean = momentum * mean + (1._real12 - momentum) * t_mean
       variance = momentum * variance + (1._real12 - momentum) * t_variance
    else
       mean = t_mean
       variance = t_variance
    end if


  end subroutine forward
!!!########################################################################


!!!########################################################################
!!! 
!!!########################################################################
  subroutine backward(input_norm, output_gradients, &
       input_gradients, gradients_gamma, gradients_beta, &
       variance, epsilon)
    real(real12), dimension(:,:,:,:), intent(in) :: output_gradients
    real(real12), dimension(:,:,:,:), intent(in) :: input_norm
    real(real12), dimension(:), intent(in) :: variance
    real(real12), dimension(:,:,:,:), intent(out) :: input_gradients
    real(real12), dimension(:), intent(out) :: gradients_gamma, gradients_beta
    real(real12), optional, intent(in) :: epsilon

    integer :: k, n, num_channels, num_samples, output_size
    real(real12) :: inv_std_dev, inv_variance !, grads_input_norm
    real(real12) :: sum_grads
    real(real12) :: t_epsilon


    if(present(epsilon))then
       t_epsilon = epsilon
    else
       t_epsilon = 1.E-8_real12
    end if


    num_samples = size(output_gradients, dim=4)
    num_channels = size(output_gradients, dim=3)
    output_size = size(output_gradients, dim =1)
    input_gradients = 0._real12
    gradients_gamma = 0._real12
    gradients_beta = 0._real12

    do n = 1, num_samples
       do k = 1, num_channels
          inv_std_dev = 1._real12 / sqrt(variance(k) + t_epsilon)
          inv_variance = 1._real12 / (variance(k) + t_epsilon)
          sum_grads = sum(output_gradients(:,:,k,n))

          !grads_input_norm = &
          !     inv_std_dev * (output_gradients(:, :, k, n) - &
          !     (sum_grads * inv_variance))
          input_gradients(:, :, k, n) = &
               inv_std_dev * &
               ( output_gradients(:, :, k, n) * output_size - &
               sum(output_gradients(:,:,k,n) * input_norm(:,:,k,n)) - &
               input_norm(:, :, k, n) * sum_grads * inv_variance )
          gradients_gamma(k) = &
               gradients_gamma(k) + &
               sum(output_gradients(:, :, k, n) * input_norm(:, :, k, n))
          gradients_beta(k) = &
               gradients_beta(k) + &
               sum(output_gradients(:, :, k, n))
       end do
    end do

  end subroutine backward
!!!########################################################################


!!!########################################################################
!!! 
!!!########################################################################
    subroutine update(learning_rate, gamma, beta, &
         gradients_gamma, gradients_beta)
        real(real12), dimension(:), intent(inout) :: gamma, beta
        real(real12), dimension(:), intent(in) :: gradients_gamma, gradients_beta
        real(real12), intent(in) :: learning_rate

        gamma = gamma - learning_rate * gradients_gamma
        beta = beta - learning_rate * gradients_beta

    end subroutine update
!!!########################################################################

end module batch_norm
