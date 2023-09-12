module misc_ml
  use constants, only: real12
  implicit none


  private

  public :: set_padding

  public :: step_decay
  public :: reduce_lr_on_plateau

  public :: drop_block, generate_bernoulli_mask


contains

!!!########################################################################
!!! DropBlock method for dropping random blocks of data from an image
!!!########################################################################
!!! https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
!!! https://pub.towardsai.net/dropblock-a-new-regularization-technique-e926bbc74adb
!!! input = input data
!!!         ... channels are provided independently
!!!         ... this tries to prevent the network from relying too ...
!!!         ... heavily one one set of activations
!!! keep_prob   = probability of keeping a unit, as in traditional dropout
!!!               ... (default = 0.75-0.95)
!!! block_size  = width of block (default = 5)
!!! gamma       = how many activation units to drop
  subroutine drop_block(input, mask, block_size)
    implicit none
    real(real12), dimension(:,:), intent(inout) :: input
    logical, dimension(:,:), intent(in) :: mask
    integer, intent(in) :: block_size

    integer :: i, j, x, y, start_idx, end_idx, mask_size

    mask_size = size(mask, dim=1)
    start_idx = -(block_size - 1)/2 !centre should be zero
    end_idx = (block_size -1)/2 + (1 - mod(block_size,2)) !centre should be zero

    ! gamma = (1 - keep_prob)/block_size**2 * input_size**2/(input_size - block_size + 1)**2

    do j = 1, mask_size
       do i = 1, mask_size
          if (.not.mask(i, j))then
             do x=start_idx,end_idx,1
                do y=start_idx,end_idx,1
                   input(i - start_idx + x, j - start_idx + y) = 0._real12
                end do
             end do
          endif
       end do
    end do

    input = input * size(mask,dim=1) * size(mask,dim=2) / count(mask)

  end subroutine drop_block
!!!########################################################################


!!!########################################################################
!!! 
!!!########################################################################
  subroutine generate_bernoulli_mask(mask, gamma, seed)
    implicit none
    logical, dimension(:,:), intent(out) :: mask
    real, intent(in) :: gamma
    integer, optional, intent(in) :: seed
    real(real12), allocatable, dimension(:,:) :: mask_real
    integer :: i, j

    !! IF seed GIVEN, INITIALISE
    ! assume random number already seeded and don't need to again
    !call random_seed()  ! Initialize random number generator
    allocate(mask_real(size(mask,1), size(mask,2)))
    call random_number(mask_real)  ! Generate random values in [0,1)

    !! Apply threshold to create binary mask
    do j = 1, size(mask, dim=2)
       do i = 1, size(mask, dim=1)
          if(mask_real(i, j).gt.gamma)then
             mask(i, j) = .false. !0 = drop
          else
             mask(i, j) = .true.  !1 = keep
          end if
       end do
    end do
    
  end subroutine generate_bernoulli_mask
!!!########################################################################


!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  pure function get_padding_half(width) result(output)
    implicit none
    integer, intent(in) :: width
    integer :: output
    
    output = ( width - (1 - mod(width,2)) - 1 ) / 2
  end function get_padding_half
!!!########################################################################


!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  subroutine set_padding(pad, kernel_size, padding_method)
    use misc, only: to_lower
    implicit none
    integer, intent(out) :: pad
    integer, intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    

    !!---------------------------------------------------------------------
    !! padding method options
    !!---------------------------------------------------------------------
    !! none  = alt. name for 'valid'
    !! zero  = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. padding added = (kernel_size - 1)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding added = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
100 select case(to_lower(trim(padding_method)))
    case("none")
       padding_method = "valid"
       goto 100
    case("zero")
       padding_method = "same"
       goto 100
    case("half")
       padding_method = "same"
       goto 100
    case("symmetric")
       padding_method = "replication"
       goto 100
    case("valid")
       write(6,*) "Padding type: 'valid' (no padding)"
       pad = 0
       return
    case("same")
       write(6,*) "Padding type: 'same' (pad with zeros)"
    case("circular")
       write(6,*) "Padding type: 'same' (circular padding)"
    case("full")
       write(6,*) "Padding type: 'full' (all possible positions)"
       pad = kernel_size - 1
       return
    case("reflection")
       write(6,*) "Padding type: 'reflection' (reflect on boundary)"
    case("replication")
       write(6,*) "Padding type: 'replication' (reflect after boundary)"
    case default
       stop "ERROR: padding type '"//padding_method//"' not known"
    end select

    pad = get_padding_half(kernel_size)

  end subroutine set_padding
!!!########################################################################


!!!########################################################################
!!! return width of padding from kernel/filter size
!!!########################################################################
  subroutine pad_1Ddata(data, kernel_size, padding_method, dim, constant)
    use misc, only: to_lower
    implicit none
    real(real12), allocatable, dimension(:,:,:), intent(inout) :: data
    integer, dimension(..) :: intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    real(real12), optional, intent(in) :: constant

    integer, optional, intent(in) :: dim
    
    integer :: i, num_samples
    integer :: t_dim = 0
    real(real12) :: t_constant = 0._real12
    integer, allocatable, dimension(:) :: padding
    integer, allocatable, dimension(:,:) :: data_shape


    if(present(constant)) t_constant = constant
    if(present(dim)) t_dim = dim
    if(t_dim.gt.0) num_samples = size(data,dim=t_dim)

!!! NEEDS TO ALSO HANDLE CHANNEL DIMENSION
!!! MAYBE MAKE THIS A FUNCTION THAT OUTPUTS A PADDED VERSION OF THE DATA?
!!! THEN CYCLE OVER CHANNELS AS A USER?


!!!-----------------------------------------------------------------------------
!!! handle padding type name
!!!-----------------------------------------------------------------------------
    !! none  = alt. name for 'valid'
    !! zero  = alt. name for 'same'
    !! symmetric = alt.name for 'replication'
    !! valid = no padding
    !! same  = maintain spatial dimensions
    !!         ... (i.e. padding added = (kernel_size - 1)/2)
    !!         ... defaults to zeros in the padding
    !! full  = enough padding for filter to slide over every possible position
    !!         ... (i.e. padding added = (kernel_size - 1)
    !! circular = maintain spatial dimensions
    !!            ... wraps data around for padding (periodic)
    !! reflection = maintains spatial dimensions
    !!              ... reflect data (about boundary index)
    !! replication = maintains spatial dimensions
    !!               ... reflect data (boundary included)
    select rank(kernel_size)
    rank(0)
       allocate(padding(1))
       call set_padding(padding(1), kernel_size, t_padding_method)
    rank(1)
       if(t_dim.eq.0.and.size(kernel_size).ne.rank(data))then
          stop "ERROR: length of kernel_size not equal to rank of data"
       elseif(t_dim.gt.0.and.size(kernel_size).ne.rank(data)-1)then
          stop "ERROR: length of kernel_size not equal to rank of data-1"
       else
          allocate(padding(size(kernel_size))
       end if
       do i=1,size(kernel_size)
          call set_padding(padding(i), kernel_size(i), t_padding_method)
       end do
    end select

!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    if(padding.eq.0) return

    allocate(data_shape(2,rank(data)))
    data_shape = shape(data)
    do i=1,size(padding)
       data_shape(1,i) = lbound(data,dim=i) - padding
       data_shape(2,i) = ubound(data,dim=i) + padding
    end do

    allocate(data_copy(&
         data_shape(1,1):data_shape(2,1),&
         data_shape(1,2):data_shape(2,2)), source=0._real12)

    !! initialise padding for constant padding types
    !!-----------------------------------------------------------------------
    select case(t_padding_method)
    case ("same")
       data_copy = t_constant
    case("full")
       data_copy = t_constant
    end select


!!!-----------------------------------------------------------------------------
!!! read in dataset
!!!-----------------------------------------------------------------------------
    data_copy(&
         lbound(data,1):ubound(data,1),&
         lbound(data,2):ubound(data,2)) = &
         data(&
         lbound(data,1):ubound(data,1),&
         lbound(data,2):ubound(data,2))
    if(t_dim.gt.0)then
       select case(t_padding_method)
       case ("circular")
          data_copy( &
               lbound(data_copy,1):lbound(data,1) - 1, &
               lbound(data,2):ubound(data,2), &
               :) = &
               data( &
               ubound(data,1) - padding(1) + 1:ubound(data,1), &
               lbound(data,2):ubound(data,2), &
               :)
          data_copy( &
               ubound(data,1) + 1:ubound(data_copy,1), &
               lbound(data,2):ubound(data,2), &
               :) = &
               data(&
               lbound(data,1):lbound(data,1) + padding(1) - 1, &
               lbound(data,2):ubound(data,2), &
               :)

          data_copy( &
               :, &
               lbound(data_copy,2):lbound(data,2) - 1, &
               :) = &
               data_copy( &
               :, ubound(data,2) - padding(2) + 1:ubound(data,2), &
               :)
          data_copy( &
               :, &
               ubound(data,2) + 1:ubound(data,2) + padding(2), &
               :) = &
               data_copy( &
               :, &
               lbound(data,2):lbound(data,2) + padding(2) - 1, &
               :)
       case("reflection")
          data_copy( &
               lbound(data,1) - 1 : lbound(data_copy,1) : -1, &
               lbound(data,2):ubound(data,2), &
               :) = &
               data(&
               lbound(data,1) + 1 : lbound(data,1) + padding(1) : 1, &
               lbound(data,2):ubound(data,2), &
               :)
          data_copy( &
               ubound(data,1) + 1 : ubound(data,1) + padding(1) : 1, &
               lbound(data,2) : ubound(data,2), &
               :) = &
               data(&
               ubound(data,1) - 1 : ubound(data,1) - padding(1) : -1, &
               lbound(data,2) : ubound(data,2), &
               :)

          data_copy( &
               :, &
               lbound(data,2) - 1 : -padding(D) + 1 : -1, &
               :) = &
               data_copy(&
               :, &
               lbound(data,2) + 1 : lbound(data,2) + padding(2) : 1, &
               :)
          data_copy( &
               :, &
               ubound(data,2) + 1 : ubound(data,2) + padding(2) : 1, &
               :) = &
               data_copy( &
               :, &
               ubound(data,2) - 1 : ubound(data,2) - padding(2) : -1, &
               :)
       case("replication")
          data_copy( &
               lbound(data,1) - 1 : -padding(1) + 1 : -1, &
               lbound(data,2) : ubound(data,2), &
               :) = &
               data( &
               lbound(data,1) : lbound(data,1) + padding(1) - 1 : 1, &
               lbound(data,2) : ubound(data,2), &
               :)
          data_copy(&
               ubound(data,1) + 1 : ubound(data,1) + padding(1) : 1, &
               lbound(data,2) : ubound(data,2), &
               :) = &
               data( &
               ubound(data,1) : ubound(data,1) - padding(1) + 1 : -1, &
               lbound(data,2) : ubound(data,D), &
               :)

          data_copy( &
               :, &
               lbound(data,2) - 1 : -padding(2) + 1 : -1, &
               :) = &
               data_copy( &
               :, &
               lbound(data,2) : lbound(data,2) + padding(2) - 1 : 1, &
               :)
          data_copy( &
               :, &
               ubound(data,2) + 1 : ubound(data,2) + padding(2) : 1, &
               :) = &
               data_copy( &
               :, &
               ubound(data,2) : ubound(data,2) - padding(2) + 1 :-1, &
               :)
       end select
    end if


    data = data_copy

  end subroutine set_padding
!!!########################################################################


!!!########################################################################
!!! adaptive learning rate
!!! method: step decay
!!!########################################################################
  subroutine step_decay(learning_rate, epoch, decay_rate, decay_steps)
    implicit none
    integer, intent(in) :: epoch
    integer, intent(in) :: decay_steps
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: decay_rate

    !! calculate new learning rate
    learning_rate = learning_rate * &
         decay_rate**((epoch - 1._real12) / decay_steps)

  end subroutine step_decay
!!!########################################################################


!!!########################################################################
!!! adaptive learning rate
!!! method: reduce learning rate on plateau
!!!########################################################################
  subroutine reduce_lr_on_plateau(learning_rate, &
       metric_value, patience, factor, min_learning_rate, & 
       best_metric_value, wait)
    implicit none
    integer, intent(in) :: patience
    integer, intent(inout) :: wait
    real(real12), intent(inout) :: learning_rate
    real(real12), intent(in) :: metric_value
    real(real12), intent(in) :: factor
    real(real12), intent(in) :: min_learning_rate
    real(real12), intent(inout) :: best_metric_value

    !! check if the metric value has improved
    if (metric_value.lt.best_metric_value) then
       best_metric_value = metric_value
       wait = 0
    else
       wait = wait + 1
       if (wait.ge.patience) then
          learning_rate = learning_rate * factor
          if (learning_rate.lt.min_learning_rate) then
             learning_rate = min_learning_rate
          endif
          wait = 0
       endif
    endif

  end subroutine reduce_lr_on_plateau
!!!########################################################################

end module misc_ml
