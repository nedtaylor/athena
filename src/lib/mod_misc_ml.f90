module misc_ml
  use constants, only: real12
  implicit none


  private

  public :: set_padding, pad_data

  public :: step_decay
  public :: reduce_lr_on_plateau



contains

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
  subroutine set_padding(pad, kernel_size, padding_method, verbose)
    use misc, only: to_lower
    implicit none
    integer, intent(out) :: pad
    integer, intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    integer, optional, intent(in) :: verbose
    
    integer :: t_verbose = 0


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose)) t_verbose = verbose

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
       if(t_verbose.gt.0) write(*,*) "Padding type: 'valid' (no padding)"
       pad = 0
       return
    case("same")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'same' (pad with zeros)"
    case("circular")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'same' (circular padding)"
    case("full")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'full' (all possible positions)"
       pad = kernel_size - 1
       return
    case("reflection")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'reflection' (reflect on boundary)"
    case("replication")
       if(t_verbose.gt.0) write(*,*) "Padding type: 'replication' (reflect after boundary)"
    case default
       stop "ERROR: padding type '"//padding_method//"' not known"
    end select

    pad = get_padding_half(kernel_size)

  end subroutine set_padding
!!!########################################################################


!!!########################################################################
!!! pad dataset
!!!########################################################################
  subroutine pad_data(data, data_padded, &
       kernel_size, padding_method, &
       sample_dim, channel_dim, constant)
    implicit none
    !real(real12), allocatable, dimension(:,:), intent(inout) :: data
    real(real12), dimension(..), intent(in) :: data
    real(real12), allocatable, dimension(..), intent(out) :: data_padded
    integer, dimension(..), intent(in) :: kernel_size
    character(*), intent(inout) :: padding_method
    real(real12), optional, intent(in) :: constant

    integer, optional, intent(in) :: sample_dim, channel_dim
    
    integer :: i, j, idim
    integer :: num_samples, num_channels, ndim, ndata_dim
    integer :: t_sample_dim = 0, t_channel_dim = 0
    real(real12) :: t_constant = 0._real12
    integer, dimension(2) :: bound_store
    integer, allocatable, dimension(:) :: padding
    integer, allocatable, dimension(:,:) :: trgt_bound, dest_bound
    integer, allocatable, dimension(:,:) :: tmp_trgt_bound, tmp_dest_bound
    !real(real12), allocatable, dimension(:,:) :: data_copy


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(constant)) t_constant = constant
    if(present(sample_dim)) t_sample_dim = sample_dim
    if(present(channel_dim)) t_channel_dim = channel_dim

    ndim = rank(data)
    if(ndim.ne.rank(data_padded)) then
       stop "ERROR: data and data_padded are not the same rank"
    end if
    ndata_dim = ndim
    if(t_sample_dim.gt.0)  ndata_dim = ndata_dim - 1
    if(t_channel_dim.gt.0) ndata_dim = ndata_dim - 1

    select rank(data)
    rank(1)
       if(t_sample_dim.gt.0) num_samples = size(data,t_sample_dim)
       if(t_channel_dim.gt.0) num_channels = size(data,t_channel_dim)
    rank(2)
       if(t_sample_dim.gt.0) num_samples = size(data,t_sample_dim)
       if(t_channel_dim.gt.0) num_channels = size(data,t_channel_dim)
    rank(3)
       if(t_sample_dim.gt.0) num_samples = size(data,t_sample_dim)
       if(t_channel_dim.gt.0) num_channels = size(data,t_channel_dim)
    rank(4)
       if(t_sample_dim.gt.0) num_samples = size(data,t_sample_dim)
       if(t_channel_dim.gt.0) num_channels = size(data,t_channel_dim)
    rank(5)
       if(t_sample_dim.gt.0) num_samples = size(data,t_sample_dim)
       if(t_channel_dim.gt.0) num_channels = size(data,t_channel_dim)
    rank default
       stop "ERROR: cannot handle data with this rank"
    end select
    

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
       allocate(padding(ndata_dim))
       do i=1,ndata_dim
          call set_padding(padding(i), kernel_size, padding_method, verbose=0)
       end do
    rank(1)
       if(size(kernel_size).eq.1.and.ndata_dim.gt.1)then
          allocate(padding(ndata_dim))
          do i=1,ndata_dim
             call set_padding(padding(i), kernel_size(1), padding_method, verbose=0)
          end do
       else
          if(t_sample_dim.eq.0.and.t_channel_dim.eq.0.and.&
               size(kernel_size).ne.ndim)then
             write(*,*) "kernel dimension:", size(kernel_size)
             write(*,*) "data rank:", ndim
             stop "ERROR: length of kernel_size not equal to rank of data"
          elseif(t_sample_dim.gt.0.and.t_channel_dim.gt.0.and.&
               size(kernel_size).ne.ndim-2)then
             write(*,*) "kernel dimension:", size(kernel_size)
             write(*,*) "data rank:", ndim
             stop "ERROR: length of kernel_size not equal to rank of data-2"
          elseif((t_sample_dim.gt.0.or.t_channel_dim.gt.0).and.&
               size(kernel_size).ne.ndim-1)then
             write(*,*) "kernel dimension:", size(kernel_size)
             write(*,*) "data rank:", ndim
             stop "ERROR: length of kernel_size not equal to rank of data-1"
          else
             allocate(padding(size(kernel_size)))
          end if
          do i=1,size(kernel_size)
             call set_padding(padding(i), kernel_size(i), padding_method, verbose=0)
          end do
       end if
    end select


!!!-----------------------------------------------------------------------------
!!! allocate data set
!!! ... if appropriate, add padding
!!!-----------------------------------------------------------------------------
    select case(padding_method)
    case ("same")
    case("full")
    case("zero")
    case default
       if(abs(t_constant).gt.1.E-8) &
            write(*,*) "WARNING: constant is ignored for selected padding method"
    end select

    
    allocate(dest_bound(2,ndim))
    allocate(trgt_bound(2,ndim))
    i = 0
    do idim=1,ndim
       trgt_bound(:,idim) = [ lbound(data,dim=idim), ubound(data,dim=idim) ]
       dest_bound(:,idim) = trgt_bound(:,idim)
       if(idim.eq.t_sample_dim.or.idim.eq.t_channel_dim) cycle
       i = i + 1
       dest_bound(:,idim) = dest_bound(:,idim) + [ -padding(i), padding(i) ]
    end do

    select rank(data_padded)
    rank(1)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1)), source = t_constant)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(1)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1))
       end select
    rank(2)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1), &
            dest_bound(1,2):dest_bound(2,2)), source = t_constant)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(2)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1),trgt_bound(1,2):trgt_bound(2,2)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1),trgt_bound(1,2):trgt_bound(2,2))
       end select
    rank(3)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3)), source = t_constant)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(3)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3))
       end select
    rank(4)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3),&
            dest_bound(1,4):dest_bound(2,4)), source = t_constant)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(4)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4))
       end select
    rank(5)
       allocate(data_padded(&
            dest_bound(1,1):dest_bound(2,1),&
            dest_bound(1,2):dest_bound(2,2),&
            dest_bound(1,3):dest_bound(2,3),&
            dest_bound(1,4):dest_bound(2,4),&
            dest_bound(1,5):dest_bound(2,5)), source = t_constant)
       !! copy input data
       !!-----------------------------------------------------------------------
       select rank(data)
       rank(5)
          data_padded( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5)) = &
               data( &
               trgt_bound(1,1):trgt_bound(2,1), &
               trgt_bound(1,2):trgt_bound(2,2), &
               trgt_bound(1,3):trgt_bound(2,3), &
               trgt_bound(1,4):trgt_bound(2,4), &
               trgt_bound(1,5):trgt_bound(2,5))
       end select
    end select


!!!-----------------------------------------------------------------------------
!!! return if constant -- or no -- padding
!!!-----------------------------------------------------------------------------
    select case(padding_method)
    case ("same")
       return
    case("full")
       return
    case("zero")
       return
    case("valid")
       return
    end select


!!!-----------------------------------------------------------------------------
!!! insert padding
!!!-----------------------------------------------------------------------------
    i = 0
    do idim=1,ndim
       if(idim.eq.t_sample_dim.or.idim.eq.t_channel_dim) cycle
       i = i + 1
       tmp_dest_bound = dest_bound
       tmp_trgt_bound = dest_bound
       tmp_dest_bound(:,idim) = [ dest_bound(1,idim), trgt_bound(1,idim) - 1 ]
       select case(padding_method)
       case ("circular")
          tmp_trgt_bound(:,idim) = [ trgt_bound(2,idim) - padding(i) + 1, trgt_bound(2,idim) ]
       case("reflection")
          tmp_trgt_bound(:,idim) = [ trgt_bound(1,idim) + 1, trgt_bound(1,idim) + padding(i) ]
       case("replication")
          tmp_trgt_bound(:,idim) = [ trgt_bound(1,idim), trgt_bound(1,idim) + padding(i) - 1 ]
       end select
       do j = 1, 2
          select rank(data_padded)
          rank(1)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1)) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1))
          rank(2)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2) )
          rank(3)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3) )
          rank(4)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4) )
          rank(5)
             data_padded( &
                  tmp_dest_bound(1,1):tmp_dest_bound(2,1), &
                  tmp_dest_bound(1,2):tmp_dest_bound(2,2), &
                  tmp_dest_bound(1,3):tmp_dest_bound(2,3), &
                  tmp_dest_bound(1,4):tmp_dest_bound(2,4), &
                  tmp_dest_bound(1,5):tmp_dest_bound(2,5) ) = &
                  data_padded( &
                  tmp_trgt_bound(1,1):tmp_trgt_bound(2,1), &
                  tmp_trgt_bound(1,2):tmp_trgt_bound(2,2), &
                  tmp_trgt_bound(1,3):tmp_trgt_bound(2,3), &
                  tmp_trgt_bound(1,4):tmp_trgt_bound(2,4), &
                  tmp_trgt_bound(1,5):tmp_trgt_bound(2,5) )
          end select

          if(j.eq.2) exit
          bound_store(:) = tmp_dest_bound(:,idim)
          select case(padding_method)
          case ("circular")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + padding(i)
             tmp_trgt_bound(:,idim) = bound_store(:) + padding(i)
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ lbound(data,idim), lbound(data,idim) + padding(i) - 1 ]
          case("reflection")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + size(data,idim) - 1
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim) - 1
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ ubound(data,idim) - padding(i), ubound(data,idim) - 1 ]
          case("replication")
             tmp_dest_bound(:,idim) = tmp_trgt_bound(:,idim) + size(data,idim)
             tmp_trgt_bound(:,idim) = bound_store(:) + size(data,idim)
             !tmp_dest_bound(:,idim) = [ ubound(data,idim) + 1, ubound(data_copy,idim) ]
             !tmp_trgt_bound(1,idim) = [ ubound(data,idim) - padding(i) + 1, ubound(data,idim) ]
          end select
       end do
    end do

  end subroutine pad_data
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
