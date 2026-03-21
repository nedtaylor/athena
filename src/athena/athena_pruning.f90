module athena__pruning
  !! Module providing post-training pruning utilities for trained networks
  !!
  !! This module implements magnitude-based weight pruning that operates on
  !! already-trained networks without modifying the training pipeline.
  !!
  !! Two pruning modes are supported:
  !!
  !! **Threshold pruning** — sets parameters to zero when:
  !! $$ |w| < \text{threshold} $$
  !!
  !! **Fraction pruning** — removes the smallest \( p\% \) of parameters
  !! in each layer by magnitude.
  !!
  !! After pruning, the network remains executable and forward evaluation
  !! behaves normally.
  use coreutils, only: real32
  use athena__network, only: network_type
  use athena__base_layer, only: learnable_layer_type
  implicit none


  private

  public :: sparsity_info_type
  public :: prune_threshold
  public :: prune_fraction
  public :: get_sparsity_info
  public :: print_sparsity_info
  public :: compact_network


  type :: sparsity_info_type
     !! Type holding sparsity statistics for a layer or network
     integer :: total_params = 0
     !! Total number of trainable parameters
     integer :: pruned_params = 0
     !! Number of parameters set to zero
     real(real32) :: sparsity = 0.0_real32
     !! Ratio of pruned to total parameters
  end type sparsity_info_type


contains

!###############################################################################
  subroutine prune_threshold(network, threshold)
    !! Prune network parameters whose absolute value is below a threshold
    !!
    !! For every learnable layer, any parameter satisfying
    !! \( |w| < \text{threshold} \) is set to zero.
    implicit none

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Trained network to prune
    real(real32), intent(in) :: threshold
    !! Magnitude threshold below which parameters are zeroed

    ! Local variables
    integer :: l, p
    !! Loop indices


    do l = 1, network%num_layers
       select type(layer => network%model(l)%layer)
       class is(learnable_layer_type)
          do p = 1, size(layer%params)
             call prune_array_threshold(layer%params(p)%val(:,1), threshold)
          end do
       end select
    end do

  end subroutine prune_threshold
!###############################################################################


!###############################################################################
  subroutine prune_fraction(network, fraction)
    !! Prune the smallest fraction of parameters in each learnable layer
    !!
    !! For each learnable layer, the smallest \( p\% \) of parameters
    !! by magnitude are set to zero.
    implicit none

    ! Arguments
    type(network_type), intent(inout) :: network
    !! Trained network to prune
    real(real32), intent(in) :: fraction
    !! Fraction of parameters to prune (0.0 to 1.0)

    ! Local variables
    integer :: l
    !! Loop index
    real(real32), allocatable :: all_vals(:)
    real(real32) :: thresh
    integer :: p, total, n_prune, idx


    do l = 1, network%num_layers
       select type(layer => network%model(l)%layer)
       class is(learnable_layer_type)
          ! Gather all parameter magnitudes for this layer
          total = 0
          do p = 1, size(layer%params)
             total = total + size(layer%params(p)%val(:,1))
          end do
          if(total .eq. 0) cycle

          allocate(all_vals(total))
          idx = 0
          do p = 1, size(layer%params)
             all_vals(idx+1:idx+size(layer%params(p)%val(:,1))) = &
                  abs(layer%params(p)%val(:,1))
             idx = idx + size(layer%params(p)%val(:,1))
          end do

          ! Find the threshold at the given fraction
          n_prune = max(0, min(total, nint(fraction * real(total, real32))))
          if(n_prune .gt. 0)then
             call sort_ascending(all_vals, total)
             thresh = all_vals(n_prune)
             do p = 1, size(layer%params)
                call prune_array_threshold( &
                     layer%params(p)%val(:,1), thresh)
             end do
          end if

          deallocate(all_vals)
       end select
    end do

  end subroutine prune_fraction
!###############################################################################


!###############################################################################
  function get_sparsity_info(network) result(info)
    !! Compute sparsity statistics for the entire network
    implicit none

    ! Arguments
    type(network_type), intent(in) :: network
    !! Network to analyse
    type(sparsity_info_type) :: info
    !! Sparsity statistics

    ! Local variables
    integer :: l, p, i, n


    info%total_params = 0
    info%pruned_params = 0

    do l = 1, network%num_layers
       select type(layer => network%model(l)%layer)
       class is(learnable_layer_type)
          do p = 1, size(layer%params)
             n = size(layer%params(p)%val(:,1))
             info%total_params = info%total_params + n
             do i = 1, n
                if(abs(layer%params(p)%val(i,1)) .lt. 1.0E-30_real32)then
                   info%pruned_params = info%pruned_params + 1
                end if
             end do
          end do
       end select
    end do

    if(info%total_params .gt. 0)then
       info%sparsity = real(info%pruned_params, real32) / &
            real(info%total_params, real32)
    else
       info%sparsity = 0.0_real32
    end if

  end function get_sparsity_info
!###############################################################################


!###############################################################################
  subroutine print_sparsity_info(network, unit)
    !! Print sparsity statistics for each layer and the full network
    implicit none

    ! Arguments
    type(network_type), intent(in) :: network
    !! Network to report on
    integer, optional, intent(in) :: unit
    !! Output unit (default: stdout = 6)

    ! Local variables
    integer :: u, l, p, i, n
    integer :: layer_total, layer_pruned
    type(sparsity_info_type) :: total_info


    u = 6
    if(present(unit)) u = unit

    total_info = get_sparsity_info(network)

    write(u, '(A)') "Sparsity Report"
    write(u, '(A)') repeat("-", 60)
    write(u, '(A20, A12, A12, A12)') &
         "Layer", "Total", "Pruned", "Sparsity"
    write(u, '(A)') repeat("-", 60)

    do l = 1, network%num_layers
       select type(layer => network%model(l)%layer)
       class is(learnable_layer_type)
          layer_total = 0
          layer_pruned = 0
          do p = 1, size(layer%params)
             n = size(layer%params(p)%val(:,1))
             layer_total = layer_total + n
             do i = 1, n
                if(abs(layer%params(p)%val(i,1)) .lt. 1.0E-30_real32)then
                   layer_pruned = layer_pruned + 1
                end if
             end do
          end do
          if(layer_total .gt. 0)then
             write(u, '(A20, I12, I12, F12.4)') &
                  layer%name, layer_total, layer_pruned, &
                  real(layer_pruned, real32) / real(layer_total, real32)
          end if
       end select
    end do

    write(u, '(A)') repeat("-", 60)
    write(u, '(A20, I12, I12, F12.4)') &
         "TOTAL", total_info%total_params, total_info%pruned_params, &
         total_info%sparsity

  end subroutine print_sparsity_info
!###############################################################################


!###############################################################################
  subroutine prune_array_threshold(arr, threshold)
    !! Set elements of a 1D array to zero if |element| < threshold
    implicit none

    ! Arguments
    real(real32), dimension(:), intent(inout) :: arr
    !! Array to prune
    real(real32), intent(in) :: threshold
    !! Magnitude threshold

    ! Local variables
    integer :: i


    do i = 1, size(arr)
       if(abs(arr(i)) .lt. threshold) arr(i) = 0.0_real32
    end do

  end subroutine prune_array_threshold
!###############################################################################


!###############################################################################
  subroutine sort_ascending(arr, n)
    !! Simple insertion sort for small-to-moderate arrays
    implicit none

    ! Arguments
    real(real32), dimension(n), intent(inout) :: arr
    !! Array to sort in-place
    integer, intent(in) :: n
    !! Array size

    ! Local variables
    integer :: i, j
    real(real32) :: key


    do i = 2, n
       key = arr(i)
       j = i - 1
       do while(j .ge. 1)
          if(arr(j) .le. key) exit
          arr(j + 1) = arr(j)
          j = j - 1
       end do
       arr(j + 1) = key
    end do

  end subroutine sort_ascending
!###############################################################################


!###############################################################################
  subroutine compact_network(source, compact, batch_size)
    !! Create a new network with dead hidden neurons removed
    !!
    !! For each hidden layer, neurons whose outgoing weights in the next
    !! layer are all zero are removed. The resulting compact network
    !! preserves the same input-output mapping as the pruned source.
    !!
    !! Only the full_layer_type layers in the source are considered.
    !! Auto-added input or flatten layers are skipped.
    !! The compact network is compiled and ready for inference via
    !! predict(). To train further, recompile with your own optimiser.
    use athena__full_layer, only: full_layer_type
    use athena__optimiser, only: sgd_optimiser_type
    implicit none

    ! Arguments
    type(network_type), intent(in) :: source
    !! Pruned source network
    type(network_type), intent(inout) :: compact
    !! Output compact network (compiled, ready for predict)
    integer, optional, intent(in) :: batch_size
    !! Batch size for compact network (default: 1)

    ! Local variables
    integer :: l, j, jj, ii, n_fc, li
    integer :: bs, max_neurons, fc_count
    integer :: m_l, d_l, m_next, orig_input
    integer :: idx, new_idx, kk
    real(real32), parameter :: zero_tol = 1.0E-30_real32
    integer, allocatable :: fc_ids(:)
    integer, allocatable :: n_kept(:)
    integer, allocatable :: kept(:,:)
    integer, allocatable :: orig_m(:), orig_d(:)
    character(len=20), allocatable :: actv_names(:)
    logical, allocatable :: bias_flags(:)
    logical, allocatable :: keep(:)


    bs = 1
    if(present(batch_size)) bs = batch_size

    ! Find all full_layer_type layers in the source
    n_fc = 0
    do l = 1, source%num_layers
       select type(layer => source%model(l)%layer)
       type is(full_layer_type)
          n_fc = n_fc + 1
       end select
    end do

    if(n_fc .eq. 0) return

    allocate(fc_ids(n_fc))
    n_fc = 0
    do l = 1, source%num_layers
       select type(layer => source%model(l)%layer)
       type is(full_layer_type)
          n_fc = n_fc + 1
          fc_ids(n_fc) = l
       end select
    end do

    ! Gather layer info
    allocate(orig_m(n_fc), orig_d(n_fc))
    allocate(actv_names(n_fc), bias_flags(n_fc))
    max_neurons = 0
    do li = 1, n_fc
       select type(layer => source%model(fc_ids(li))%layer)
       type is(full_layer_type)
          orig_m(li) = layer%num_outputs
          orig_d(li) = layer%num_inputs
          if(allocated(layer%activation))then
             actv_names(li) = layer%activation%name
          else
             actv_names(li) = "none"
          end if
          bias_flags(li) = layer%use_bias
          max_neurons = max(max_neurons, layer%num_outputs)
          if(li .eq. 1) &
               max_neurons = max(max_neurons, layer%num_inputs)
       end select
    end do

    ! Determine kept outputs per FC layer
    allocate(n_kept(n_fc), kept(max_neurons, n_fc))
    kept = 0

    do li = 1, n_fc
       m_l = orig_m(li)
       if(li .lt. n_fc)then
          ! Check outgoing weights in next FC layer
          allocate(keep(m_l))
          keep = .false.
          select type(next => source%model(fc_ids(li + 1))%layer)
          type is(full_layer_type)
             m_next = next%num_outputs
             ! Column-major: weight(kk,j) = val((j-1)*m_next + kk)
             ! j is input (neuron of current layer)
             do j = 1, m_l
                do kk = 1, m_next
                   idx = (j - 1) * m_next + kk
                   if(abs(next%params(1)%val(idx, 1)) &
                        .gt. zero_tol)then
                      keep(j) = .true.
                      exit
                   end if
                end do
             end do
          end select

          n_kept(li) = count(keep)
          if(n_kept(li) .eq. 0)then
             keep(1) = .true.
             n_kept(li) = 1
          end if
          fc_count = 0
          do j = 1, m_l
             if(keep(j))then
                fc_count = fc_count + 1
                kept(fc_count, li) = j
             end if
          end do
          deallocate(keep)
       else
          ! Last FC layer: keep all outputs
          n_kept(li) = m_l
          do j = 1, m_l
             kept(j, li) = j
          end do
       end if
    end do

    ! Build compact network
    do li = 1, n_fc
       if(li .eq. 1)then
          call compact%add(full_layer_type( &
               num_inputs=orig_d(1), num_outputs=n_kept(li), &
               use_bias=bias_flags(li), &
               activation=trim(actv_names(li))))
       else
          call compact%add(full_layer_type( &
               num_outputs=n_kept(li), &
               use_bias=bias_flags(li), &
               activation=trim(actv_names(li))))
       end if
    end do

    call compact%compile( &
         optimiser=sgd_optimiser_type(learning_rate=1.0E-3_real32), &
         loss_method='mse', metrics=['loss'], &
         batch_size=bs, verbose=0)

    ! Copy weights from source to compact
    ! After compile, compact%model also has auto-added input layers.
    ! Find the FC layers in the compact model.
    do li = 1, n_fc
       select type(src => source%model(fc_ids(li))%layer)
       type is(full_layer_type)
          ! Find the li-th full_layer_type in compact
          fc_count = 0
          do l = 1, compact%num_layers
             select type(dst => compact%model(l)%layer)
             type is(full_layer_type)
                fc_count = fc_count + 1
                if(fc_count .eq. li)then
                   d_l = orig_d(li)
                   ! Column-major: weight(j,i) = val((i-1)*M + j)
                   do jj = 1, n_kept(li)
                      j = kept(jj, li)
                      if(li .eq. 1)then
                         do ii = 1, d_l
                            idx = (ii - 1) * orig_m(li) + j
                            new_idx = (ii - 1) * &
                                 n_kept(li) + jj
                            dst%params(1)%val(new_idx, 1) = &
                                 src%params(1)%val(idx, 1)
                         end do
                      else
                         do ii = 1, n_kept(li - 1)
                            orig_input = kept(ii, li - 1)
                            idx = (orig_input - 1) * &
                                 orig_m(li) + j
                            new_idx = (ii - 1) * &
                                 n_kept(li) + jj
                            dst%params(1)%val(new_idx, 1) = &
                                 src%params(1)%val(idx, 1)
                         end do
                      end if
                      if(bias_flags(li))then
                         dst%params(2)%val(jj, 1) = &
                              src%params(2)%val(j, 1)
                      end if
                   end do
                   exit
                end if
             end select
          end do
       end select
    end do

    deallocate(fc_ids, n_kept, kept, orig_m, orig_d, &
         actv_names, bias_flags)

  end subroutine compact_network
!###############################################################################

end module athena__pruning
