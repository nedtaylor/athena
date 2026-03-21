program test_pruning
  !! Test suite for the athena__pruning module
  use coreutils, only: real32
  use athena, only: &
       network_type, &
       full_layer_type, &
       kan_layer_type, &
       sgd_optimiser_type, &
       sparsity_info_type, &
       prune_threshold, &
       prune_fraction, &
       get_sparsity_info, &
       print_sparsity_info, &
       compact_network
  use diffstruc, only: array_type
  implicit none

  logical :: success = .true.


!-------------------------------------------------------------------------------
! Test 1: Threshold pruning zeros out small parameters
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: Threshold pruning"
  threshold_test: block
    type(network_type) :: network
    type(sparsity_info_type) :: info
    real(real32), dimension(2,1) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_out

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set known weights: some small, some large
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       ! weights [3, 2] = 6 values, bias [3] = 3 values
       ! Set weights manually
       layer%params(1)%val(:,1) = &
            [0.001_real32, 0.5_real32, -0.002_real32, &
                 -0.8_real32, 0.003_real32, 0.9_real32]
       layer%params(2)%val(:,1) = [0.0001_real32, 0.1_real32, -0.0002_real32]
    end select

    ! Prune with threshold 0.01
    call prune_threshold(network, 0.01_real32)

    ! Check that small weights are zeroed
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       ! 0.001, -0.002, 0.003 should be zeroed (weights)
       if(abs(layer%params(1)%val(1,1)) .gt. 1.0E-30_real32)then
          success = .false.
          write(0,*) 'Weight 0.001 should have been pruned'
       end if

       ! 0.5 should remain
       if(abs(layer%params(1)%val(2,1) - 0.5_real32) .gt. 1.0E-6_real32)then
          success = .false.
          write(0,*) 'Weight 0.5 should not have been pruned'
       end if

       ! -0.8 should remain
       if(abs(layer%params(1)%val(4,1) + 0.8_real32) .gt. 1.0E-6_real32)then
          success = .false.
          write(0,*) 'Weight -0.8 should not have been pruned'
       end if

       ! Bias 0.0001 and -0.0002 should be zeroed
       if(abs(layer%params(2)%val(1,1)) .gt. 1.0E-30_real32)then
          success = .false.
          write(0,*) 'Bias 0.0001 should have been pruned'
       end if

       ! Bias 0.1 should remain
       if(abs(layer%params(2)%val(2,1) - 0.1_real32) .gt. 1.0E-6_real32)then
          success = .false.
          write(0,*) 'Bias 0.1 should not have been pruned'
       end if
    end select

  end block threshold_test


!-------------------------------------------------------------------------------
! Test 2: Pruned network still evaluates correctly
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: Pruned network forward evaluation"
  forward_after_prune: block
    type(network_type) :: network
    real(real32), dimension(2,1) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_before, y_after

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    x_in(:,1) = [1.0_real32, 2.0_real32]

    ! Evaluate before pruning
    y_before = network%predict(input=x_in)

    ! Apply very large threshold (prune everything)
    call prune_threshold(network, 1.0E+6_real32)

    ! Evaluate after pruning — should produce zeros (all weights zeroed)
    y_after = network%predict(input=x_in)

    if(size(y_after, 1) .ne. 3)then
       success = .false.
       write(0,*) 'Output shape changed after pruning'
    end if

    ! All outputs should be approximately zero
    if(any(abs(y_after(:,1)) .gt. 1.0E-6_real32))then
       success = .false.
       write(0,*) 'Fully pruned network should output ~0'
    end if

  end block forward_after_prune


!-------------------------------------------------------------------------------
! Test 3: Sparsity statistics are correct
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: Sparsity statistics"
  sparsity_stats: block
    type(network_type) :: network
    type(sparsity_info_type) :: info

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set known values: 6 weights + 3 biases = 9 total
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = &
            [0.0_real32, 0.5_real32, 0.0_real32, &
                 -0.8_real32, 0.0_real32, 0.9_real32]
       layer%params(2)%val(:,1) = [0.0_real32, 0.1_real32, 0.0_real32]
    end select

    info = get_sparsity_info(network)

    ! Total: 9, Pruned: 5 (zeros), Sparsity: 5/9
    if(info%total_params .ne. 9)then
       success = .false.
       write(0,*) 'Expected total_params=9, got:', info%total_params
    end if

    if(info%pruned_params .ne. 5)then
       success = .false.
       write(0,*) 'Expected pruned_params=5, got:', info%pruned_params
    end if

    if(abs(info%sparsity - 5.0_real32/9.0_real32) .gt. 0.01_real32)then
       success = .false.
       write(0,*) 'Sparsity ratio incorrect:', info%sparsity
    end if

    ! Test print_sparsity_info runs without error
    call print_sparsity_info(network)

  end block sparsity_stats


!-------------------------------------------------------------------------------
! Test 4: Fraction pruning removes correct amount
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: Fraction pruning"
  fraction_test: block
    type(network_type) :: network
    type(sparsity_info_type) :: info_before, info_after

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set weights to known values with no zeros
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = &
            [0.1_real32, 0.2_real32, 0.3_real32, &
                 0.4_real32, 0.5_real32, 0.6_real32]
       layer%params(2)%val(:,1) = [0.7_real32, 0.8_real32, 0.9_real32]
    end select

    info_before = get_sparsity_info(network)

    ! Prune 50% of parameters
    call prune_fraction(network, 0.5_real32)

    info_after = get_sparsity_info(network)

    ! Should have pruned some parameters
    if(info_after%pruned_params .le. info_before%pruned_params)then
       success = .false.
       write(0,*) 'Fraction pruning did not prune any parameters'
    end if

    ! Sparsity should be approximately 0.5
    if(info_after%sparsity .lt. 0.3_real32)then
       success = .false.
       write(0,*) 'Fraction pruning sparsity too low:', info_after%sparsity
    end if

  end block fraction_test


!-------------------------------------------------------------------------------
! Test 5: Pruning on KAN layer
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: Pruning on KAN layer"
  kan_prune: block
    type(network_type) :: network
    type(sparsity_info_type) :: info
    real(real32), dimension(1,1) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_out

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=5, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Prune with threshold
    call prune_threshold(network, 0.1_real32)

    ! Network should still evaluate
    x_in(1,1) = 0.5_real32
    y_out = network%predict(input=x_in)

    if(size(y_out, 1) .ne. 1 .or. size(y_out, 2) .ne. 1)then
       success = .false.
       write(0,*) 'KAN output shape incorrect after pruning'
    end if

    info = get_sparsity_info(network)
    if(info%total_params .le. 0)then
       success = .false.
       write(0,*) 'KAN sparsity info has zero total params'
    end if

  end block kan_prune


!-------------------------------------------------------------------------------
! Test 6: Compact network removes dead neurons
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: Compact network"
  compact_test: block
    type(network_type) :: network, compacted
    real(real32), dimension(2,1) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_src, y_cmp

    ! Build 2 -> 4 -> 3 network
    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=4, activation="linear"))
    call network%add(full_layer_type( &
         num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set layer 1 weights [4, 2] and bias [4]
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = &
            [1.0_real32, 0.5_real32, &
                 0.0_real32, 0.0_real32, &
                 0.3_real32, -0.7_real32, &
                 0.0_real32, 0.0_real32]
       layer%params(2)%val(:,1) = &
            [0.1_real32, 0.0_real32, -0.2_real32, 0.0_real32]
    end select

    ! Set layer 2 weights [3, 4] and bias [3]
    ! Column-major: val((i-1)*3+j) = w(j,i)
    ! Make column 2 and 4 all zero (inputs from neurons 2,4 are dead)
    select type(layer => network%model(2)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = &
            [0.5_real32, -0.4_real32, 0.1_real32, &
                 0.0_real32, 0.0_real32, 0.0_real32, &
                 0.3_real32, 0.6_real32, -0.2_real32, &
                 0.0_real32, 0.0_real32, 0.0_real32]
       layer%params(2)%val(:,1) = &
            [0.05_real32, -0.1_real32, 0.02_real32]
    end select

    ! Evaluate source network
    x_in(:,1) = [1.0_real32, 2.0_real32]
    y_src = network%predict(input=x_in)

    ! Build compact network
    call compact_network(network, compacted, batch_size=1)

    ! Compact should have fewer parameters (neurons 2,4 removed)
    if(compacted%num_params .ge. network%num_params)then
       success = .false.
       write(0,*) 'Compact network should have fewer parameters'
       write(0,*) 'Source:', network%num_params, &
            'Compact:', compacted%num_params
    end if

    ! Evaluate compact network — should give same output
    y_cmp = compacted%predict(input=x_in)

    if(size(y_cmp, 1) .ne. 3)then
       success = .false.
       write(0,*) 'Compact output shape wrong'
    end if

    if(any(abs(y_cmp(:,1) - y_src(:,1)) .gt. 1.0E-5_real32))then
       success = .false.
       write(0,*) 'Compact output differs from source'
       write(0,*) 'Source:', y_src(:,1)
       write(0,*) 'Compact:', y_cmp(:,1)
    end if

  end block compact_test


!-------------------------------------------------------------------------------
! Result
!-------------------------------------------------------------------------------
  if(success)then
     write(*,*) "All pruning tests passed"
     stop 0
  else
     write(0,*) "Some pruning tests FAILED"
     stop 1
  end if

end program test_pruning
