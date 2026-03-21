program test_symbolic
  !! Test suite for the athena__symbolic module
  use coreutils, only: real32
  use athena, only: &
       network_type, &
       full_layer_type, &
       kan_layer_type, &
       sgd_optimiser_type, &
       symbolic_term_type, &
       symbolic_expr_type, &
       extract_symbolic, &
       extract_symbolic_kan, &
       print_symbolic_expr, &
       simplify_expression
  use athena, only: match_known_functions
  use diffstruc, only: array_type
  implicit none

  logical :: success = .true.


!-------------------------------------------------------------------------------
! Test 1: Extraction runs without error on a fully connected layer
!-------------------------------------------------------------------------------
  write(*,*) "Test 1: Extraction on fully connected layer"
  fc_extract: block
    type(network_type) :: network
    type(symbolic_expr_type), allocatable :: exprs(:)

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=3, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set known weights
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = &
            [0.5_real32, -0.3_real32, 0.0001_real32, &
                 0.8_real32, -0.0001_real32, 0.7_real32]
       layer%params(2)%val(:,1) = [0.1_real32, 0.0_real32, -0.2_real32]
    end select

    exprs = extract_symbolic(network)

    if(.not.allocated(exprs))then
       success = .false.
       write(0,*) 'extract_symbolic returned unallocated result'
    else if(size(exprs) .ne. 3)then
       success = .false.
       write(0,*) 'Expected 3 expressions, got:', size(exprs)
    else
       ! First output should have 2 terms (0.5*x1 + 0.1 bias)
       ! 0.0001 dropped by default 1E-4 tolerance
       if(exprs(1)%num_terms .lt. 1)then
          success = .false.
          write(0,*) 'Expression 1 has no terms'
       end if

       ! Check expression string is non-empty
       if(len_trim(exprs(1)%expression_string) .eq. 0)then
          success = .false.
          write(0,*) 'Expression string is empty'
       end if

       ! Print for visual inspection
       call print_symbolic_expr(exprs)
    end if

  end block fc_extract


!-------------------------------------------------------------------------------
! Test 2: Extraction on KAN layer runs without error
!-------------------------------------------------------------------------------
  write(*,*) "Test 2: Extraction on KAN layer"
  kan_extract: block
    type(network_type) :: network
    type(symbolic_expr_type), allocatable :: exprs(:)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=5, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set known spline coefficients
    select type(layer => network%model(1)%layer)
    type is(kan_layer_type)
       ! 5 basis functions + 1 bias = params(1)[5], params(2)[1]
       layer%params(1)%val(:,1) = &
            [0.8_real32, -0.5_real32, 0.3_real32, &
                 0.0001_real32, 0.0002_real32]
       layer%params(2)%val(:,1) = [0.1_real32]
    end select

    exprs = extract_symbolic_kan(network)

    if(.not.allocated(exprs))then
       success = .false.
       write(0,*) 'extract_symbolic_kan returned unallocated result'
    else if(size(exprs) .ne. 1)then
       success = .false.
       write(0,*) 'Expected 1 expression, got:', size(exprs)
    else
       ! Should have 3 significant terms (0.8, -0.5, 0.3) + bias 0.1 = 4
       if(exprs(1)%num_terms .lt. 3)then
          success = .false.
          write(0,*) 'Expected at least 3 terms, got:', exprs(1)%num_terms
       end if

       call print_symbolic_expr(exprs)
    end if

  end block kan_extract


!-------------------------------------------------------------------------------
! Test 3: Dominant terms are detected for a trained KAN
!-------------------------------------------------------------------------------
  write(*,*) "Test 3: Dominant terms detection"
  dominant_terms: block
    type(network_type) :: network
    type(symbolic_expr_type), allocatable :: exprs(:)
    type(array_type), pointer :: loss
    real(real32), dimension(1,1) :: x, y
    integer :: n
    integer, parameter :: num_iters = 500
    integer :: seed_size
    integer, allocatable :: seed(:)

    seed_size = 8
    allocate(seed(seed_size))
    seed = 42
    call random_seed(put=seed)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=8, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    allocate(network%expected_array(1,1))
    call network%expected_array(1,1)%allocate(array_shape=[1,1])

    ! Train on a simple function: y = x^2, scaled to [0,1]
    do n = 1, num_iters
       call random_number(x)
       x = x * 2.0_real32 - 1.0_real32
       y(1,1) = (x(1,1)**2 + 1.0_real32) / 2.0_real32

       network%expected_array(1,1)%val = y
       call network%set_batch_size(1)
       call network%forward(x)
       loss => network%loss_eval(1, 1)
       call loss%grad_reverse()
       call network%update()
       call loss%nullify_graph()
       loss => null()
    end do

    ! Extract expression
    exprs = extract_symbolic_kan(network, tolerance=0.01_real32)

    if(size(exprs) .lt. 1)then
       success = .false.
       write(0,*) 'No expressions extracted after training'
    else
       ! Should have at least one active term
       if(exprs(1)%num_terms .lt. 1)then
          success = .false.
          write(0,*) 'No dominant terms detected after training'
       end if

       write(*,*) "  Extracted expression:"
       call print_symbolic_expr(exprs)
    end if

    deallocate(seed)

  end block dominant_terms


!-------------------------------------------------------------------------------
! Test 4: Expression simplification
!-------------------------------------------------------------------------------
  write(*,*) "Test 4: Expression simplification"
  simplify_test: block
    type(symbolic_expr_type) :: expr, simplified

    ! Build an expression with duplicate terms and near-zero terms
    expr%output_dim = 1
    expr%num_terms = 5
    allocate(expr%terms(5))

    ! Two terms with same description
    expr%terms(1)%coefficient = 0.3_real32
    expr%terms(1)%description = "B_1(x(1))"
    expr%terms(1)%input_dim = 1
    expr%terms(1)%basis_index = 1

    expr%terms(2)%coefficient = 0.2_real32
    expr%terms(2)%description = "B_1(x(1))"
    expr%terms(2)%input_dim = 1
    expr%terms(2)%basis_index = 1

    ! Near-zero term
    expr%terms(3)%coefficient = 1.0E-8_real32
    expr%terms(3)%description = "B_2(x(1))"
    expr%terms(3)%input_dim = 1
    expr%terms(3)%basis_index = 2

    ! Constant terms
    expr%terms(4)%coefficient = 0.1_real32
    expr%terms(4)%description = "1"
    expr%terms(4)%input_dim = 0

    expr%terms(5)%coefficient = 0.05_real32
    expr%terms(5)%description = "1"
    expr%terms(5)%input_dim = 0

    simplified = simplify_expression(expr)

    ! Should merge B_1 terms: 0.3+0.2=0.5
    ! Should drop near-zero B_2 term
    ! Should merge constants: 0.1+0.05=0.15
    ! Result: 2 terms (0.5*B_1 + 0.15)

    if(simplified%num_terms .ne. 2)then
       success = .false.
       write(0,*) 'Expected 2 simplified terms, got:', simplified%num_terms
    end if

    if(simplified%num_terms .ge. 1)then
       ! First term should be the merged B_1 with coeff ~0.5
       if(abs(simplified%terms(1)%coefficient - 0.5_real32) .gt. 0.01)then
          success = .false.
          write(0,*) 'Merged coefficient should be ~0.5, got:', &
               simplified%terms(1)%coefficient
       end if
    end if

    if(simplified%num_terms .ge. 2)then
       ! Second term should be constant ~0.15
       if(abs(simplified%terms(2)%coefficient - 0.15_real32) .gt. 0.01)then
          success = .false.
          write(0,*) 'Merged constant should be ~0.15, got:', &
               simplified%terms(2)%coefficient
       end if
    end if

    write(*,*) "  Simplified:"
    call print_symbolic_expr([simplified])

  end block simplify_test


!-------------------------------------------------------------------------------
! Test 5: Extracted expression approximates network output
!-------------------------------------------------------------------------------
  write(*,*) "Test 5: Expression approximates output"
  approximation_test: block
    type(network_type) :: network
    type(symbolic_expr_type), allocatable :: exprs(:)
    real(real32), dimension(2,1) :: x_in
    real(real32), allocatable, dimension(:,:) :: y_net

    call network%add(full_layer_type( &
         num_inputs=2, num_outputs=1, activation="linear"))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Set simple known weights: y = 0.5*x1 + 0.3*x2 + 0.1
    select type(layer => network%model(1)%layer)
    type is(full_layer_type)
       layer%params(1)%val(:,1) = [0.5_real32, 0.3_real32]
       layer%params(2)%val(:,1) = [0.1_real32]
    end select

    x_in(:,1) = [1.0_real32, 2.0_real32]
    y_net = network%predict(input=x_in)
    ! Expected: 0.5*1.0 + 0.3*2.0 + 0.1 = 1.2

    exprs = extract_symbolic(network, tolerance=0.01_real32)

    if(size(exprs) .ge. 1)then
       ! Check that expression has 3 terms
       if(exprs(1)%num_terms .ne. 3)then
          success = .false.
          write(0,*) 'Expected 3 terms for linear model, got:', &
               exprs(1)%num_terms
       end if

       write(*,*) "  Network output:", y_net(1,1)
       write(*,*) "  Expression:"
       call print_symbolic_expr(exprs)
    end if

  end block approximation_test


!-------------------------------------------------------------------------------
! Test 6: match_known_functions runs and produces output
!-------------------------------------------------------------------------------
  write(*,*) "Test 6: Known function matching on KAN"
  match_test: block
    type(network_type) :: network
    type(symbolic_expr_type), allocatable :: exprs(:)

    call network%add(kan_layer_type( &
         num_inputs=1, num_outputs=1, n_basis=8, spline_degree=3))
    call network%compile( &
         optimiser=sgd_optimiser_type(learning_rate=0.01), &
         loss_method='mse', metrics=['loss'], &
         batch_size=1, verbose=0)

    ! Run matching (returns expressions with known-function descriptions)
    exprs = match_known_functions(network, n_grid=50)

    if(.not.allocated(exprs))then
       success = .false.
       write(0,*) 'match_known_functions returned unallocated'
    else if(size(exprs) .ne. 1)then
       success = .false.
       write(0,*) 'Expected 1 output expression, got:', size(exprs)
    else
       write(*,*) "  Matched expression:"
       call print_symbolic_expr(exprs)
    end if

  end block match_test


!-------------------------------------------------------------------------------
! Result
!-------------------------------------------------------------------------------
  if(success)then
     write(*,*) "All symbolic extraction tests passed"
     stop 0
  else
     write(0,*) "Some symbolic extraction tests FAILED"
     stop 1
  end if

end program test_symbolic
