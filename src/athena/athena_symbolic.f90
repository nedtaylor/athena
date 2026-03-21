module athena__symbolic
  !! Module providing symbolic regression extraction for trained networks
  !!
  !! This module attempts to extract approximate symbolic expressions from
  !! trained networks by analysing basis-function contributions.
  !!
  !! Best suited for:
  !! - small networks
  !! - KAN layers (B-spline and FastKAN)
  !! - sparse networks
  !!
  !! For KAN layers, each basis function contribution is treated as a
  !! candidate term and terms with weight magnitude below a tolerance
  !! are discarded.
  !!
  !! Example recovered expression:
  !!   y ~ 0.80 * exp(-(x - 0.30)^2 / 0.08) - 0.50 * exp(-(x + 0.70)^2 / 0.18) + 0.10
  use coreutils, only: real32
  use athena__network, only: network_type
  use athena__base_layer, only: learnable_layer_type
  implicit none


  private

  public :: symbolic_term_type
  public :: symbolic_expr_type
  public :: extract_symbolic
  public :: extract_symbolic_kan
  public :: print_symbolic_expr
  public :: simplify_expression
  public :: match_known_functions


  type :: symbolic_term_type
     !! A single term in a symbolic expression
     real(real32) :: coefficient = 0.0_real32
     !! Weight/coefficient of the term
     character(len=256) :: description = ' '
     !! Human-readable description of the basis function
     integer :: input_dim = 0
     !! Input dimension this term acts on (0 = constant)
     integer :: basis_index = 0
     !! Index of the basis function within its dimension
     real(real32) :: centre = 0.0_real32
     !! Centre of basis function (for RBF/B-spline reporting)
  end type symbolic_term_type


  type :: symbolic_expr_type
     !! A symbolic expression composed of multiple terms
     integer :: output_dim = 0
     !! Which output neuron this expression corresponds to
     integer :: num_terms = 0
     !! Number of active terms
     type(symbolic_term_type), allocatable :: terms(:)
     !! Array of terms
     character(len=4096) :: expression_string = ' '
     !! Simplified human-readable expression
  end type symbolic_expr_type


contains

!###############################################################################
  function extract_symbolic(network, tolerance) result(exprs)
    !! Extract symbolic expressions from a trained network
    !!
    !! Dispatches to layer-specific extraction based on layer type.
    !! Currently supports KAN and FastKAN layers.
    !! For fully-connected layers, reports weighted sums.
    implicit none

    ! Arguments
    type(network_type), intent(in) :: network
    !! Trained network
    real(real32), optional, intent(in) :: tolerance
    !! Tolerance below which terms are discarded (default: 1E-4)
    type(symbolic_expr_type), allocatable :: exprs(:)
    !! Extracted expressions (one per output neuron of the last layer)

    ! Local variables
    real(real32) :: tol
    integer :: l


    tol = 1.0E-4_real32
    if(present(tolerance)) tol = tolerance

    ! Use the last learnable layer
    do l = network%num_layers, 1, -1
       select type(layer => network%model(l)%layer)
       class is(learnable_layer_type)
          exprs = extract_from_learnable(layer, tol)
          return
       end select
    end do

    ! No learnable layers found
    allocate(exprs(0))

  end function extract_symbolic
!###############################################################################


!###############################################################################
  function extract_symbolic_kan(network, tolerance) result(exprs)
    !! Extract symbolic expressions specifically from KAN layers
    !!
    !! Searches for the last KAN or FastKAN layer and extracts
    !! basis function contributions.
    use athena__kan_layer, only: kan_layer_type
    use athena__fastkan_layer, only: fastkan_layer_type
    implicit none

    ! Arguments
    type(network_type), intent(in) :: network
    !! Trained network containing KAN layers
    real(real32), optional, intent(in) :: tolerance
    !! Tolerance below which terms are discarded
    type(symbolic_expr_type), allocatable :: exprs(:)
    !! Extracted expressions

    ! Local variables
    real(real32) :: tol
    integer :: l


    tol = 1.0E-4_real32
    if(present(tolerance)) tol = tolerance

    do l = network%num_layers, 1, -1
       select type(layer => network%model(l)%layer)
       type is(kan_layer_type)
          exprs = extract_from_kan(layer, tol)
          return
       type is(fastkan_layer_type)
          exprs = extract_from_fastkan(layer, tol)
          return
       end select
    end do

    ! Fallback
    exprs = extract_symbolic(network, tol)

  end function extract_symbolic_kan
!###############################################################################


!###############################################################################
  subroutine print_symbolic_expr(exprs, unit)
    !! Print symbolic expressions to a unit
    implicit none

    ! Arguments
    type(symbolic_expr_type), dimension(:), intent(in) :: exprs
    !! Expressions to print
    integer, optional, intent(in) :: unit
    !! Output unit (default: stdout)

    ! Local variables
    integer :: u, e, t


    u = 6
    if(present(unit)) u = unit

    do e = 1, size(exprs)
       write(u, '(A,I0,A)') "Output ", exprs(e)%output_dim, ":"
       write(u, '(2X,A)') trim(exprs(e)%expression_string)
       write(u, '(A)')
       if(exprs(e)%num_terms .gt. 0)then
          write(u, '(2X,A)') "Terms:"
          do t = 1, exprs(e)%num_terms
             write(u, '(4X,SP,ES12.4,SS,A,A)') &
                  exprs(e)%terms(t)%coefficient, " * ", &
                  trim(exprs(e)%terms(t)%description)
          end do
          write(u, '(A)')
       end if
    end do

  end subroutine print_symbolic_expr
!###############################################################################


!###############################################################################
  function simplify_expression(expr) result(simplified)
    !! Simplify a symbolic expression
    !!
    !! Applies the following simplifications:
    !! - merge duplicate terms (same description)
    !! - remove near-zero coefficients
    !! - combine constants
    implicit none

    ! Arguments
    type(symbolic_expr_type), intent(in) :: expr
    !! Expression to simplify
    type(symbolic_expr_type) :: simplified
    !! Simplified expression

    ! Local variables
    type(symbolic_term_type), allocatable :: merged(:)
    integer :: i, j, n_merged
    logical, allocatable :: used(:)
    real(real32) :: constant_sum
    real(real32), parameter :: merge_tol = 1.0E-6_real32


    if(expr%num_terms .eq. 0)then
       simplified = expr
       return
    end if

    allocate(merged(expr%num_terms))
    allocate(used(expr%num_terms))
    used = .false.
    n_merged = 0
    constant_sum = 0.0_real32

    ! Merge duplicate terms and accumulate constants
    do i = 1, expr%num_terms
       if(used(i)) cycle

       ! Check if constant term
       if(expr%terms(i)%input_dim .eq. 0)then
          constant_sum = constant_sum + expr%terms(i)%coefficient
          used(i) = .true.
          cycle
       end if

       ! Start a new merged term
       n_merged = n_merged + 1
       merged(n_merged) = expr%terms(i)
       used(i) = .true.

       ! Look for duplicates
       do j = i + 1, expr%num_terms
          if(used(j)) cycle
          if(trim(expr%terms(j)%description) .eq. &
               trim(merged(n_merged)%description))then
             merged(n_merged)%coefficient = &
                  merged(n_merged)%coefficient + expr%terms(j)%coefficient
             used(j) = .true.
          end if
       end do

       ! Remove if merged coefficient is near zero
       if(abs(merged(n_merged)%coefficient) .lt. merge_tol)then
          n_merged = n_merged - 1
       end if
    end do

    ! Add constant term if non-zero
    if(abs(constant_sum) .ge. merge_tol)then
       n_merged = n_merged + 1
       merged(n_merged)%coefficient = constant_sum
       merged(n_merged)%description = "1"
       merged(n_merged)%input_dim = 0
       merged(n_merged)%basis_index = 0
    end if

    ! Build simplified expression
    simplified%output_dim = expr%output_dim
    simplified%num_terms = n_merged
    if(n_merged .gt. 0)then
       allocate(simplified%terms(n_merged))
       simplified%terms(1:n_merged) = merged(1:n_merged)
    else
       allocate(simplified%terms(0))
    end if

    ! Build expression string
    call build_expression_string(simplified)

    deallocate(merged, used)

  end function simplify_expression
!###############################################################################


!###############################################################################
  function extract_from_learnable(layer, tol) result(exprs)
    !! Extract symbolic expressions from a generic learnable layer
    !!
    !! Reports weighted connections as linear combination terms.
    implicit none

    ! Arguments
    class(learnable_layer_type), intent(in) :: layer
    !! Learnable layer
    real(real32), intent(in) :: tol
    !! Tolerance for dropping terms
    type(symbolic_expr_type), allocatable :: exprs(:)

    ! Local variables
    integer :: m, d, j, i, idx, n_active
    real(real32) :: w
    type(symbolic_term_type), allocatable :: terms_buf(:)
    character(len=16) :: input_label


    ! Use weight_shape if available
    if(.not.allocated(layer%weight_shape))then
       allocate(exprs(0))
       return
    end if

    m = layer%weight_shape(1, 1)
    d = layer%weight_shape(2, 1)
    allocate(exprs(m))

    do j = 1, m
       exprs(j)%output_dim = j
       allocate(terms_buf(d + 1))
       n_active = 0

       ! Weight terms
       do i = 1, d
          idx = (j - 1) * d + i
          if(idx .gt. size(layer%params(1)%val(:,1))) exit
          w = layer%params(1)%val(idx, 1)
          if(abs(w) .ge. tol)then
             n_active = n_active + 1
             terms_buf(n_active)%coefficient = w
             write(input_label, '(A,I0,A)') "x(", i, ")"
             terms_buf(n_active)%description = trim(input_label)
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = 0
          end if
       end do

       ! Bias term (if present, params(2))
       if(size(layer%params) .ge. 2)then
          if(j .le. size(layer%params(2)%val(:,1)))then
             w = layer%params(2)%val(j, 1)
             if(abs(w) .ge. tol)then
                n_active = n_active + 1
                terms_buf(n_active)%coefficient = w
                terms_buf(n_active)%description = "1"
                terms_buf(n_active)%input_dim = 0
                terms_buf(n_active)%basis_index = 0
             end if
          end if
       end if

       exprs(j)%num_terms = n_active
       allocate(exprs(j)%terms(n_active))
       if(n_active .gt. 0) &
            exprs(j)%terms(1:n_active) = terms_buf(1:n_active)
       call build_expression_string(exprs(j))
       deallocate(terms_buf)
    end do

  end function extract_from_learnable
!###############################################################################


!###############################################################################
  function extract_from_kan(layer, tol) result(exprs)
    !! Extract symbolic expressions from a B-spline KAN layer
    !!
    !! Each term corresponds to a B-spline basis function contribution:
    !!   y_j = sum_{i,k} W_{j,(i-1)*K+k} * B_k(x_i) + bias_j
    use athena__kan_layer, only: kan_layer_type
    implicit none

    ! Arguments
    type(kan_layer_type), intent(in) :: layer
    !! KAN layer
    real(real32), intent(in) :: tol
    !! Tolerance for dropping terms
    type(symbolic_expr_type), allocatable :: exprs(:)

    ! Local variables
    integer :: m, d, n_b, j, i, kb, idx, n_active
    real(real32) :: w, centre, left_knot, right_knot
    type(symbolic_term_type), allocatable :: terms_buf(:)
    character(len=256) :: desc


    m = layer%num_outputs
    d = layer%num_inputs
    n_b = layer%n_basis
    allocate(exprs(m))

    do j = 1, m
       exprs(j)%output_dim = j
       allocate(terms_buf(d * n_b + 1))
       n_active = 0

       do i = 1, d
          do kb = 1, n_b
             idx = (j - 1) * (d * n_b) + (i - 1) * n_b + kb
             if(idx .gt. size(layer%params(1)%val(:,1))) cycle
             w = layer%params(1)%val(idx, 1)
             if(abs(w) .lt. tol) cycle

             n_active = n_active + 1
             terms_buf(n_active)%coefficient = w
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = kb

             ! Compute approximate centre of the B-spline basis function
             ! For degree p, basis kb spans knots(kb)..knots(kb+p+1)
             ! Centre is approximately the average of the support
             if(kb .le. size(layer%knots) - layer%spline_degree)then
                left_knot = layer%knots(kb)
                right_knot = layer%knots( &
                     min(kb + layer%spline_degree + 1, size(layer%knots)))
                centre = 0.5_real32 * (left_knot + right_knot)
             else
                centre = 0.0_real32
             end if
             terms_buf(n_active)%centre = centre

             write(desc, '(A,I0,A,I0,A,I0,A)') &
                  "B_", kb, "(x(", i, ")) [centre=", &
                  nint(centre * 100.0_real32), "/100]"
             terms_buf(n_active)%description = desc
          end do
       end do

       ! Bias term
       if(.not.layer%use_base_activation .and. layer%use_bias)then
          if(size(layer%params) .ge. 2)then
             if(j .le. size(layer%params(2)%val(:,1)))then
                w = layer%params(2)%val(j, 1)
                if(abs(w) .ge. tol)then
                   n_active = n_active + 1
                   terms_buf(n_active)%coefficient = w
                   terms_buf(n_active)%description = "1"
                   terms_buf(n_active)%input_dim = 0
                   terms_buf(n_active)%basis_index = 0
                end if
             end if
          end if
       end if

       exprs(j)%num_terms = n_active
       allocate(exprs(j)%terms(n_active))
       if(n_active .gt. 0) &
            exprs(j)%terms(1:n_active) = terms_buf(1:n_active)
       call build_expression_string(exprs(j))
       deallocate(terms_buf)
    end do

  end function extract_from_kan
!###############################################################################


!###############################################################################
  function extract_from_fastkan(layer, tol) result(exprs)
    !! Extract symbolic expressions from a FastKAN (RBF) layer
    !!
    !! Each term corresponds to an RBF contribution:
    !!   y_j = sum_{i,k} W_{j,(i-1)*K+k} * exp(-(x_i - c_k)^2 / (2*s_k^2))
    use athena__fastkan_layer, only: fastkan_layer_type
    implicit none

    ! Arguments
    type(fastkan_layer_type), intent(in) :: layer
    !! FastKAN layer
    real(real32), intent(in) :: tol
    !! Tolerance for dropping terms
    type(symbolic_expr_type), allocatable :: exprs(:)

    ! Local variables
    integer :: m, d, n_b, j, i, kb, idx, w_idx, n_active
    real(real32) :: w, ctr, bw
    type(symbolic_term_type), allocatable :: terms_buf(:)
    character(len=256) :: desc


    m = layer%num_outputs
    d = layer%num_inputs
    n_b = layer%n_basis
    allocate(exprs(m))

    do j = 1, m
       exprs(j)%output_dim = j
       allocate(terms_buf(d * n_b + 1))
       n_active = 0

       do i = 1, d
          do kb = 1, n_b
             w_idx = (j - 1) * (d * n_b) + (i - 1) * n_b + kb
             if(w_idx .gt. size(layer%params(3)%val(:,1))) cycle
             w = layer%params(3)%val(w_idx, 1)
             if(abs(w) .lt. tol) cycle

             ! Get centre and bandwidth
             idx = (i - 1) * n_b + kb
             ctr = 0.0_real32
             bw = 1.0_real32
             if(idx .le. size(layer%params(1)%val(:,1))) &
                  ctr = layer%params(1)%val(idx, 1)
             if(idx .le. size(layer%params(2)%val(:,1))) &
                  bw = layer%params(2)%val(idx, 1)

             n_active = n_active + 1
             terms_buf(n_active)%coefficient = w
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = kb
             terms_buf(n_active)%centre = ctr

             write(desc, '(A,I0,A,F7.3,A,F7.3,A)') &
                  "exp(-(x(", i, ") - ", ctr, &
                  ")^2 / (2*", bw**2, "))"
             terms_buf(n_active)%description = desc
          end do
       end do

       ! Bias term (params(4) if present)
       if(size(layer%params) .ge. 4)then
          if(j .le. size(layer%params(4)%val(:,1)))then
             w = layer%params(4)%val(j, 1)
             if(abs(w) .ge. tol)then
                n_active = n_active + 1
                terms_buf(n_active)%coefficient = w
                terms_buf(n_active)%description = "1"
                terms_buf(n_active)%input_dim = 0
                terms_buf(n_active)%basis_index = 0
             end if
          end if
       end if

       exprs(j)%num_terms = n_active
       allocate(exprs(j)%terms(n_active))
       if(n_active .gt. 0) &
            exprs(j)%terms(1:n_active) = terms_buf(1:n_active)
       call build_expression_string(exprs(j))
       deallocate(terms_buf)
    end do

  end function extract_from_fastkan
!###############################################################################


!###############################################################################
  subroutine build_expression_string(expr)
    !! Build a human-readable expression string from terms
    implicit none

    ! Arguments
    type(symbolic_expr_type), intent(inout) :: expr
    !! Expression to build string for

    ! Local variables
    integer :: t
    character(len=32) :: coeff_str
    logical :: first


    expr%expression_string = "y ~ "
    if(expr%num_terms .eq. 0)then
       expr%expression_string = "y ~ 0"
       return
    end if

    first = .true.
    do t = 1, expr%num_terms
       write(coeff_str, '(SP,ES10.3)') expr%terms(t)%coefficient

       if(first)then
          if(expr%terms(t)%input_dim .eq. 0)then
             ! Constant term
             write(coeff_str, '(SP,ES10.3)') expr%terms(t)%coefficient
             expr%expression_string = &
                  trim(expr%expression_string) // trim(adjustl(coeff_str))
          else
             expr%expression_string = &
                  trim(expr%expression_string) // &
                  trim(adjustl(coeff_str)) // " * " // &
                  trim(expr%terms(t)%description)
          end if
          first = .false.
       else
          if(expr%terms(t)%input_dim .eq. 0)then
             expr%expression_string = &
                  trim(expr%expression_string) // " " // &
                  trim(adjustl(coeff_str))
          else
             expr%expression_string = &
                  trim(expr%expression_string) // " " // &
                  trim(adjustl(coeff_str)) // " * " // &
                  trim(expr%terms(t)%description)
          end if
       end if
    end do

  end subroutine build_expression_string
!###############################################################################


!###############################################################################
  function match_known_functions(network, n_grid) result(exprs)
    !! Match KAN/FastKAN activations to known mathematical functions
    !!
    !! For each activation in the last KAN or FastKAN layer, evaluates
    !! the learned function on a grid and fits candidate functions
    !! (sin, cos, exp, x, x^2, x^3, |x|, tanh) using least-squares.
    !! Returns symbolic expressions using the best-matching functions.
    !!
    !! Falls back to raw basis-function extraction if R^2 < 0.95.
    use athena__kan_layer, only: kan_layer_type
    use athena__fastkan_layer, only: fastkan_layer_type
    implicit none

    ! Arguments
    type(network_type), intent(in) :: network
    !! Trained network with KAN layers
    integer, optional, intent(in) :: n_grid
    !! Number of grid points for evaluation (default: 100)
    type(symbolic_expr_type), allocatable :: exprs(:)
    !! Matched expressions

    ! Local variables
    integer :: l, ng


    ng = 100
    if(present(n_grid)) ng = n_grid

    ! Find last KAN or FastKAN layer
    do l = network%num_layers, 1, -1
       select type(layer => network%model(l)%layer)
       type is(kan_layer_type)
          exprs = match_from_kan(layer, ng)
          return
       type is(fastkan_layer_type)
          exprs = match_from_fastkan(layer, ng)
          return
       end select
    end do

    ! No KAN layer found
    allocate(exprs(0))

  end function match_known_functions
!###############################################################################


!###############################################################################
  function match_from_kan(layer, ng) result(exprs)
    !! Match B-spline KAN activations to known functions
    use athena__kan_layer, only: kan_layer_type
    implicit none

    ! Arguments
    type(kan_layer_type), intent(in) :: layer
    !! KAN layer
    integer, intent(in) :: ng
    !! Number of grid points
    type(symbolic_expr_type), allocatable :: exprs(:)
    !! Matched expressions

    ! Local variables
    integer :: m, d, n_b, j, i, kb, idx, n_active, g
    real(real32) :: w
    real(real32), allocatable :: x_grid(:), phi_vals(:)
    real(real32), allocatable :: basis_at_grid(:,:)
    real(real32) :: grid_min, grid_max
    real(real32) :: best_a, best_b, best_r2
    character(len=256) :: best_desc
    type(symbolic_term_type), allocatable :: terms_buf(:)
    real(real32) :: constant_accum


    m = layer%num_outputs
    d = layer%num_inputs
    n_b = layer%n_basis
    allocate(exprs(m))

    ! Set up evaluation grid from knot boundaries
    grid_min = layer%knots(1)
    grid_max = layer%knots(size(layer%knots))
    allocate(x_grid(ng))
    do g = 1, ng
       x_grid(g) = grid_min + (grid_max - grid_min) * &
            real(g - 1, real32) / real(ng - 1, real32)
    end do

    ! Pre-compute basis functions on grid
    allocate(basis_at_grid(n_b, ng))
    do g = 1, ng
       call eval_bspline_at(x_grid(g), layer%knots, &
            size(layer%knots), layer%spline_degree, n_b, &
            basis_at_grid(:, g))
    end do

    allocate(phi_vals(ng))

    do j = 1, m
       exprs(j)%output_dim = j
       allocate(terms_buf(d + 1))
       n_active = 0
       constant_accum = 0.0_real32

       do i = 1, d
          ! Evaluate phi_{j,i}(x) on the grid
          phi_vals = 0.0_real32
          do kb = 1, n_b
             idx = (j - 1) * (d * n_b) + (i - 1) * n_b + kb
             if(idx .gt. size(layer%params(1)%val(:,1))) cycle
             w = layer%params(1)%val(idx, 1)
             do g = 1, ng
                phi_vals(g) = phi_vals(g) + &
                     w * basis_at_grid(kb, g)
             end do
          end do

          ! Skip near-zero activations
          if(maxval(abs(phi_vals)) .lt. 1.0E-4_real32) cycle

          ! Match to known functions
          call match_best_function(x_grid, phi_vals, ng, i, &
               best_a, best_b, best_r2, best_desc)

          n_active = n_active + 1
          if(best_r2 .gt. 0.95_real32)then
             terms_buf(n_active)%coefficient = best_a
             terms_buf(n_active)%description = best_desc
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = 0
             constant_accum = constant_accum + best_b
          else
             ! Fall back to generic description
             terms_buf(n_active)%coefficient = 1.0_real32
             write(best_desc, '(A,I0,A)') "phi(x(", i, "))"
             terms_buf(n_active)%description = best_desc
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = 0
          end if
       end do

       ! Bias + accumulated constant offsets
       if(.not.layer%use_base_activation .and. &
            layer%use_bias)then
          if(size(layer%params) .ge. 2)then
             if(j .le. size(layer%params(2)%val(:,1)))then
                constant_accum = constant_accum + &
                     layer%params(2)%val(j, 1)
             end if
          end if
       end if

       if(abs(constant_accum) .ge. 1.0E-4_real32)then
          n_active = n_active + 1
          terms_buf(n_active)%coefficient = constant_accum
          terms_buf(n_active)%description = "1"
          terms_buf(n_active)%input_dim = 0
          terms_buf(n_active)%basis_index = 0
       end if

       exprs(j)%num_terms = n_active
       allocate(exprs(j)%terms(n_active))
       if(n_active .gt. 0) &
            exprs(j)%terms(1:n_active) = terms_buf(1:n_active)
       call build_expression_string(exprs(j))
       deallocate(terms_buf)
    end do

    deallocate(x_grid, phi_vals, basis_at_grid)

  end function match_from_kan
!###############################################################################


!###############################################################################
  function match_from_fastkan(layer, ng) result(exprs)
    !! Match FastKAN (RBF) activations to known functions
    use athena__fastkan_layer, only: fastkan_layer_type
    implicit none

    ! Arguments
    type(fastkan_layer_type), intent(in) :: layer
    !! FastKAN layer
    integer, intent(in) :: ng
    !! Number of grid points
    type(symbolic_expr_type), allocatable :: exprs(:)
    !! Matched expressions

    ! Local variables
    integer :: m, d, n_b, j, i, kb, w_idx, idx, n_active, g
    real(real32) :: w, ctr, bw, diff
    real(real32), allocatable :: x_grid(:), phi_vals(:)
    real(real32) :: grid_min, grid_max
    real(real32) :: best_a, best_b, best_r2
    character(len=256) :: best_desc
    type(symbolic_term_type), allocatable :: terms_buf(:)
    real(real32) :: constant_accum


    m = layer%num_outputs
    d = layer%num_inputs
    n_b = layer%n_basis
    allocate(exprs(m))

    ! Set up evaluation grid
    grid_min = -1.0_real32
    grid_max = 1.0_real32
    if(size(layer%params(1)%val(:,1)) .gt. 0)then
       grid_min = minval(layer%params(1)%val(:,1)) - 0.5_real32
       grid_max = maxval(layer%params(1)%val(:,1)) + 0.5_real32
    end if
    allocate(x_grid(ng))
    do g = 1, ng
       x_grid(g) = grid_min + (grid_max - grid_min) * &
            real(g - 1, real32) / real(ng - 1, real32)
    end do

    allocate(phi_vals(ng))

    do j = 1, m
       exprs(j)%output_dim = j
       allocate(terms_buf(d + 1))
       n_active = 0
       constant_accum = 0.0_real32

       do i = 1, d
          ! Evaluate phi_{j,i}(x) on the grid
          phi_vals = 0.0_real32
          do kb = 1, n_b
             w_idx = (j - 1) * (d * n_b) + (i - 1) * n_b + kb
             if(w_idx .gt. size(layer%params(3)%val(:,1))) cycle
             w = layer%params(3)%val(w_idx, 1)

             idx = (i - 1) * n_b + kb
             ctr = 0.0_real32
             bw = 1.0_real32
             if(idx .le. size(layer%params(1)%val(:,1))) &
                  ctr = layer%params(1)%val(idx, 1)
             if(idx .le. size(layer%params(2)%val(:,1))) &
                  bw = layer%params(2)%val(idx, 1)

             do g = 1, ng
                diff = (x_grid(g) - ctr) / max(abs(bw), 1.0E-8_real32)
                phi_vals(g) = phi_vals(g) + &
                     w * exp(-0.5_real32 * diff * diff)
             end do
          end do

          ! Skip near-zero activations
          if(maxval(abs(phi_vals)) .lt. 1.0E-4_real32) cycle

          ! Match to known functions
          call match_best_function(x_grid, phi_vals, ng, i, &
               best_a, best_b, best_r2, best_desc)

          n_active = n_active + 1
          if(best_r2 .gt. 0.95_real32)then
             terms_buf(n_active)%coefficient = best_a
             terms_buf(n_active)%description = best_desc
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = 0
             constant_accum = constant_accum + best_b
          else
             terms_buf(n_active)%coefficient = 1.0_real32
             write(best_desc, '(A,I0,A)') "phi(x(", i, "))"
             terms_buf(n_active)%description = best_desc
             terms_buf(n_active)%input_dim = i
             terms_buf(n_active)%basis_index = 0
          end if
       end do

       ! Bias + accumulated constant offsets
       if(size(layer%params) .ge. 4)then
          if(j .le. size(layer%params(4)%val(:,1)))then
             constant_accum = constant_accum + &
                  layer%params(4)%val(j, 1)
          end if
       end if

       if(abs(constant_accum) .ge. 1.0E-4_real32)then
          n_active = n_active + 1
          terms_buf(n_active)%coefficient = constant_accum
          terms_buf(n_active)%description = "1"
          terms_buf(n_active)%input_dim = 0
          terms_buf(n_active)%basis_index = 0
       end if

       exprs(j)%num_terms = n_active
       allocate(exprs(j)%terms(n_active))
       if(n_active .gt. 0) &
            exprs(j)%terms(1:n_active) = terms_buf(1:n_active)
       call build_expression_string(exprs(j))
       deallocate(terms_buf)
    end do

    deallocate(x_grid, phi_vals)

  end function match_from_fastkan
!###############################################################################


!###############################################################################
  subroutine eval_bspline_at(x, knots, n_knots, degree, n_basis, &
       basis)
    !! Evaluate B-spline basis functions at a single point
    !!
    !! Uses the Cox-de Boor recursion to evaluate all n_basis
    !! B-spline basis functions of the given degree at point x.
    implicit none

    ! Arguments
    real(real32), intent(in) :: x
    !! Evaluation point
    real(real32), dimension(:), intent(in) :: knots
    !! Knot vector
    integer, intent(in) :: n_knots
    !! Number of knots
    integer, intent(in) :: degree
    !! B-spline degree
    integer, intent(in) :: n_basis
    !! Number of basis functions
    real(real32), intent(out) :: basis(n_basis)
    !! Output basis function values

    ! Local variables
    integer :: p, i_k, n_work
    real(real32) :: left, right, w1, w2
    real(real32), allocatable :: work(:,:)


    n_work = n_knots - 1
    allocate(work(n_work, 0:degree))
    work = 0.0_real32

    ! Degree 0
    do i_k = 1, n_work
       if(x .ge. knots(i_k) .and. x .lt. knots(i_k + 1))then
          work(i_k, 0) = 1.0_real32
       end if
    end do
    ! Handle right boundary
    if(x .ge. knots(n_knots))then
       work(n_work, 0) = 1.0_real32
    end if

    ! Recursion to higher degrees
    do p = 1, degree
       do i_k = 1, n_knots - p - 1
          w1 = 0.0_real32
          w2 = 0.0_real32
          left = knots(i_k + p) - knots(i_k)
          right = knots(i_k + p + 1) - knots(i_k + 1)
          if(left .gt. 0.0_real32) &
               w1 = (x - knots(i_k)) / left * work(i_k, p - 1)
          if(right .gt. 0.0_real32) &
               w2 = (knots(i_k + p + 1) - x) / right * &
               work(i_k + 1, p - 1)
          work(i_k, p) = w1 + w2
       end do
    end do

    basis(1:n_basis) = work(1:n_basis, degree)
    deallocate(work)

  end subroutine eval_bspline_at
!###############################################################################


!###############################################################################
  subroutine match_best_function(x, y, n, input_dim, &
       best_a, best_b, best_r2, best_desc)
    !! Find the best-matching known function for sampled data
    !!
    !! Tries multiple candidate functions and selects the one with
    !! highest R^2 using linear least-squares: y ~ a * f(x) + b.
    implicit none

    ! Arguments
    real(real32), intent(in) :: x(n)
    !! Grid points
    real(real32), intent(in) :: y(n)
    !! Sampled function values
    integer, intent(in) :: n
    !! Number of grid points
    integer, intent(in) :: input_dim
    !! Input dimension index (for description strings)
    real(real32), intent(out) :: best_a
    !! Best-fit scale coefficient
    real(real32), intent(out) :: best_b
    !! Best-fit offset
    real(real32), intent(out) :: best_r2
    !! Best R^2 value
    character(len=256), intent(out) :: best_desc
    !! Description of best-matching function

    ! Local variables
    real(real32), allocatable :: f(:)
    real(real32) :: a, b, r2
    real(real32), parameter :: pi = 4.0_real32 * atan(1.0_real32)
    character(len=8) :: x_str
    integer :: g


    allocate(f(n))
    write(x_str, '(A,I0,A)') "x(", input_dim, ")"

    best_r2 = -1.0_real32
    best_a = 0.0_real32
    best_b = 0.0_real32
    best_desc = " "

    ! Candidate: x (linear)
    f = x
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = trim(x_str)
    end if

    ! Candidate: x^2
    f = x * x
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = trim(x_str) // "^2"
    end if

    ! Candidate: x^3
    f = x * x * x
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = trim(x_str) // "^3"
    end if

    ! Candidate: sin(x)
    f = sin(x)
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "sin(" // trim(x_str) // ")"
    end if

    ! Candidate: sin(pi*x)
    do g = 1, n
       f(g) = sin(pi * x(g))
    end do
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "sin(pi*" // trim(x_str) // ")"
    end if

    ! Candidate: sin(2*pi*x)
    do g = 1, n
       f(g) = sin(2.0_real32 * pi * x(g))
    end do
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "sin(2*pi*" // trim(x_str) // ")"
    end if

    ! Candidate: cos(x)
    f = cos(x)
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "cos(" // trim(x_str) // ")"
    end if

    ! Candidate: cos(pi*x)
    do g = 1, n
       f(g) = cos(pi * x(g))
    end do
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "cos(pi*" // trim(x_str) // ")"
    end if

    ! Candidate: exp(x)
    f = exp(x)
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "exp(" // trim(x_str) // ")"
    end if

    ! Candidate: |x|
    f = abs(x)
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "|" // trim(x_str) // "|"
    end if

    ! Candidate: tanh(x)
    f = tanh(x)
    call fit_and_score(y, f, n, a, b, r2)
    if(r2 .gt. best_r2)then
       best_a = a; best_b = b; best_r2 = r2
       best_desc = "tanh(" // trim(x_str) // ")"
    end if

    deallocate(f)

  end subroutine match_best_function
!###############################################################################


!###############################################################################
  subroutine fit_and_score(y, f, n, a, b, r2)
    !! Fit y ~ a * f + b by least squares and compute R^2
    implicit none

    ! Arguments
    real(real32), intent(in) :: y(n)
    !! Target values
    real(real32), intent(in) :: f(n)
    !! Candidate function values
    integer, intent(in) :: n
    !! Number of points
    real(real32), intent(out) :: a
    !! Fitted scale
    real(real32), intent(out) :: b
    !! Fitted offset
    real(real32), intent(out) :: r2
    !! Coefficient of determination

    ! Local variables
    real(real32) :: sum_f, sum_y, sum_ff, sum_fy
    real(real32) :: mean_y, ss_tot, ss_res, denom
    real(real32) :: rn
    integer :: i


    rn = real(n, real32)
    sum_f = 0.0_real32
    sum_y = 0.0_real32
    sum_ff = 0.0_real32
    sum_fy = 0.0_real32
    do i = 1, n
       sum_f = sum_f + f(i)
       sum_y = sum_y + y(i)
       sum_ff = sum_ff + f(i) * f(i)
       sum_fy = sum_fy + f(i) * y(i)
    end do

    denom = rn * sum_ff - sum_f * sum_f
    if(abs(denom) .lt. 1.0E-30_real32)then
       a = 0.0_real32
       b = sum_y / rn
       r2 = 0.0_real32
       return
    end if

    a = (rn * sum_fy - sum_f * sum_y) / denom
    b = (sum_y - a * sum_f) / rn

    mean_y = sum_y / rn
    ss_tot = 0.0_real32
    ss_res = 0.0_real32
    do i = 1, n
       ss_tot = ss_tot + (y(i) - mean_y) ** 2
       ss_res = ss_res + (y(i) - a * f(i) - b) ** 2
    end do

    if(ss_tot .lt. 1.0E-30_real32)then
       r2 = 1.0_real32
    else
       r2 = 1.0_real32 - ss_res / ss_tot
    end if

  end subroutine fit_and_score
!###############################################################################

end module athena__symbolic
