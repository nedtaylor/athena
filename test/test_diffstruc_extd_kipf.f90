program test_diffstruc_extd_kipf
  use coreutils, only: real32
  use athena__diffstruc_extd, only: kipf_propagate
  use diffstruc, only: array_type
  implicit none

  logical :: success
  type(array_type), target :: vertex_features
  type(array_type), target :: helper_operand
  type(array_type), target :: upstream_grad
  type(array_type), target :: second_upstream_grad
  type(array_type), pointer :: output
  type(array_type) :: reverse_partial
  type(array_type) :: forward_partial
  real(real32) :: partial_vals(2,2)
  integer :: adj_ia(3)
  integer :: adj_ja(2,2)

  success = .true.

  write(*,'("Testing kipf_propagate on identity graph")')

  call vertex_features%allocate([2, 2])
  vertex_features%val(:,1) = [1._real32, 2._real32]
  vertex_features%val(:,2) = [3._real32, 4._real32]
  call vertex_features%set_requires_grad(.true.)
  vertex_features%is_temporary = .false.

  adj_ia = [1, 2, 3]
  adj_ja(:,1) = [1, 0]
  adj_ja(:,2) = [2, 0]

  output => kipf_propagate(vertex_features, adj_ia, adj_ja)
  if(any(abs(output%val - vertex_features%val) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'kipf_propagate returned the wrong forward values'
  end if

  call output%grad_reverse(reset_graph=.true.)

  if(.not. associated(vertex_features%grad) .or. &
       any(abs(vertex_features%grad%val - 1._real32) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'kipf_propagate returned the wrong gradient'
  end if

  call output%nullify_graph()
  nullify(output)
  call vertex_features%deallocate()

  write(*,'("Testing Kipf partial-function paths")')

  call vertex_features%allocate([2, 2])
  vertex_features%val(:,1) = [1._real32, 2._real32]
  vertex_features%val(:,2) = [3._real32, 4._real32]
  call vertex_features%set_requires_grad(.true.)
  vertex_features%is_temporary = .false.

  call helper_operand%allocate([2, 2])
  helper_operand%val = 0._real32

  call upstream_grad%allocate([2, 2])
  upstream_grad%val(:,1) = [5._real32, 6._real32]
  upstream_grad%val(:,2) = [7._real32, 8._real32]

  call second_upstream_grad%allocate([2, 2])
  second_upstream_grad%val(:,1) = [2._real32, 1._real32]
  second_upstream_grad%val(:,2) = [4._real32, 3._real32]

  output => kipf_propagate(vertex_features, adj_ia, adj_ja)
  output%right_operand => helper_operand

  reverse_partial = output%get_partial_left(upstream_grad)
  if(any(abs(reverse_partial%val - upstream_grad%val) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'kipf partial function returned wrong reverse values'
  end if

  forward_partial = reverse_partial%get_partial_left(second_upstream_grad)
  if(any(abs(forward_partial%val - second_upstream_grad%val) .gt. &
       1.e-6_real32))then
     success = .false.
     write(0,*) 'reverse kipf partial function returned wrong values'
  end if

  call reverse_partial%get_partial_left_val(second_upstream_grad%val, &
       partial_vals)
  if(any(abs(partial_vals - second_upstream_grad%val) .gt. 1.e-6_real32))then
     success = .false.
     write(0,*) 'reverse kipf partial value function returned wrong values'
  end if

  call output%nullify_graph()
  nullify(output)

  if(success)then
     write(*,*) 'test_kipf passed all tests'
  else
     write(0,*) 'test_kipf failed one or more tests'
     stop 1
  end if

end program test_diffstruc_extd_kipf
