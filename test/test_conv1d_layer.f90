program test_conv1d_layer
  use athena, only: &
       conv1d_layer_type, &
       input_layer_type, &
       base_layer_type, &
       learnable_layer_type
  use athena__conv1d_layer, only: read_conv1d_layer
  use coreutils, only: real32
  use diffstruc, only: array_type, operator(+), operator(-)
  implicit none

  class(base_layer_type), allocatable, target :: conv_layer
  class(base_layer_type), allocatable :: conv_layer1, conv_layer2
  class(base_layer_type), allocatable :: input_layer, read_layer
  integer, parameter :: num_filters = 32, kernel_size = 3
  integer :: unit
  type(array_type) :: input(1,1)
  type(array_type), pointer :: output
  real, parameter :: tol = 1.E-7
  logical :: success = .true.

  real, allocatable, dimension(:) :: params


!-------------------------------------------------------------------------------
! set up layer
!-------------------------------------------------------------------------------
  conv_layer = conv1d_layer_type( &
       num_filters = num_filters, &
       kernel_size = kernel_size &
  )

  !! check layer name
  if(.not. conv_layer%name .eq. 'conv1d')then
     success = .false.
     write(0,*) 'conv1d layer has wrong name'
  end if

  !! check layer type
  select type(conv_layer)
  type is(conv1d_layer_type)
     !! check default layer transfer/activation function
     if(conv_layer%activation%name .ne. 'none')then
        success = .false.
        write(0,*) 'conv1d layer has wrong activation: '//conv_layer%activation%name
     end if

     !! check number of filters
     if(conv_layer%num_filters .ne. num_filters)then
        success = .false.
        write(0,*) 'conv1d layer has wrong num_filters'
     end if

     !! check kernel size
     if(any(conv_layer%knl .ne. kernel_size))then
        success = .false.
        write(0,*) 'conv1d layer has wrong kernel_size'
     end if

     !! check input shape allocated
     if(allocated(conv_layer%input_shape))then
        success = .false.
        write(0,*) 'conv1d layer shape should not be allocated yet'
     end if
  class default
     success = .false.
     write(0,*) 'conv1d layer has wrong type'
  end select


!!!-----------------------------------------------------------------------------
!!! check layer input and output shape based on input layer
!!! conv1d layer: 32 x 32 pixel image, 3 channels
!!!-----------------------------------------------------------------------------
  input_layer = input_layer_type([32,3])
  call conv_layer%init(input_layer%input_shape)
  select type(conv_layer)
  type is(conv1d_layer_type)
     if(any(conv_layer%input_shape .ne. [32,3]))then
        success = .false.
        write(0,*) 'conv1d layer has wrong input_shape'
     end if
     if(any(conv_layer%output_shape .ne. [30,num_filters]))then
        success = .false.
        write(0,*) 'conv1d layer has wrong output_shape'
     end if
  end select


!!!-----------------------------------------------------------------------------
!!! test forward pass and check expected output
!!! use existing layer
!!!-----------------------------------------------------------------------------
  !! initialise sample input
  !! conv1d layer: 3 element signal, 1 channel, batch 1
  call input(1,1)%allocate(array_shape=[3,1,1], source = 0._real32)
  call input(1,1)%set_requires_grad(.true.)
  conv_layer = conv1d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
  )
  call conv_layer%init(input(1,1)%shape, batch_size=1)
  call conv_layer%set_ptrs()

  !! run forward pass
  call conv_layer%forward(input)
  output => conv_layer%output(1,1)

  !! check outputs have expected value
  if (any(abs(output%val(:,1) - 0.5) .gt. tol)) then
     success = .false.
     write(*,*) 'conv1d layer with zero input and sigmoid activation must'//&
          ' return outputs all equal to 0.5'
     write(*,*) output%val(:,1)
  end if


!-------------------------------------------------------------------------------
! check handling of layer parameters, gradients, and outputs
!-------------------------------------------------------------------------------
  select type(conv_layer)
  class is(learnable_layer_type)
     !! check layer parameter handling
     params = conv_layer%get_params()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv1d layer has wrong number of parameters'
     end if
     params = 1.E0
     call conv_layer%set_params(params)
     params = conv_layer%get_params()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv1d layer has wrong parameters'
     end if

     !! check layer gradient handling
     params = conv_layer%get_gradients()
     if(size(params) .eq. 0)then
        success = .false.
        write(0,*) 'conv1d layer has wrong number of gradients'
     end if
     params = 1.E0
     call conv_layer%set_gradients(params)
     params = conv_layer%get_gradients()
     if(any(abs(params - 1.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv1d layer has wrong gradients'
     end if
     call conv_layer%set_gradients(10.E0)
     params = conv_layer%get_gradients()
     if(any(abs(params - 10.E0).gt.tol))then
        success = .false.
        write(0,*) 'conv1d layer has wrong gradients'
     end if

     !! check layer output handling
     output => conv_layer%output(1,1)
     if(size(output%val,dim=1) .ne. product(conv_layer%output_shape))then
        success = .false.
        write(0,*) 'conv1d layer has wrong number of outputs'
     end if
     if(any(shape(output%val) .ne. [product(conv_layer%output_shape), 1]))then
        success = .false.
        write(0,*) 'conv1d layer has wrong number of outputs'
     end if
  end select


!-------------------------------------------------------------------------------
! check expected initialisation of kernel size and stride
!-------------------------------------------------------------------------------
  conv_layer = conv1d_layer_type( &
       kernel_size = [2], &
       stride = [2] &
  )
  select type(conv_layer)
  type is (conv1d_layer_type)
     if(any(conv_layer%knl .ne. 2))then
        success = .false.
        write(0,*) 'conv1d layer has wrong pool size'
     end if
     if(any(conv_layer%stp .ne. 2))then
        success = .false.
        write(0,*) 'conv1d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! check expected initialisation of kernel size and stride
!-------------------------------------------------------------------------------
  conv_layer = conv1d_layer_type( &
       kernel_size = [4], &
       stride = [4] &
  )
  select type(conv_layer)
  type is (conv1d_layer_type)
     if(any(conv_layer%knl .ne. 4))then
        success = .false.
        write(0,*) 'conv1d layer has wrong pool size'
     end if
     if(any(conv_layer%stp .ne. 4))then
        success = .false.
        write(0,*) 'conv1d layer has wrong stride size'
     end if
  end select


!-------------------------------------------------------------------------------
! check layer operations
!-------------------------------------------------------------------------------
  conv_layer1 = conv1d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
  )
  call conv_layer1%init(input_layer%input_shape, batch_size=1)
  conv_layer2 = conv1d_layer_type( &
       num_filters = 1, &
       kernel_size = 3, &
       activation_function = 'sigmoid' &
  )
  call conv_layer2%init(input_layer%input_shape, batch_size=1)
  select type(conv_layer1)
  type is(conv1d_layer_type)
     select type(conv_layer2)
     type is(conv1d_layer_type)
        conv_layer = conv_layer1 + conv_layer2
        select type(conv_layer)
        type is(conv1d_layer_type)
           !! check layer addition
           call compare_conv1d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)

           !! check layer reduction
           conv_layer = conv_layer1
           call conv_layer%reduce(conv_layer2)
           call compare_conv1d_layers(&
                conv_layer, conv_layer1, success, conv_layer2)
        class default
           success = .false.
           write(0,*) 'conv1d layer has wrong type'
        end select
     class default
        success = .false.
        write(0,*) 'conv1d layer has wrong type'
     end select
  class default
     success = .false.
     write(0,*) 'conv1d layer has wrong type'
  end select


!-------------------------------------------------------------------------------
! Test file I/O operations
!-------------------------------------------------------------------------------
  write(*,*) "Testing file I/O operations..."

  ! Create a temporary file for testing
  open(newunit=unit, file='test_conv1d_layer.tmp', &
       status='replace', action='write')

  ! Write layer to file
  write(unit,'("CONV1D")')
  call conv_layer%print_to_unit(unit)
  write(unit,'("END CONV1D")')
  close(unit)

  ! Read layer from file
  open(newunit=unit, file='test_conv1d_layer.tmp', &
       status='old', action='read')
  read(unit,*) ! Skip first line
  read_layer = read_conv1d_layer(unit)
  close(unit)

  ! Check that read layer has correct properties
  select type(read_layer)
  type is (conv1d_layer_type)
     if (.not. read_layer%name .eq. 'conv1d') then
        success = .false.
        write(0,*) 'read conv1d layer has wrong name'
     end if
  class default
     success = .false.
     write(0,*) 'read layer is not conv1d_layer_type'
  end select

  ! Clean up temporary file
  open(newunit=unit, file='test_conv1d_layer.tmp', status='old')
  close(unit, status='delete')


!-------------------------------------------------------------------------------
! Check for any failed tests
!-------------------------------------------------------------------------------
  write(*,*) "----------------------------------------"
  if(success)then
     write(*,*) 'test_conv1d_layer passed all tests'
  else
     write(0,*) 'test_conv1d_layer failed one or more tests'
     stop 1
  end if


contains

!-------------------------------------------------------------------------------
! compare two or three layers
!-------------------------------------------------------------------------------
  subroutine compare_conv1d_layers(layer1, layer2, success, layer3)
    type(conv1d_layer_type), intent(in) :: layer1, layer2
    logical, intent(inout) :: success
    type(conv1d_layer_type), optional, intent(in) :: layer3

    if(all(layer1%knl .ne. layer2%knl))then
       success = .false.
       write(0,*) 'conv1d layer has wrong kernel_size'
    end if
    if(layer1%num_filters .ne. layer2%num_filters)then
       success = .false.
       write(0,*) 'conv1d layer has wrong num_filters'
    end if
    if(layer1%activation%name .ne. 'sigmoid')then
       success = .false.
       write(0,*) 'conv1d layer has wrong transfer: '//&
            layer1%activation%name
    end if
    if(present(layer3))then
       if( &
            associated(layer1%params_array(1)%grad).and. &
            associated(layer2%params_array(1)%grad).and. &
            associated(layer3%params_array(1)%grad) &
       )then
          if(any(abs( &
               layer1%params_array(1)%grad%val - &
               layer2%params_array(1)%grad%val - &
               layer3%params_array(1)%grad%val &
          ).gt.1.E-6))then
             success = .false.
             write(0,*) 'conv1d layer has wrong gradients'
          end if
       end if

       if( &
            associated(layer1%params_array(2)%grad).and. &
            associated(layer2%params_array(2)%grad).and. &
            associated(layer3%params_array(2)%grad) &
       )then
          if(any(abs( &
               layer1%params_array(2)%grad%val - &
               layer2%params_array(2)%grad%val - &
               layer3%params_array(2)%grad%val &
          ).gt.1.E-6))then
             success = .false.
             write(0,*) 'conv1d layer has wrong bias gradients'
          end if
       end if
    end if

  end subroutine compare_conv1d_layers

end program test_conv1d_layer
