module athena__onnx
  !! Module containing the types and interfaces for ONNX operations
  use athena__constants, only: real32
  use athena__network, only: network_type
  use athena__base_layer, only: base_layer_type, learnable_layer_type
  use athena__misc_types, only: attributes_type
  use athena__misc, only: to_lower, to_upper, to_camel_case, icount
  implicit none


  private

  public :: write_onnx



contains

!###############################################################################
  subroutine write_onnx(file, network)
    !! Export the network to ONNX format
    implicit none

    ! Arguments
    class(network_type), intent(in) :: network
    !! Instance of network
    character(*), intent(in) :: file
    !! File to export the network to

    ! Local variables
    integer :: unit, i, j, idx
    !! Unit number and loop indices
    character(256) :: layer_name
    !! Layer name for ONNX
    character(20) :: node_name
    !! Node name

    open(newunit=unit, file=file, status='replace')

    ! Write ONNX header
    write(unit, '(A)') 'ir_version: 8'
    write(unit, '(A)') 'producer_name: "Athena"'
    write(unit, '(A)') 'producer_version: "1.0"'
    write(unit, '(A)') 'domain: "ai.onnx"'
    write(unit, '(A)') 'model_version: 1'
    write(unit, '(A)') 'doc_string: "Athena neural network model"'
    write(unit, '(A)') ''

    ! Write graph definition
    write(unit, '(A)') 'graph {'
    write(unit, '(A)') '  name: "athena_network"'
    write(unit, '(A)') ''

    ! Write nodes (layers)
    write(unit, '(A)') '  # Nodes'
    do i = 1, network%auto_graph%num_vertices
       idx = network%auto_graph%vertex(network%vertex_order(i))%id
       write(node_name, '("node_", I0)') network%model(idx)%layer%id

       select case(trim(network%model(idx)%layer%type))
       case('inpt')
          layer_name = 'Input'
          cycle
       case('full')
          layer_name = 'MatMul'
       case('conv')
          layer_name = 'Conv'
       case('pool')
          layer_name = to_camel_case( &
               trim(adjustl(network%model(idx)%layer%subtype))//"_"//&
               trim(adjustl(network%model(idx)%layer%type)), &
               capitalise_first_letter = .true. &
          )
       case('actv')
          layer_name = to_camel_case( &
               adjustl(network%model(idx)%layer%subtype), &
               capitalise_first_letter = .true. &
          )
       case('flat')
          layer_name = 'Flatten'
       case('batc')
          layer_name = 'BatchNormalization'
       case('drop')
          layer_name = 'Dropout'
       case default
          layer_name = 'Unknown'
       end select

       write(unit, '(A)') '  node {'
       write(unit, '(A,A,A)') '    name: "', trim(node_name), '"'
       write(unit, '(A,A,A)') '    op_type: "', trim(layer_name), '"'

       ! Write input connections
       if(all(network%auto_graph%adjacency(:,network%vertex_order(i)).eq.0)) then
          write(unit, '(A,I0,A)') '    input: "input',network%model(idx)%layer%id,'"'
       else
          do j = 1, network%auto_graph%num_vertices
             if(network%auto_graph%adjacency(j,network%vertex_order(i)).eq.0) cycle
             if(all(network%auto_graph%adjacency(:,j).eq.0))then
                write(unit, '(4X,"input: ""input_",I0,"""")') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
             else
                write(unit, '(4X,"input: ""node_",I0,"_output""")') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
             end if
          end do
       end if
       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          do j = 1, size(layer%weight_shape, dim=2)
             write(unit, '(4X,"input: ""node_",I0,"_weight",I0,"""")') &
                  network%model(idx)%layer%id, j
             if(layer%has_bias) then
                write(unit, '(4X,"input: ""node_",I0,"_bias",I0,"""")') &
                     network%model(idx)%layer%id, j
             end if
          end do
       end select

       ! Write output
       write(unit, '(4X,"output: ""node_",I0,"_output""")') network%model(idx)%layer%id

       call write_onnx_attributes(unit, network%model(idx)%layer)

       write(unit, '(A)') '  }'
       write(unit, '(A)') ''

       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          call write_onnx_initializers(unit, layer, prefix = trim(node_name) )
       end select
    end do

    ! Write inputs
    do i = 1, size(network%root_vertices, dim=1)
       idx = network%root_vertices(i)
       write(unit, '(A)') '  # Inputs'
       write(unit, '(A)') '  input {'
       write(unit, '(A,I0,A)') '    name: "input_',i,'"'
       write(unit, '(A)') '    type {'
       write(unit, '(A)') '      tensor_type {'
       write(unit, '(A)') '        elem_type: 1'  ! FLOAT
       write(unit, '(A)') '        shape {'
       if (allocated(network%model(idx)%layer%input_shape)) then
          do j = 1, size(network%model(idx)%layer%input_shape)
             write(unit, '(A,I0)') '          dim { dim_value: ', &
                  network%model(idx)%layer%input_shape(j)
             write(unit, '(A)') '          }'
          end do
       end if
       write(unit, '(A)') '        }'
       write(unit, '(A)') '      }'
       write(unit, '(A)') '    }'
       write(unit, '(A)') '  }'
       write(unit, '(A)') ''
    end do

    ! Write outputs
    do i = 1, size(network%output_vertices, dim=1)
       idx = network%output_vertices(i)
       write(unit, '(A)') '  # Outputs'
       write(unit, '(A)') '  output {'
       write(unit, '(4X,"name: ""node_",I0,"_output""")') network%model(idx)%layer%id
       write(unit, '(A)') '    type {'
       write(unit, '(A)') '      tensor_type {'
       write(unit, '(A)') '        elem_type: 1'  ! FLOAT
       write(unit, '(A)') '        shape {'
       if (allocated(network%model(idx)%layer%output_shape)) then
          do j = 1, size(network%model(idx)%layer%output_shape)
             write(unit, '(A,I0)') '          dim { dim_value: ', &
                  network%model(idx)%layer%output_shape(j)
             write(unit, '(A)') '          }'
          end do
       end if
       write(unit, '(A)') '        }'
       write(unit, '(A)') '      }'
       write(unit, '(A)') '    }'
       write(unit, '(A)') '  }'
    end do

    write(unit, '(A)') '}'

    ! Write ONNX footer
    write(unit, '(A)') 'opset_import {'
    write(unit, '(A)') '  domain: "ai.onnx"'
    write(unit, '(A,I0)') '  version: ', 13  ! ONNX version
    write(unit, '(A)') '}'

    close(unit)

  end subroutine write_onnx
!###############################################################################


!###############################################################################
  module subroutine write_onnx_initializers(unit, layer, prefix)
    !! Write ONNX initializers (weights and biases)
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    class(learnable_layer_type), intent(in) :: layer
    !! Instance of a layer
    character(*), intent(in) :: prefix
    !! Optional prefix for weight and bias names

    ! Local variables
    integer :: i, j, k, num_params, num_params_old
    !! Loop indices
    character(20) :: name
    !! Names for weights and biases


    if(allocated(layer%params))then
       num_params_old = 0
       do i = 1, size(layer%weight_shape, 2)
          write(name, '(A,A,I0)') trim(prefix), '_weight', i
          write(unit, '(2X,A)') 'initializer {'
          write(unit, '(4X,"name: """,A,"""")') trim(name)
          write(unit, '(4X,A)') 'data_type: 1'  ! FLOAT
          do j = 1, size(layer%weight_shape, 1)
             write(unit, '(4X,A,I0)') 'dims: ', layer%weight_shape(j,i)
          end do
          num_params = product(layer%weight_shape(:, i))

          write(unit, '(4X,"float_data: [ ",F0.6)', advance='no') &
               layer%params(num_params_old + 1)
          do j = num_params_old + 2, num_params + num_params_old, 1
             write(unit, '(", ",F0.6)', advance='no') layer%params(j)
          end do
          write(unit, '(A)') ' ]'
          write(unit, '(A)') '  }'
          write(unit, '(A)') ''

          num_params_old = num_params_old + num_params

          if(layer%has_bias)then
             write(name, '(A,A,I0)') trim(prefix), '_bias', i
             write(unit, '(2X,A)') 'initializer {'
             write(unit, '(4X,"name: """,A,"""")') trim(name)
             write(unit, '(4X,A)') 'data_type: 1'  ! FLOAT
             write(unit, '(4X,A,I0)') 'dims: ', layer%bias_shape(i)
             num_params = layer%bias_shape(i)

             write(unit, '(4X,"float_data: [ ",F0.6)', advance='no') &
                  layer%params(num_params_old + 1)
             do j = num_params_old + 2, num_params + num_params_old, 1
                write(unit, '(", ",F0.6)', advance='no') layer%params(j)
             end do
             write(unit, '(A)') ' ]'
             write(unit, '(A)') '  }'
             write(unit, '(A)') ''

             num_params_old = num_params_old + layer%bias_shape(i)
          end if
       end do
    end if

  end subroutine write_onnx_initializers
!###############################################################################


!###############################################################################
  subroutine write_onnx_attributes(unit, layer)
    !! Write ONNX attributes for a layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    class(base_layer_type), intent(in) :: layer
    !! Instance of a layer

    ! Local variables
    integer :: i, j, itmp1
    !! Loop index
    type(attributes_type), allocatable, dimension(:) :: attributes
    character(:), allocatable :: type_lw, type_up
    integer, allocatable, dimension(:) :: ivar_list
    real(real32), allocatable, dimension(:) :: rvar_list


    attributes = layer%get_attributes()
    if(allocated(attributes).and. size(attributes) .gt. 0) then
       do i = 1, size(attributes)
          write(unit, '(4X,A)') 'attribute {'
          write(unit, '(6X,"name: """,A,"""")') trim(attributes(i)%name)
          ! determine whether the attribute is a list or a single value
          type_lw = to_lower(trim(adjustl(attributes(i)%type)))
          type_up = to_upper(trim(adjustl(attributes(i)%type)))
          write(*,*) 'Attribute type: ', trim(type_lw)
          itmp1 = icount(attributes(i)%value)
          select case(type_lw)
          case('ints','int')
             allocate(ivar_list(itmp1))
             read(attributes(i)%value,*) ivar_list
             do j = 1, size(ivar_list)
                write(*,*) ivar_list(j)
                write(unit, '(6X,A,": ",I0)') type_lw, ivar_list(j)
             end do
             deallocate(ivar_list)
          case('floats','float')
             allocate(rvar_list(itmp1))
             read(attributes(i)%value,*) rvar_list
             do j = 1, size(rvar_list), 1
                write(unit, '(6X,A,": ",F0.6)') type_lw, rvar_list(j)
             end do
             deallocate(rvar_list)
          case('strings','string')
          case default
             write(unit, '(6X,A,": ",A)') trim(adjustl(attributes(i)%type)), &
                  trim(adjustl(attributes(i)%value))
          end select
          write(unit,'(6X,"type: ",A)') type_up
          write(unit,'(4X,"}")')
       end do
    end if

  end subroutine write_onnx_attributes
!###############################################################################

end module athena__onnx
