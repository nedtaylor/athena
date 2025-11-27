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
    character(20) :: node_name, input_name, tmp_input_name
    !! Node name
    character(:), allocatable :: suffix
    !!! Suffix for input names

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
       case('msgp')
          layer_name = 'GNNLayer'
       case default
          layer_name = 'Unknown'
       end select

       write(unit, '(A)') '  node {'
       write(unit, '(A,A,A)') '    name: "', trim(node_name), '"'
       write(unit, '(A,A,A)') '    op_type: "', trim(layer_name), '"'

       ! Write input connections
       if(all(network%auto_graph%adjacency(:,network%vertex_order(i)).eq.0)) then
          cycle
          ! write(unit, '(A,I0,A)') '    input: "input_',network%model(idx)%layer%id,'"'
       else
          do j = 1, network%auto_graph%num_vertices
             if(network%auto_graph%adjacency(j,network%vertex_order(i)).eq.0) cycle
             if(all(network%auto_graph%adjacency(:,j).eq.0))then
                write(input_name,'("input_",I0)') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
                suffix = ''
             else
                write(input_name,'("node_",I0)') &
                     network%model(network%auto_graph%vertex(j)%id)%layer%id
                suffix = '_output'
             end if
             if(network%model(idx)%layer%use_graph_input)then
                write(tmp_input_name,'(A,A,A)') &
                     trim(adjustl(input_name)), '_vertex', suffix
                write(unit,'(4X,"input: """,A,"""")') trim(adjustl(tmp_input_name))
                if(network%model(idx)%layer%input_shape(2) .gt. 0)then
                   write(tmp_input_name,'(A,A,A)') &
                        trim(adjustl(input_name)), '_edge', suffix
                   write(unit,'(4X,"input: """,A,"""")') trim(adjustl(tmp_input_name))
                end if
             else
                write(unit,'(4X,"input: """,A,A,"""")') &
                     trim(adjustl(input_name)), suffix
             end if
          end do
       end if
       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          if(layer%transfer%name.ne."none")then
             suffix = '_pre_function'
          else
             suffix = ''
          end if
          do j = 1, size(layer%weight_shape, dim=2)
             write(unit, '(4X,"input: ""node_",I0,"_weight",I0,"""")') &
                  network%model(idx)%layer%id, j
             if(layer%has_bias) then
                write(unit, '(4X,"input: ""node_",I0,"_bias",I0,"""")') &
                     network%model(idx)%layer%id, j
             end if
          end do
       class default
          suffix = ''
       end select

       ! Write output
       if(network%model(idx)%layer%use_graph_output) then
          write(unit, '(4X,"output: ""node_",I0,"_vertex_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
          write(unit, '(4X,"output: ""node_",I0,"_edge_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
       else
          write(unit, '(4X,"output: ""node_",I0,"_output",A,"""")') &
               network%model(idx)%layer%id, trim(adjustl(suffix))
       end if

       call write_onnx_attributes(unit, network%model(idx)%layer)

       write(unit, '(A)') '  }'
       write(unit, '(A)') ''

       select type(layer => network%model(idx)%layer)
       class is(learnable_layer_type)
          call write_onnx_initializers(unit, layer, prefix = trim(node_name) )
          if(layer%transfer%name.ne."none")then
             if(layer%use_graph_output)then
                call write_onnx_function( &
                     unit, layer%transfer%name, &
                     prefix = trim(node_name)//'_vertex' &
                )
                if(network%model(idx)%layer%input_shape(2) .gt. 0)then
                   call write_onnx_function( &
                        unit, layer%transfer%name, &
                        prefix = trim(node_name)//'_edge' &
                   )
                end if
             else
                call write_onnx_function( &
                     unit, layer%transfer%name, &
                     prefix = trim(node_name) &
                )
             end if
          end if
       end select
    end do


    ! write all layer output shapes
    do i = 1, network%auto_graph%num_vertices
       idx = network%auto_graph%vertex(network%vertex_order(i))%id
       if(.not.allocated(network%model(idx)%layer%output_shape)) cycle
       if(network%model(idx)%layer%use_graph_output) then
          write(node_name, '("node_",I0,"_vertex_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "value_info", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%output_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%output_shape(2) .gt. 0)then
             write(node_name, '("node_",I0,"_edge_output")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "value_info", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%output_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("node_",I0,"_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "value_info", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%output_shape, &
               network%batch_size &
          )
       end if
    end do

    ! Write inputs
    write(unit, '(A)') '  # Inputs'
    do i = 1, size(network%root_vertices, dim=1)
       idx = network%root_vertices(i)
       if(network%model(idx)%layer%use_graph_output) then
          write(node_name, '("input_",I0,"_vertex")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "input", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%input_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%input_shape(2) .gt. 0)then
             write(node_name, '("input_",I0,"_edge")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "input", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%input_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("input_",I0)') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "input", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%input_shape, &
               network%batch_size &
          )
       end if
    end do

    ! Write outputs
    write(unit, '(A)') '  # Outputs'
    do i = 1, size(network%output_vertices, dim=1)
       idx = network%output_vertices(i)
       if(network%model(idx)%layer%use_graph_output) then
          write(node_name, '("node_",I0,"_vertex_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "output", &
               trim(adjustl(node_name)), &
               [ network%model(idx)%layer%output_shape(1) ], &
               network%batch_size &
          )
          if(network%model(idx)%layer%output_shape(2) .gt. 0)then
             write(node_name, '("node_",I0,"_edge_output")') network%model(idx)%layer%id
             call write_onnx_tensor( &
                  unit, &
                  "output", &
                  trim(adjustl(node_name)), &
                  [ network%model(idx)%layer%output_shape(2) ], &
                  network%batch_size &
             )
          end if
       else
          write(node_name, '("node_",I0,"_output")') network%model(idx)%layer%id
          call write_onnx_tensor( &
               unit, &
               "output", &
               trim(adjustl(node_name)), &
               network%model(idx)%layer%output_shape, &
               network%batch_size &
          )
       end if
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
  module subroutine write_onnx_tensor(unit, tensor_type, name, output_shape, batch_size)
    !! Write ONNX value info for a layer
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    character(*), intent(in) :: tensor_type
    !! Type of the tensor
    character(*), intent(in) :: name
    !! Name of the layer
    integer, intent(in), dimension(:) :: output_shape
    !! Shape of the layer output
    integer, intent(in) :: batch_size
    !! Batch size for the output

    ! Local variables
    integer :: i
    !! Loop index


    write(unit, '(A,A,A)') '  ',tensor_type,' {'
    write(unit, '(A,A,A)') '    name: "',name,'"'
    write(unit, '(A)') '    type {'
    write(unit, '(A)') '      tensor_type {'
    write(unit, '(A)') '        elem_type: 1'
    write(unit, '(A)') '        shape {'
    write(unit, '(A,I0)') '          dim { dim_value: ', max(1,batch_size)
    write(unit, '(A)') '          }'
    do i = size(output_shape), 1, -1
       write(unit, '(A,I0)') '          dim { dim_value: ', output_shape(i)
       write(unit, '(A)') '          }'
    end do
    write(unit, '(A)') '        }'
    write(unit, '(A)') '      }'
    write(unit, '(A)') '    }'
    write(unit, '(A)') '  }'

  end subroutine write_onnx_tensor
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
    integer :: i, j, num_params, num_params_old
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
          do j = size(layer%weight_shape, 1), 1, -1
             write(unit, '(4X,A,I0)') 'dims: ', layer%weight_shape(j,i)
          end do
          num_params = product(layer%weight_shape(:, i))

          write(unit, '(4X,"float_data: [ ",F0.6)', advance='no') &
               layer%params(num_params_old + 1)
          do j = 2, num_params, 1
             write(unit, '(", ",F0.6)', advance='no') layer%params(num_params_old + j)
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
  subroutine write_onnx_function(unit, function_name, prefix)
    !! Write ONNX function definition
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! File unit
    character(*), intent(in) :: function_name
    !! Name of the function
    character(*), intent(in) :: prefix
    !! Optional prefix for the function name

    ! Local variables
    character(256) :: full_name
    !! Full name of the function
    character(:), allocatable :: function_name_camel_case
    !! Camel case version of the function name

    function_name_camel_case = &
         to_camel_case(trim(adjustl(function_name)), capitalise_first_letter = .true.)
    if(prefix .eq. "")then
       full_name = trim(adjustl(function_name))
    else
       full_name = trim(prefix) // "_" // trim(adjustl(function_name))
    end if


    write(unit, '(A)') '  node {'
    write(unit, '(A,A,A)') '    name: "', trim(full_name), '"'
    write(unit, '(A,A,A)') '    op_type: "', trim(function_name_camel_case), '"'
    write(unit, '(A,A,A)') '    input: "', trim(prefix), '_output_pre_function"'
    write(unit, '(A,A,A)') '    output: "', trim(prefix), '_output"'
    write(unit, '(A)') '  }'
    write(unit, '(A)') ''

  end subroutine write_onnx_function
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
          itmp1 = icount(attributes(i)%value)
          select case(type_lw)
          case('ints','int')
             allocate(ivar_list(itmp1))
             read(attributes(i)%value,*) ivar_list
             do j = 1, size(ivar_list)
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
