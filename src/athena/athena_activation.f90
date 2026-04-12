module athena__activation
  !! Module containing the activation function setup
  use coreutils, only: stop_program, to_lower
  use athena__misc_types, only: base_actv_type, onnx_attribute_type
  use athena__activation_gaussian, only: gaussian_actv_type, &
       create_from_onnx_gaussian_activation
  use athena__activation_linear, only: linear_actv_type, &
       create_from_onnx_linear_activation
  use athena__activation_piecewise, only: piecewise_actv_type, &
       create_from_onnx_piecewise_activation
  use athena__activation_relu, only: relu_actv_type, &
       create_from_onnx_relu_activation
  use athena__activation_leaky_relu, only: leaky_relu_actv_type, &
       create_from_onnx_leaky_relu_activation
  use athena__activation_sigmoid, only: sigmoid_actv_type, &
       create_from_onnx_sigmoid_activation
  use athena__activation_softmax, only: softmax_actv_type, &
       create_from_onnx_softmax_activation
  use athena__activation_swish, only: swish_actv_type, &
       create_from_onnx_swish_activation
  use athena__activation_tanh, only: tanh_actv_type, &
       create_from_onnx_tanh_activation
  use athena__activation_none, only: none_actv_type, &
       create_from_onnx_none_activation
  use athena__activation_selu, only: selu_actv_type, &
       create_from_onnx_selu_activation
  implicit none


  private

  public :: activation_setup
  public :: list_of_onnx_activation_creators
  public :: allocate_list_of_onnx_activation_creators
  public :: read_activation


  type :: onnx_create_actv_container
     !! Type containing information needed to create an activation from ONNX
     character(20) :: name
     !! Name of the layer
     procedure(create_from_onnx_activation), nopass, pointer :: create_ptr => null()
     !! Pointer to the specific activation creation function
  end type onnx_create_actv_container
  type(onnx_create_actv_container), dimension(:), allocatable :: &
       list_of_onnx_activation_creators
  !! List of activation names and their associated ONNX creation functions

  interface
     module function create_from_onnx_activation(attributes) result(activation)
       !! Function to create an activation function from ONNX attributes
       type(onnx_attribute_type), dimension(:), intent(in) :: attributes
       !! Array of ONNX attributes
       class(base_actv_type), allocatable :: activation
       !! Resulting activation function
     end function create_from_onnx_activation
  end interface



contains

!###############################################################################
  function activation_setup(input, error) result(activation)
    !! Setup the desired activation function
    implicit none

    ! Arguments
    class(*), intent(in) :: input
    !! Name of the activation function or activation object
    class(base_actv_type), allocatable :: activation
    !! Activation function object
    integer, optional, intent(out) :: error
    !! Error code

    ! Local variables
    character(256) :: err_msg
    !! Error message


    !---------------------------------------------------------------------------
    ! select desired activation function
    !---------------------------------------------------------------------------
    select type(input)
    class is(base_actv_type)
       activation = input
    type is(character(*))
       select case(trim(to_lower(input)))
       case("gaussian")
          activation = gaussian_actv_type()
       case ("linear")
          activation = linear_actv_type()
       case ("piecewise")
          activation = piecewise_actv_type()
       case ("relu")
          activation = relu_actv_type()
       case ("leaky_relu")
          activation = leaky_relu_actv_type()
       case ("sigmoid")
          activation = sigmoid_actv_type()
       case ("softmax")
          activation = softmax_actv_type()
       case("swish")
          activation = swish_actv_type()
       case ("tanh")
          activation = tanh_actv_type()
       case ("none")
          activation = none_actv_type()
       case ("selu")
          activation = selu_actv_type()
       case ("silu")
          activation = swish_actv_type()
       case default
          if(present(error))then
             error = -1
             return
          else
             write(err_msg,'("Incorrect activation name given ''",A,"''")') &
                  trim(to_lower(input))
             call stop_program(trim(err_msg))
             write(*,*) "BB"
             return
          end if
       end select
    class default
       if(present(error))then
          error = -1
          return
       else
          write(err_msg,'("Unknown input type given for activation setup")')
          call stop_program(trim(err_msg))
          return
       end if
    end select

  end function activation_setup
!###############################################################################


!###############################################################################
  subroutine allocate_list_of_onnx_activation_creators(addit_list)
    !! Allocate and populate the list of ONNX activation creation functions
    implicit none

    ! Arguments
    type(onnx_create_actv_container), dimension(:), intent(in), optional :: &
         addit_list

    if(.not.allocated(list_of_onnx_activation_creators)) &
         allocate(list_of_onnx_activation_creators(0))
    list_of_onnx_activation_creators = [ &
         onnx_create_actv_container('gaussian', create_from_onnx_gaussian_activation), &
         onnx_create_actv_container('leaky_relu', &
              create_from_onnx_leaky_relu_activation), &
         onnx_create_actv_container('linear', create_from_onnx_linear_activation), &
         onnx_create_actv_container('none', create_from_onnx_none_activation), &
         onnx_create_actv_container('piecewise', &
              create_from_onnx_piecewise_activation), &
         onnx_create_actv_container('relu', create_from_onnx_relu_activation), &
         onnx_create_actv_container('selu', create_from_onnx_selu_activation), &
         onnx_create_actv_container('sigmoid', create_from_onnx_sigmoid_activation), &
         onnx_create_actv_container('silu', create_from_onnx_swish_activation), &
         onnx_create_actv_container('softmax', create_from_onnx_softmax_activation), &
         onnx_create_actv_container('swish', create_from_onnx_swish_activation), &
         onnx_create_actv_container('tanh', create_from_onnx_tanh_activation) &
    ]
    if(present(addit_list))then
       list_of_onnx_activation_creators = [ &
            list_of_onnx_activation_creators, addit_list &
       ]
    end if

  end subroutine allocate_list_of_onnx_activation_creators
!###############################################################################


!###############################################################################
  function read_activation_attributes(unit, iline) result(attributes)
    use coreutils, only: stop_program, to_lower
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number for input file
    integer, intent(inout), optional :: iline
    !! Indicator for inline reading

    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes

    ! Local variables
    integer :: i
    !! Loop variable
    character(256) :: buffer
    !! Buffer for reading lines
    character(256) :: err_msg
    !! Error message
    character(20) :: attr_name
    !! Attribute name
    character(20) :: attr_value
    !! Attribute value as string
    integer :: stat
    !! I/O status
    integer :: eq_pos
    !! Position of equals sign
    integer :: n_attrs
    !! Number of attributes
    type(onnx_attribute_type), allocatable, dimension(:) :: temp_attrs
    !! Temporary array for growing attributes
    character(20), dimension(:), allocatable :: names
    !! Array of attribute names
    integer :: iline_
    !! Line number


    ! Initialise empty attributes array
    allocate(attributes(0))
    allocate(names(0))
    iline_ = 0

    ! Read lines until END or END ACTIVATION
    read_loop: do
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(err_msg,'("File encountered error (EoF?) before END ACTIVATION")')
          call stop_program(err_msg)
          return
       end if
       iline_ = iline_ + 1

       ! Skip empty or comment lines
       if(trim(adjustl(buffer)).eq."") cycle read_loop
       if(index(trim(adjustl(buffer)),"#") .eq. 1) cycle read_loop
       if(index(trim(adjustl(buffer)),"!") .eq. 1) cycle read_loop

       ! Check for end of activation block
       if(trim(adjustl(buffer)).eq."END" .or. &
            trim(adjustl(buffer)).eq."END ACTIVATION")then
          exit read_loop
       end if

       ! Look for NAME = VALUE pattern
       eq_pos = scan(buffer, "=")


       if(eq_pos .gt. 0)then
          ! Extract name (everything before =)
          attr_name = to_lower(adjustl(buffer(:eq_pos-1)))
          ! Extract value (everything after =)
          attr_value = adjustl(buffer(eq_pos+1:))


          if(index(trim(adjustl(buffer)),"ACTIVATION") .eq. 1 .and. iline_ .eq. 1)then
             attributes = [ onnx_attribute_type("name", "string", trim(attr_value)) ]
             exit read_loop
          end if

          ! Check if attribute already exists
          if(any(names .eq. attr_name))then
             write(err_msg,'("Duplicate activation attribute name: ''",A,"''")') &
                  trim(attr_name)
             call stop_program(trim(err_msg))
             return
          end if

          ! Grow attributes array
          attributes = [ &
               attributes, &
               onnx_attribute_type( &
                    trim(attr_name), &
                    "float", &
                    trim(attr_value) &
               ) &
          ]
          names = [ names, attr_name ]

       end if
    end do read_loop

    if(present(iline)) iline = iline + iline_

  end function read_activation_attributes
!###############################################################################


!###############################################################################
  function read_activation(unit, iline) result(activation)
    !! Read activation function from input file
    implicit none

    ! Arguments
    integer, intent(in) :: unit
    !! Unit number for input file
    integer, intent(inout), optional :: iline
    !! Line number

    class(base_actv_type), allocatable :: activation
    !! Activation function object

    ! Local variables
    type(onnx_attribute_type), allocatable, dimension(:) :: attributes
    !! Array of ONNX attributes
    integer :: i
    !! Loop variable
    character(20) :: actv_name
    !! Activation function name
    logical :: found
    !! Flag for finding activation creator
    integer :: creator_index
    !! Index of activation creator
    integer :: iline_ = 0
    !! Line number

    ! initialise list if needed
    if(.not.allocated(list_of_onnx_activation_creators)) &
         call allocate_list_of_onnx_activation_creators()

    ! Read activation attributes
    attributes = read_activation_attributes(unit, iline=iline_)
    if(present(iline)) iline = iline + iline_

    ! Extract activation name
    actv_name = ""
    do i=1, size(attributes,dim=1)
       if(trim(to_lower(attributes(i)%name)) .eq. "name")then
          actv_name = trim(to_lower(attributes(i)%val))
          exit
       end if
    end do
    if(actv_name .eq. "")then
       call stop_program( &
            "Activation name '"// actv_name //"' not specified in activation block" &
       )
       return
    end if
    do i = 1, size(list_of_onnx_activation_creators,dim=1)
       if(trim(to_lower(list_of_onnx_activation_creators(i)%name)) .eq. actv_name)then
          found = .true.
          creator_index = i
          exit
       end if
    end do
    if(.not.found)then
       call stop_program( &
            "Activation name '"// actv_name //"' not recognised" &
       )
       return
    end if
    allocate(activation, source = list_of_onnx_activation_creators(creator_index)% &
         create_ptr(attributes))

  end function read_activation
!###############################################################################

end module athena__activation
