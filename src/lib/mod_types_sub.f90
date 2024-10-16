submodule(custom_types) custom_types_submodule
  use constants, only: real32

contains

  pure module function add_array(a, b) result(output)
    implicit none
    class(array_type), intent(in) :: a, b 
    class(array_type), allocatable :: output

    output = a
    if(.not.allocated(a%val).or..not.allocated(b%val))then
       return
    elseif(a%size.ne.b%size) then
       return
    end if

    output%val = a%val + b%val

  end function add_array

  pure module function flatten_array(this) result(output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(this%size) :: output

    output = reshape(this%val, [this%size])
  end function flatten_array

  pure module subroutine get_array(this, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(..), allocatable, intent(out) :: output

    select rank(output)
    rank(1)
       output = this%flatten()
    rank(2)
       output = this%val
    rank(3)
       select type(this)
       type is(array3d_type)
          output = this%val_ptr
       end select
    rank(4)
       select type(this)
       type is(array4d_type)
          output = this%val_ptr
       end select
    rank(5)
       select type(this)
       type is(array5d_type)
          output = this%val_ptr
       end select
    rank default
       return
    end select
  end subroutine get_array

  pure module subroutine set_array(this, input)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank default
       return
    end select
  end subroutine set_array

  module function init_array1d(array_shape) result(output)
    implicit none
    integer, dimension(:), intent(in), optional :: array_shape
    type(array1d_type) :: output

    output%rank = 1
    allocate(output%shape(1))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array1d

  module subroutine allocate_array1d(this, array_shape, source)
    implicit none
    class(array1d_type), intent(inout), target :: this
    integer, dimension(:), intent(in), optional :: array_shape
    class(*), dimension(..), intent(in), optional :: source

    if(allocated(this%val).or.this%allocated) &
         stop "ERROR: Trying to allocate already allocated array values"
    this%rank = 1
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate( this%val( array_shape(1), 1 ) )
       this%val_ptr( 1:array_shape(1) ) => this%val
    end if
    if(present(source))then
      select rank(source)
      rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape)) &
                  stop 'ERROR: Source shape not provided'
             this%val(:,:) = source
          type is (array1d_type)
             if(present(array_shape))then
                 if(any(array_shape.ne.shape(source%val))) &
                    stop 'ERROR: Source shape does not match array shape'
             end if
             this = source
          class default
             stop 'ERROR: Incompatible source type for rank 0'
          end select
       rank(1)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                  stop 'ERROR: Source shape does not match array shape'
             else
                allocate( this%val( size(source, dim=1), 1 ) )
                this%val_ptr( 1:size(source, dim=1) ) => this%val
             end if
             this%val_ptr = source
          class default
             stop 'ERROR: Incompatible source type for rank 1'
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             stop 'ERROR: Incompatible source type for rank 2'
          end select
       rank(3)
       rank default
          stop 'ERROR: Unrecognised source type'
       end select
    end if
    if(.not.present(source).and.present(array_shape)) &
         stop 'ERROR: No shape or source provided'
    this%shape = shape(this%val_ptr)
    this%size = product(this%shape)

  end subroutine allocate_array1d

  pure module subroutine deallocate_array1d(this, keep_shape)
    implicit none
    class(array1d_type), intent(inout) :: this
    logical, intent(in), optional :: keep_shape

    logical :: keep_shape_

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array1d

  pure module subroutine assign_array1d(this, input)
    implicit none
    type(array1d_type), intent(in) :: input
    type(array1d_type), intent(out) :: this

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
  end subroutine assign_array1d





  module function init_array2d(array_shape) result(output)
    implicit none
    integer, dimension(:), intent(in), optional :: array_shape
    type(array2d_type) :: output

    output%rank = 1
    allocate(output%shape(1))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array2d

  module subroutine allocate_array2d(this, array_shape, source)
    implicit none
    class(array2d_type), intent(inout), target :: this
    integer, dimension(:), intent(in), optional :: array_shape
    class(*), dimension(..), intent(in), optional :: source

    if(allocated(this%val).or.this%allocated) &
         stop "ERROR: Trying to allocate already allocated array values"
    this%rank = 1
    this%allocated = .true.
    if(present(array_shape)) allocate(this%val(array_shape(1), array_shape(2)))
    if(present(source))then
      select rank(source)
      rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape)) &
                  stop 'ERROR: Source shape not provided'
             this%val(:,:) = source
          type is (array2d_type)
             if(present(array_shape))then
                 if(any(array_shape.ne.shape(source%val))) &
                    stop 'ERROR: Source shape does not match array shape'
             end if
             this = source
          class default
             stop 'ERROR: Incompatible source type for rank 0'
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                  stop 'ERROR: Source shape does not match array shape'
             end if
             this%val = source
          class default
             stop 'ERROR: Incompatible source type for rank 2'
          end select
       rank default
          stop 'ERROR: Unrecognised source type'
       end select
    end if
    if(.not.present(source).and.present(array_shape)) &
         stop 'ERROR: No shape or source provided'
    this%val_ptr(1:size(this%val, dim=1), 1:size(this%val, dim=2)) => this%val
    this%shape(1) = size(this%val, dim=1)
    this%size = product(this%shape)

  end subroutine allocate_array2d

  pure module subroutine deallocate_array2d(this, keep_shape)
    implicit none
    class(array2d_type), intent(inout) :: this
    logical, intent(in), optional :: keep_shape

    logical :: keep_shape_

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array2d

  pure module subroutine assign_array2d(this, input)
    implicit none
    type(array2d_type), intent(out) :: this
    type(array2d_type), intent(in) :: input

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
  end subroutine assign_array2d




  module function init_array3d(array_shape) result(output)
    implicit none
    integer, dimension(:), intent(in), optional :: array_shape
    type(array3d_type) :: output

    output%rank = 2
    allocate(output%shape(2))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array3d

  module subroutine allocate_array3d(this, array_shape, source)
    implicit none
    class(array3d_type), intent(inout), target :: this
    integer, dimension(:), intent(in), optional :: array_shape
    class(*), dimension(..), intent(in), optional :: source

    if(allocated(this%val).or.this%allocated) &
         stop "ERROR: Trying to allocate already allocated array values"
    this%rank = 2
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:2)),&
            array_shape(3) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3) &
       ) => this%val
    end if
    if(present(source))then
      select rank(source)
      rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape)) &
                  stop 'ERROR: Source shape not provided'
             this%val(:,:) = source
          type is (array3d_type)
             if(present(array_shape))then
                 if(any(array_shape.ne.shape(source%val))) &
                    stop 'ERROR: Source shape does not match array shape'
             end if
             this = source
          class default
             stop 'ERROR: Incompatible source type for rank 0'
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             stop 'ERROR: Incompatible source type for rank 2'
          end select
       rank(3)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                  stop 'ERROR: Source shape does not match array shape'
             else
                allocate(this%val(size(source(:,:,1)), size(source,3)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             stop 'ERROR: Incompatible source type for rank 3'
          end select
       rank default
          stop 'ERROR: Unrecognised source type'
       end select
    end if
    if(.not.present(source).and.present(array_shape)) &
         stop 'ERROR: No shape or source provided'
    this%shape(:) = shape(this%val_ptr(:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array3d

  pure module subroutine deallocate_array3d(this, keep_shape)
    implicit none
    class(array3d_type), intent(inout) :: this
    logical, intent(in), optional :: keep_shape

    logical :: keep_shape_

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array3d

  pure module subroutine set_array3d(this, input)
    implicit none
    class(array3d_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(3)
       this%val_ptr(:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array3d

  pure module subroutine assign_array3d(this, input)
    implicit none
    type(array3d_type), intent(out) :: this
    type(array3d_type), intent(in) :: input

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
  end subroutine assign_array3d




  module function init_array4d(array_shape) result(output)
    implicit none
    integer, dimension(:), intent(in), optional :: array_shape
    type(array4d_type) :: output

    output%rank = 3
    allocate(output%shape(3))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array4d

  module subroutine allocate_array4d(this, array_shape, source)
    implicit none
    class(array4d_type), intent(inout), target :: this
    integer, dimension(:), intent(in), optional :: array_shape
    class(*), dimension(..), intent(in), optional :: source

    if(allocated(this%val).or.this%allocated) &
         stop "ERROR: Trying to allocate already allocated array values"
    this%rank = 3
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:3)),&
            array_shape(4) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3), &
            1:array_shape(4) &
       ) => this%val
    end if
    if(present(source))then
      select rank(source)
      rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape)) &
                  stop 'ERROR: Source shape not provided'
             this%val(:,:) = source
          type is (array4d_type)
             if(present(array_shape))then
                 if(any(array_shape.ne.shape(source%val))) &
                    stop 'ERROR: Source shape does not match array shape'
             end if
             this = source
          class default
             stop 'ERROR: Incompatible source type for rank 0'
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             stop 'ERROR: Incompatible source type for rank 2'
          end select
       rank(4)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                  stop 'ERROR: Source shape does not match array shape'
             else
                allocate(this%val(size(source(:,:,:,1)), size(source,4)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3), &
                     1:size(source, dim=4) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             stop 'ERROR: Incompatible source type for rank 4'
          end select
       rank default
          stop 'ERROR: Unrecognised source type'
       end select
    end if
    if(.not.present(source).and.present(array_shape)) &
         stop 'ERROR: No shape or source provided'
    this%shape(:) = shape(this%val_ptr(:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array4d

  pure module subroutine deallocate_array4d(this, keep_shape)
    implicit none
    class(array4d_type), intent(inout) :: this
    logical, intent(in), optional :: keep_shape

    logical :: keep_shape_

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array4d

  pure module subroutine set_array4d(this, input)
    implicit none
    class(array4d_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(4)
       this%val_ptr(:,:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array4d

  pure module subroutine assign_array4d(this, input)
    implicit none
    type(array4d_type), intent(out) :: this
    type(array4d_type), intent(in) :: input

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
  end subroutine assign_array4d




  module function init_array5d(array_shape) result(output)
    implicit none
    integer, dimension(:), intent(in), optional :: array_shape
    type(array5d_type) :: output

    output%rank = 4
    allocate(output%shape(4))
    if(present(array_shape)) call output%allocate(array_shape)

  end function init_array5d

  module subroutine allocate_array5d(this, array_shape, source)
    implicit none
    class(array5d_type), intent(inout), target :: this
    integer, dimension(:), intent(in), optional :: array_shape
    class(*), dimension(..), intent(in), optional :: source

    if(allocated(this%val).or.this%allocated) &
         stop "ERROR: Trying to allocate already allocated array values"
    this%rank = 4
    this%allocated = .true.
    if(present(array_shape))then
       this%shape = array_shape
       allocate(this%val(&
            product(array_shape(1:4)),&
            array_shape(5) &
       ) )
       this%val_ptr( &
            1:array_shape(1), &
            1:array_shape(2), &
            1:array_shape(3), &
            1:array_shape(4), &
            1:array_shape(5) &
       ) => this%val
    end if
    if(present(source))then
      select rank(source)
      rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape)) &
                  stop 'ERROR: Source shape not provided'
             this%val(:,:) = source
          type is (array5d_type)
             if(present(array_shape))then
                 if(any(array_shape.ne.shape(source%val))) &
                    stop 'ERROR: Source shape does not match array shape'
             end if
             this = source
          class default
             stop 'ERROR: Incompatible source type for rank 0'
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             this%val(:,:) = source
          class default
             stop 'ERROR: Incompatible source type for rank 2'
          end select
       rank(5)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source))) &
                  stop 'ERROR: Source shape does not match array shape'
             else
                allocate(this%val(size(source(:,:,:,:,1)), size(source,5)))
                this%val_ptr( &
                     1:size(source, dim=1), &
                     1:size(source, dim=2), &
                     1:size(source, dim=3), &
                     1:size(source, dim=4), &
                     1:size(source, dim=5) &
                ) => this%val
             end if
             this%val_ptr = source
          class default
             stop 'ERROR: Incompatible source type for rank 5'
          end select
       rank default
          stop 'ERROR: Unrecognised source type'
       end select
    end if
    if(.not.present(source).and.present(array_shape)) &
         stop 'ERROR: No shape or source provided'
    this%shape(:) = shape(this%val_ptr(:,:,:,:,1))
    this%size = product(this%shape)

  end subroutine allocate_array5d

  pure module subroutine deallocate_array5d(this, keep_shape)
    implicit none
    class(array5d_type), intent(inout) :: this
    logical, intent(in), optional :: keep_shape

    logical :: keep_shape_

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape
    if(.not.keep_shape_) this%shape = 0
    deallocate(this%val)
    this%allocated = .false.
    this%size = 0

  end subroutine deallocate_array5d

  pure module subroutine set_array5d(this, input)
    implicit none
    class(array5d_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(5)
       this%val_ptr(:,:,:,:,:) = input
    rank default
       return
    end select
  end subroutine set_array5d

  pure module subroutine assign_array5d(this, input)
    implicit none
    type(array5d_type), intent(out) :: this
    type(array5d_type), intent(in) :: input

    this%rank = input%rank
    this%shape = input%shape
    this%size = input%size
    this%allocated = input%allocated
    this%val = input%val
  end subroutine assign_array5d

end submodule custom_types_submodule