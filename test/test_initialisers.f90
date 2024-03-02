program test_initialisers
   use athena, only: &
        full_layer_type, &
        conv2d_layer_type, &
        conv3d_layer_type, &
        base_layer_type
   implicit none
 
   class(base_layer_type), allocatable :: full_layer, conv2d_layer, conv3d_layer
   logical :: success = .true.
 
   integer :: i
   integer :: width = 5, num_channels = 3, batch_size = 2
   character(len=20) :: initialiser_names(11)
   
   initialiser_names(1)  = 'zeros'
   initialiser_names(2)  = 'ones'
   initialiser_names(3)  = 'ident'
   initialiser_names(4)  = 'gaussian'
   initialiser_names(5)  = 'normal'
   initialiser_names(6)  = 'glorot_normal'
   initialiser_names(7)  = 'glorot_uniform'
   initialiser_names(8)  = 'he_normal'
   initialiser_names(9)  = 'he_uniform'
   initialiser_names(10)  = 'lecun_normal'
   initialiser_names(11) = 'lecun_uniform'


   do i = 1, size(initialiser_names)
      !! check for rank 2 data
      !!------------------------------------------------------------------------
      !! set up full layer
      full_layer = full_layer_type( &
           num_inputs=1, &
           num_outputs=10, &
           batch_size = batch_size, &
           kernel_initialiser = initialiser_names(i))

      !! check layer name
      select type(full_layer)
      type is(full_layer_type)
         if(.not. full_layer%kernel_initialiser .eq. &
              trim(initialiser_names(i)))then
            success = .false.
            write(0,*) 'kernel initialiser has wrong name for ', &
                 trim(initialiser_names(i))
            write(*,*) 
         end if
      class default
         success = .false.
         write(0,*) 'full layer is not of type full_layer_type'
      end select

      !! check for rank 3 data
      !!------------------------------------------------------------------------
      !! set up full layer
      conv2d_layer = conv2d_layer_type( &
           input_shape = [width, width, num_channels], &
           batch_size = batch_size, &
           kernel_initialiser = initialiser_names(i))

      !! check layer name
      select type(conv2d_layer)
      type is(conv2d_layer_type)
         if(.not. conv2d_layer%kernel_initialiser .eq. &
              trim(initialiser_names(i)))then
            success = .false.
            write(0,*) 'kernel initialiser has wrong name for ', &
                 trim(initialiser_names(i))
            write(*,*) 
         end if
      class default
         success = .false.
         write(0,*) 'conv layer is not of type conv2d_layer_type'
      end select

      !! check for rank 4 data
      !!------------------------------------------------------------------------
      conv3d_layer = conv3d_layer_type( &
           input_shape = [width, width, width, num_channels], &
           batch_size = batch_size, &
           kernel_initialiser = initialiser_names(i))

      !! check layer name
      select type(conv3d_layer)
      type is(conv3d_layer_type)
         if(.not. conv3d_layer%kernel_initialiser .eq. &
              trim(initialiser_names(i)))then
            success = .false.
            write(0,*) 'kernel initialiser has wrong name for ', &
                 trim(initialiser_names(i))
            write(*,*) 
         end if
      class default
         success = .false.
         write(0,*) 'conv layer is not of type conv2d_layer_type'
      end select
   end do


   !! check for any fails
   write(*,*) "----------------------------------------"
   if(success)then
      write(*,*) 'test_initialisers passed all tests'
   else
      write(0,*) 'test_initialisers failed one or more tests'
      stop 1
   end if
 
 end program test_initialisers