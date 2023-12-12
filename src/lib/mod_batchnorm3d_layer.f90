!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ARTEMIS group (Hepplestone research group)
!!! Think Hepplestone, think HRG
!!!#############################################################################
module batchnorm3d_layer
  use constants, only: real12
  use base_layer, only: batch_layer_type, learnable_layer_type
  use custom_types, only: initialiser_type
  implicit none
  
  
  type, extends(batch_layer_type) :: batchnorm3d_layer_type
     real(real12), allocatable, dimension(:,:,:,:,:) :: output
     real(real12), allocatable, dimension(:,:,:,:,:) :: di ! gradient of input (i.e. delta)
   contains
     procedure, pass(this) :: init => init_batchnorm3d
     procedure, pass(this) :: set_batch_size => set_batch_size_batchnorm3d
     procedure, pass(this) :: print => print_batchnorm3d
   
     procedure, pass(this) :: forward  => forward_rank
     procedure, pass(this) :: backward => backward_rank
     procedure, private, pass(this) :: forward_5d
     procedure, private, pass(this) :: backward_5d

     procedure, pass(this) :: reduce => layer_reduction
     procedure, pass(this) :: merge => layer_merge
     procedure :: add_t_t => layer_add  !t = type, r = real, i = int
     generic :: operator(+) => add_t_t !, public
  end type batchnorm3d_layer_type

  
  interface batchnorm3d_layer_type
     module function layer_setup( &
          input_shape, batch_size, &
          momentum, epsilon, &
          gamma_init_mean, gamma_init_std, &
          beta_init_mean, beta_init_std, &
          kernel_initialiser, bias_initialiser, &
          moving_mean_initialiser, moving_variance_initialiser &
          ) result(layer)
       integer, dimension(:), optional, intent(in) :: input_shape
       integer, optional, intent(in) :: batch_size
       real(real12), optional, intent(in) :: momentum, epsilon
       real(real12), optional, intent(in) :: gamma_init_mean, gamma_init_std
       real(real12), optional, intent(in) :: beta_init_mean, beta_init_std
       character(*), optional, intent(in) :: &
            kernel_initialiser, bias_initialiser, &
            moving_mean_initialiser, moving_variance_initialiser
       type(batchnorm3d_layer_type) :: layer
     end function layer_setup
  end interface batchnorm3d_layer_type


  private
  public :: batchnorm3d_layer_type
  public :: read_batchnorm3d_layer


contains

!!!#############################################################################
!!! layer reduction
!!!#############################################################################
  subroutine layer_reduction(this, rhs)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: rhs

    select type(rhs)
    class is(batchnorm3d_layer_type)
       this%dg = this%dg + rhs%dg
       this%db = this%db + rhs%db
    end select

  end subroutine  layer_reduction
!!!#############################################################################


!!!#############################################################################
!!! layer addition
!!!#############################################################################
  function layer_add(a, b) result(output)
    implicit none
    class(batchnorm3d_layer_type), intent(in) :: a, b
    type(batchnorm3d_layer_type) :: output

    output = a
    output%dg = output%dg + b%dg
    output%db = output%db + b%db

  end function layer_add
!!!#############################################################################


!!!#############################################################################
!!! layer merge
!!!#############################################################################
  subroutine layer_merge(this, input)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    class(learnable_layer_type), intent(in) :: input

    select type(input)
    class is(batchnorm3d_layer_type)
       this%dg = this%dg + input%dg
       this%db = this%db + input%db
    end select

  end subroutine layer_merge
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!



!!!#############################################################################
!!! forward propagation assumed rank handler
!!!#############################################################################
  pure subroutine forward_rank(this, input)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input

    select rank(input); rank(5)
       call forward_5d(this, input)
    end select
  end subroutine forward_rank
!!!#############################################################################


!!!#############################################################################
!!! backward propagation assumed rank handler
!!!#############################################################################
  pure subroutine backward_rank(this, input, gradient)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    real(real12), dimension(..), intent(in) :: input
    real(real12), dimension(..), intent(in) :: gradient

    select rank(input); rank(5)
    select rank(gradient); rank(5)
      call backward_5d(this, input, gradient)
    end select
    end select
  end subroutine backward_rank
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! set up layer
!!!#############################################################################
  module function layer_setup( &
       input_shape, batch_size, &
       momentum, epsilon, &
       gamma_init_mean, gamma_init_std, &
       beta_init_mean, beta_init_std, &
       kernel_initialiser, bias_initialiser, &
       moving_mean_initialiser, moving_variance_initialiser &
       ) result(layer)
    use initialiser, only: get_default_initialiser
    implicit none
    integer, dimension(:), optional, intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    real(real12), optional, intent(in) :: momentum, epsilon
    real(real12), optional, intent(in) :: gamma_init_mean, gamma_init_std
    real(real12), optional, intent(in) :: beta_init_mean, beta_init_std
    character(*), optional, intent(in) :: &
         kernel_initialiser, bias_initialiser, &
         moving_mean_initialiser, moving_variance_initialiser
    
    type(batchnorm3d_layer_type) :: layer
    

    layer%name = "batchnorm3d"
    layer%input_rank = 4
    !!--------------------------------------------------------------------------
    !! initialise batch size
    !!--------------------------------------------------------------------------
    if(present(batch_size)) layer%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! set up momentum and epsilon
    !!--------------------------------------------------------------------------
    if(present(momentum))then
       layer%momentum = momentum
    else
       layer%momentum = 0._real12
    end if
    if(present(epsilon))then
       layer%epsilon = epsilon
    else
       layer%epsilon = 1.E-5_real12
    end if


    !!--------------------------------------------------------------------------
    !! set up initialiser mean and standard deviations
    !!--------------------------------------------------------------------------
    if(present(gamma_init_mean)) layer%gamma_init_mean = gamma_init_mean
    if(present(gamma_init_std))  layer%gamma_init_std = gamma_init_std
    if(present(beta_init_mean))  layer%beta_init_mean = beta_init_mean
    if(present(beta_init_std))   layer%beta_init_std = beta_init_std


    !!--------------------------------------------------------------------------
    !! define gamma and beta initialisers
    !!--------------------------------------------------------------------------
    if(present(kernel_initialiser)) layer%kernel_initialiser = kernel_initialiser
    if(trim(layer%kernel_initialiser).eq.'') &
         layer%kernel_initialiser = 'ones'
         !get_default_initialiser("batch")
    write(*,'("BATCHNORM3D kernel (gamma) initialiser: ",A)') &
         trim(layer%kernel_initialiser)
    if(present(bias_initialiser)) layer%bias_initialiser = bias_initialiser
    if(trim(layer%bias_initialiser).eq.'') &
         layer%bias_initialiser = 'zeros'
         !get_default_initialiser("batch")
    write(*,'("BATCHNORM3D bias (beta) initialiser: ",A)') &
         trim(layer%bias_initialiser)

    if(present(moving_mean_initialiser)) &
         layer%moving_mean_initialiser = moving_mean_initialiser
    if(trim(layer%moving_mean_initialiser).eq.'') &
         layer%moving_mean_initialiser = 'zeros'
         !get_default_initialiser("batch")
    write(*,'("BATCHNORM3D moving mean initialiser: ",A)') &
         trim(layer%moving_mean_initialiser)
    if(present(moving_variance_initialiser)) &
         layer%moving_variance_initialiser = moving_variance_initialiser
    if(trim(layer%moving_variance_initialiser).eq.'') &
         layer%moving_variance_initialiser = 'ones'
         !get_default_initialiser("batch")
    write(*,'("BATCHNORM3D moving variance initialiser: ",A)') &
         trim(layer%moving_variance_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise layer shape
    !!--------------------------------------------------------------------------
    if(present(input_shape)) call layer%init(input_shape=input_shape)

  end function layer_setup
!!!#############################################################################


!!!#############################################################################
!!! initialise layer
!!!#############################################################################
  subroutine init_batchnorm3d(this, input_shape, batch_size, verbose)
    use initialiser, only: initialiser_setup
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    integer, dimension(:), intent(in) :: input_shape
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose

    integer :: t_verb
    class(initialiser_type), allocatable :: t_initialiser


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if
    if(present(batch_size)) this%batch_size = batch_size


    !!--------------------------------------------------------------------------
    !! initialise input shape
    !!--------------------------------------------------------------------------
    if(.not.allocated(this%input_shape)) call this%set_shape(input_shape)


    !!-----------------------------------------------------------------------
    !! set up number of channels, width, height
    !!-----------------------------------------------------------------------
    this%output_shape = this%input_shape
    this%num_channels = this%input_shape(this%input_rank)


    !!-----------------------------------------------------------------------
    !! allocate mean, variance, gamma, beta, dg, db
    !!-----------------------------------------------------------------------
    allocate(this%mean(this%num_channels), source=0._real12)
    allocate(this%variance, source=this%mean)
    allocate(this%gamma, source=this%mean)
    allocate(this%beta, source=this%mean)
    allocate(this%dg, source=this%mean)
    allocate(this%db, source=this%mean)


    !!-----------------------------------------------------------------------
    !! initialise gamma
    !!-----------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%kernel_initialiser))
    t_initialiser%mean = this%gamma_init_mean
    t_initialiser%std  = this%gamma_init_std
    call t_initialiser%initialise(this%gamma, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    !! initialise beta
    !!-----------------------------------------------------------------------
    allocate(t_initialiser, source=initialiser_setup(this%bias_initialiser))
    t_initialiser%mean = this%beta_init_mean
    t_initialiser%std  = this%beta_init_std
    call t_initialiser%initialise(this%beta, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !!-----------------------------------------------------------------------
    !! initialise moving mean
    !!-----------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_mean_initialiser))
    call t_initialiser%initialise(this%mean, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)

    !! initialise moving variance
    !!-----------------------------------------------------------------------
    allocate(t_initialiser, &
         source=initialiser_setup(this%moving_variance_initialiser))
    call t_initialiser%initialise(this%variance, &
         fan_in =this%num_channels, &
         fan_out=this%num_channels)
    deallocate(t_initialiser)


    !!--------------------------------------------------------------------------
    !! initialise batch size-dependent arrays
    !!--------------------------------------------------------------------------
    if(this%batch_size.gt.0) call this%set_batch_size(this%batch_size)

  end subroutine init_batchnorm3d
!!!#############################################################################

  
!!!#############################################################################
!!! set batch size
!!!#############################################################################
  subroutine set_batch_size_batchnorm3d(this, batch_size, verbose)
   implicit none
   class(batchnorm3d_layer_type), intent(inout) :: this
   integer, intent(in) :: batch_size
   integer, optional, intent(in) :: verbose

   integer :: t_verb


   !!--------------------------------------------------------------------------
   !! initialise optional arguments
   !!--------------------------------------------------------------------------
   if(present(verbose))then
      t_verb = verbose
   else
      t_verb = 0
   end if
   this%batch_size = batch_size


   !!--------------------------------------------------------------------------
   !! set norm
   !!--------------------------------------------------------------------------
   this%norm = real( &
        this%batch_size * &
        product(this%input_shape(1:this%input_rank-1) ),real12)


   !!--------------------------------------------------------------------------
   !! allocate arrays
   !!--------------------------------------------------------------------------
   if(allocated(this%input_shape))then
      if(allocated(this%output)) deallocate(this%output)
      allocate(this%output( &
           this%output_shape(1), &
           this%output_shape(2), &
           this%output_shape(3), this%num_channels, &
           this%batch_size), &
           source=0._real12)
      if(allocated(this%di)) deallocate(this%di)
      allocate(this%di, source=this%output)
   end if

 end subroutine set_batch_size_batchnorm3d
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print layer to file
!!!#############################################################################
  subroutine print_batchnorm3d(this, file)
    implicit none
    class(batchnorm3d_layer_type), intent(in) :: this
    character(*), intent(in) :: file

    integer :: unit
    integer :: m

    !! open file with new unit
    !!--------------------------------------------------------------------------
    open(newunit=unit, file=trim(file), access='append')

    !! write convolution initial parameters
    !!--------------------------------------------------------------------------
    write(unit,'("BATCHNORM3D")')
    write(unit,'(3X,"INPUT_SHAPE = ",3(1X,I0))') this%input_shape
    write(unit,'(3X,"MOMENTUM = ",F0.9)') this%momentum
    write(unit,'(3X,"EPSILON = ",F0.9)') this%epsilon
    write(unit,'(3X,"NUM_CHANNELS = ",I0)') this%num_channels
    write(unit,'("GAMMA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%gamma(m)
    end do
    write(unit,'("END GAMMA")')
    write(unit,'("BETA")')
    do m=1,this%num_channels
       write(unit,'(5(E16.8E2))') this%beta(m)
    end do
    write(unit,'("END BETA")')
    write(unit,'("END BATCHNORM3D")')

    !! close unit
    !!--------------------------------------------------------------------------
    close(unit)

  end subroutine print_batchnorm3d
!!!#############################################################################


!!!#############################################################################
!!! read layer from file
!!!#############################################################################
  function read_batchnorm3d_layer(unit, verbose) result(layer)
    use infile_tools, only: assign_val, assign_vec
    use misc, only: to_lower, icount
    implicit none
    integer, intent(in) :: unit
    integer, optional, intent(in) :: verbose
 
    class(batchnorm3d_layer_type), allocatable :: layer
 
    integer :: stat, t_verb
    integer :: itmp1, c, i, j, k
    integer :: num_channels
    real(real12) :: momentum = 0._real12, epsilon = 1.E-5_real12
    logical :: found_gamma=.false., found_beta=.false.
    character(14) :: kernel_initialiser='', bias_initialiser=''
    character(256) :: buffer, tag

    integer, dimension(3) :: input_shape
    real(real12), allocatable, dimension(:) :: data_list


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(verbose))then
       t_verb = verbose
    else
       t_verb = 0
    end if


    !!--------------------------------------------------------------------------
    !! loop over tags in layer card
    !!--------------------------------------------------------------------------
    tag_loop: do

       !! check for end of file
       read(unit,'(A)',iostat=stat) buffer
       if(stat.ne.0)then
          write(0,*) "ERROR: file encountered error (EoF?) before END BATCHNORM3D"
          stop "Exiting..."
       end if
       if(trim(adjustl(buffer)).eq."") cycle tag_loop
 
       !! check for end of convolution card
       if(trim(adjustl(buffer)).eq."END BATCHNORM3D")then
          backspace(unit)
          exit tag_loop
       end if

       tag=trim(adjustl(buffer))
       if(scan(buffer,"=").ne.0) tag=trim(tag(:scan(tag,"=")-1))

       !! read parameters from save file
       select case(trim(tag))
       case("INPUT_SHAPE")
          call assign_vec(buffer, input_shape, itmp1)
       case("MOMENTUM")
          call assign_val(buffer, momentum, itmp1)
       case("EPSILON")
          call assign_val(buffer, epsilon, itmp1)
       case("KERNEL_INITIALISER")
          call assign_val(buffer, kernel_initialiser, itmp1)
       case("BIAS_INITIALISER")
          call assign_val(buffer, bias_initialiser, itmp1)
       case("GAMMA")
          found_gamma = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
       case("beta")
          found_beta = .true.
          kernel_initialiser = 'zeros'
          bias_initialiser   = 'zeros'
          exit tag_loop
       case default
          !! don't look for "e" due to scientific notation of numbers
          !! ... i.e. exponent (E+00)
          if(scan(to_lower(trim(adjustl(buffer))),&
               'abcdfghijklmnopqrstuvwxyz').eq.0)then
             cycle tag_loop
          elseif(tag(:3).eq.'END')then
             cycle tag_loop
          end if
          stop "Unrecognised line in input file: "//trim(adjustl(buffer))
       end select
    end do tag_loop


    !!--------------------------------------------------------------------------
    !! set transfer activation function
    !!--------------------------------------------------------------------------
    num_channels = input_shape(size(input_shape,1))
    layer = batchnorm3d_layer_type( input_shape=input_shape, &
         momentum = momentum, epsilon = epsilon &
         )


    !!--------------------------------------------------------------------------
    !! check if WEIGHTS card was found
    !!--------------------------------------------------------------------------
    allocate(data_list(num_channels), source=0._real12)
    do i=1,2
      if(found_gamma.or.found_beta)then
         c = 1
         k = 1
         data_list = 0._real12
         data_concat_loop: do while(c.le.num_channels)
            read(unit,'(A)',iostat=stat) buffer
            if(stat.ne.0) exit data_concat_loop
            k = icount(buffer)
            read(buffer,*,iostat=stat) (data_list(j),j=c,c+k-1)
            c = c + k
         end do data_concat_loop
         if(found_gamma)then
            layer%gamma = data_list
            found_gamma = .false.
         elseif(found_beta)then
            layer%beta = data_list
            found_beta = .false.
         end if
         read(unit,'(A)',iostat=stat) buffer
         if(index(trim(adjustl(buffer)),"GAMMA").eq.1) found_gamma = .true.
         if(index(trim(adjustl(buffer)),"BETA").eq.1) found_beta = .true.
      end if
    end do
    deallocate(data_list)


    !! check for end of layer card
    !!-----------------------------------------------------------------------
    read(unit,'(A)') buffer
    if(trim(adjustl(buffer)).ne."END BATCHNORM3D")then
       write(*,*) trim(adjustl(buffer))
       stop "ERROR: END BATCHNORM3D not where expected"
    end if

  end function read_batchnorm3d_layer
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward propagation
!!!#############################################################################
  pure subroutine forward_5d(this, input)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    real(real12), dimension(:,:,:,:,:), intent(in) :: input

    integer :: m
    real(real12), dimension(this%num_channels) :: t_mean, t_variance

    
    select case(this%inference)
    case(.true.)
       do concurrent(m=1:this%num_channels)
          !! normalize each feature
          this%output(:,:,:,m,:) = this%gamma(m) * (input(:,:,:,m,:) - &
                this%mean(m)) / &
                sqrt( &
                this%batch_size / (this%batch_size - 1) * this%variance(m) + &
                this%epsilon) + &
                this%beta(m)
       end do
    case default
       do concurrent(m=1:this%num_channels)
             !! calculate current mean and variance
             t_mean(m) = sum(input(:,:,:,m,:)) / this%norm
             t_variance(m) = sum((input(:,:,:,m,:) - t_mean(m))**2._real12) / this%norm
      
             !! CONVERT TO USING inverse square root of variance (i.e. inverse std)
             !! would also need to include epsilon in the sqrt denominator

             !! update running averages
             if(this%momentum.ne.0._real12)then
                this%mean(m) = this%momentum * this%mean(m) + &
                      (1._real12 - this%momentum) * t_mean(m)
                this%variance(m) = this%momentum * this%variance(m) + &
                      (1._real12 - this%momentum) * t_variance(m)
             else
                this%mean(m) = t_mean(m)
                this%variance(m) = t_variance(m)
             end if

          !! normalize each feature
          this%output(:,:,:,m,:) = this%gamma(m) * (input(:,:,:,m,:) - &
                this%mean(m)) / &
                sqrt(this%variance(m) + this%epsilon) + this%beta(m)
       end do
    end select

  end subroutine forward_5d
!!!#############################################################################


!!!#############################################################################
!!! backward propagation
!!!#############################################################################
!!! https://kevinzakka.github.io/2016/09/14/batch_normalization/
  pure subroutine backward_5d(this, input, gradient)
    implicit none
    class(batchnorm3d_layer_type), intent(inout) :: this
    real(real12), dimension( &
         this%input_shape(1), &
         this%input_shape(2), &
         this%input_shape(3), &
         this%num_channels,this%batch_size), &
         intent(in) :: input
    real(real12), dimension( &
         this%output_shape(1), &
         this%output_shape(2), &
         this%output_shape(3), &
         this%num_channels,this%batch_size), &
         intent(in) :: gradient

    integer :: m
    real(real12), dimension( &
          this%input_shape(1), &
          this%input_shape(2), &
          this%input_shape(3), &
          this%num_channels,this%batch_size) :: x_hat, dx_hat


    do concurrent(m=1:this%num_channels)
       !! recalculate x_hat (i.e. normalised input)
       x_hat(:,:,:,m,:) = (input(:,:,:,m,:) - this%mean(m)) / &
            sqrt(this%variance(m) + this%epsilon)
 
       !! calculate gradient of normalised input
       dx_hat(:,:,:,m,:) = gradient(:,:,:,m,:) * this%gamma(m)

       !! calculate gradient of inputs
       this%di(:,:,:,m,:) = &
            1._real12 / (this%norm * sqrt(this%variance(m) + this%epsilon)) * &
            ( this%norm * dx_hat(:,:,:,m,:) - &
            sum(dx_hat(:,:,:,m,:)) - x_hat(:,:,:,m,:) * &
            sum(dx_hat(:,:,:,m,:) * x_hat(:,:,:,m,:)))

       !! calculate gradient of gamma and beta
       this%dg(m) = sum(gradient(:,:,:,m,:) * x_hat(:,:,:,m,:))
       this%db(m) = sum(gradient(:,:,:,m,:))
    end do

  end subroutine backward_5d
!!!#############################################################################

end module batchnorm3d_layer
!!!#############################################################################
