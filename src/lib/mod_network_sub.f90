!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor
!!! Code part of the ATHENA library - a feedforward neural network library
!!!#############################################################################
!!! submodule of the network module
!!! submodule contains the associated methods from the network module
!!!#############################################################################
submodule(network) network_submodule
#ifdef _OPENMP
  use omp_lib
#endif
  use misc_ml, only: shuffle

  use accuracy, only: categorical_score, mae_score, mse_score, r2_score
  use base_layer, only: &
       drop_layer_type, &
       learnable_layer_type, &
       batch_layer_type, &
       conv_layer_type, &
       pool_layer_type
#if defined(GFORTRAN)
  use container_layer, only: container_reduction
#endif

  use custom_types, only: &
       array1d_type, &
       array2d_type, &
       array3d_type, &
       array4d_type, &
       array5d_type
  use container_layer, only: list_of_layer_types

  !! layer types
  use input_layer,   only: input_layer_type
  use flatten_layer, only: flatten_layer_type

  !! fully connected (dense) layer types
  use full_layer,      only: full_layer_type

  implicit none

! #ifdef _OPENMP
!   !$omp declare reduction(network_reduction:network_type:omp_out%network_reduction(omp_in)) &
!   !$omp& initializer(omp_priv = omp_orig)
! #endif

contains

!!!#############################################################################
!!! network addition
!!!#############################################################################
  module subroutine network_reduction(this, source)
    implicit none
    class(network_type), intent(inout) :: this
    type(network_type), intent(in) :: source

    integer :: i
    
    this%metrics(1)%val = this%metrics(1)%val + source%metrics(1)%val
    this%metrics(2)%val = this%metrics(2)%val + source%metrics(2)%val
    do i=1,size(this%model)
       select type(layer_this => this%model(i)%layer)
       class is(learnable_layer_type)
          select type(layer_source => source%model(i)%layer)
          class is(learnable_layer_type)
             call layer_this%merge(layer_source)
          end select
       end select
    end do

  end subroutine network_reduction
!!!#############################################################################


!!!#############################################################################
!!! network addition
!!!#############################################################################
  module subroutine network_copy(this, source)
    implicit none
    class(network_type), intent(inout) :: this
    type(network_type), intent(in) :: source

    this%metrics = source%metrics
    this%model   = source%model
  end subroutine network_copy
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! generate graph order (i.e. sequential order of layers)
!!!#############################################################################
  module subroutine generate_vertex_order(this)
    implicit none
    class(network_type), intent(inout) :: this

    integer :: i, order_index
    logical, dimension(this%auto_graph%num_vertices) :: visited

    visited = .false.
    if(allocated(this%vertex_order)) deallocate(this%vertex_order)
    allocate(this%vertex_order(this%auto_graph%num_vertices), source=0)
    
    order_index = 0
    do i = this%auto_graph%num_vertices, 1
       if(.not.visited(i)) call this%dfs( &
            i, visited, this%vertex_order, order_index &
       )
    end do

  end subroutine generate_vertex_order
!!!#############################################################################


!!!#############################################################################
!!! depth first search
!!!#############################################################################
  module recursive subroutine dfs( &
       this, vertex_index, visited, order, order_index &
  )
    implicit none
    class(network_type), intent(inout) :: this
    integer, intent(in) :: vertex_index
    logical, dimension(this%auto_graph%num_vertices), intent(inout) :: visited
    integer, dimension(this%auto_graph%num_vertices), intent(inout) :: order
    integer, intent(inout) :: order_index

    integer :: i

    visited(vertex_index) = .true.
    do i = 1, this%auto_graph%num_vertices
       if(this%auto_graph%adjacency(i,vertex_index).ne.0)then
          if(.not.visited(i)) call this%dfs(i, visited, order, order_index)
       end if
    end do
    order_index = order_index + 1
    order(order_index) = vertex_index

  end subroutine dfs
!!!#############################################################################


!!!#############################################################################
!!! get root vertices
!!!#############################################################################
  module subroutine calculate_root_vertices(this)
    implicit none
    class(network_type), intent(inout) :: this

    integer :: i

   if(allocated(this%root_vertices)) deallocate(this%root_vertices)
   allocate(this%root_vertices(0))
   do i = 1, this%auto_graph%num_vertices
       if(all(this%auto_graph%adjacency(:,i).eq.0))then
          this%root_vertices = [this%root_vertices, i]
       end if
    end do
  end subroutine calculate_root_vertices
!!!#############################################################################


!!!#############################################################################
!!! get root vertices
!!!#############################################################################
  module subroutine calculate_output_vertices(this)
   implicit none
   class(network_type), intent(inout) :: this

   integer :: i

   if(allocated(this%output_vertices)) deallocate(this%output_vertices)
   allocate(this%output_vertices(0))
   do i = 1, this%auto_graph%num_vertices
      if(all(this%auto_graph%adjacency(i,:).eq.0))then
         this%output_vertices = [this%output_vertices, i]
      end if
   end do
 end subroutine calculate_output_vertices
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! print network to file
!!!#############################################################################
  module subroutine print(this, file)
    implicit none
    class(network_type), intent(in) :: this
    character(*), intent(in) :: file
    
    integer :: l, unit

    open(newunit=unit,file=file,status='replace')
    close(unit)
    
    do l=1,this%num_layers
       call this%model(l)%layer%print(file)
    end do

  end subroutine print
!!!#############################################################################


!!!#############################################################################
!!! read network from file
!!!#############################################################################
  module subroutine read(this, file)
   implicit none
   class(network_type), intent(inout) :: this
   character(*), intent(in) :: file
   
   integer :: i, unit, stat
   character(256) :: buffer
   integer :: layer_index
   open(newunit=unit,file=file,action='read')
   i = 0
   card_loop: do
      i = i + 1
      read(unit,'(A)',iostat=stat) buffer
      if(stat.lt.0)then
         exit card_loop
      elseif(stat.gt.0)then
         write(0,*) "ERROR: error encountered in network read"
         stop "Exiting..."
      end if
      if(trim(adjustl(buffer)).eq."") cycle card_loop

      !! check if a tag line
      if(scan(buffer,'=').ne.0)then
         write(0,*) "WARNING: unexpected line in read file"
         write(0,*) trim(buffer)
         write(0,*) " skipping..."
         cycle card_loop
      end if

      !! check for card
      layer_index = findloc(list_of_layer_types%name, trim(adjustl(buffer)), dim=1)
      if(layer_index.eq.0)then
         write(0,*) "ERROR: unrecognised card '"//&
              &trim(adjustl(buffer))//"'"
         stop "Exiting..."
      end if
      call this%add(list_of_layer_types(layer_index)%read_ptr(unit))
   end do card_loop
   close(unit)

 end subroutine read
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! append layer to network
!!!#############################################################################
  module subroutine add(this, layer, input_list, output_list)
    implicit none
    class(network_type), intent(inout) :: this
    class(base_layer_type), intent(in) :: layer
    integer, dimension(:), optional, intent(in) :: input_list !for now, assume all operations are addition. Later on, make it optional rank 2, and 2nd rank is the operation
    ! in the future, make this a list of base_layer_type and query their id values
    integer, dimension(:), optional, intent(in) :: output_list

    integer :: i, vertex_index

    
    if(.not.allocated(this%model))then
       this%model = [container_layer_type(name=layer%type)]
       this%num_layers = 1
    else
       this%model = [this%model(1:), container_layer_type(name=layer%type)]
       this%num_layers = this%num_layers + 1
    end if
    allocate(this%model(size(this%model,dim=1))%layer, source=layer)
    this%model(size(this%model,dim=1))%layer%id = this%num_layers

    ! directed graph, true
    ! num_vertex_features = 1, representing whether the vertex needs to be updated (for backward only?), 1 = update, 0 = no update
    ! num_edge_features = 1, representing the operation of the edge
    ! edge_index(1) = index of the previous layer
    ! abs(edge_index(2)) = index of the current layer
    ! the sign of edge_index(2) is the direction of the edge, 
    !    +ve = forward
    !    -ve = backward
    ! then, adjacency(i,:) is all of the layers that i feeds forward to
    ! and adjacency(:,i) is all of the layers that feed forward to i (i.e. the backward pass)

    !! have another check. If the layer is an input layer, then it should have no vertex added
    !! vertex feature(1) = layer number
    call this%auto_graph%add_vertex(feature=[1._real32], id=this%num_layers)
    if(present(input_list))then
       do i=1,size(input_list)
          vertex_index = findloc(this%auto_graph%vertex(:)%id, input_list(i))
          call this%auto_graph%add_edge( &
                index = [ vertex_idx, -this%auto_graph%num_vertices ], &
                feature = [ 1._real32 ] &
                )
       end do
    elseif(trim(layer%type).ne."inpt")then
       call this%auto_graph%add_edge( &
            index = [ this%num_layers-1, -this%auto_graph%num_vertices ], &
            feature = [ 1._real32 ] &
       )
    end if

    if(present(output_list))then
       do i=1,size(output_list)
          vertex_index = findloc(this%auto_graph%vertex(:)%id, output_list(i))
          call this%auto_graph%add_edge( &
                index = [ this%auto_graph%num_vertices, -vertex_idx ], &
                feature = [ 1._real32 ] &
                )
       end do
    end if
       
  end subroutine add
!!!#############################################################################


!!!#############################################################################
!!! set up network
!!!#############################################################################
  module function network_setup( &
       layers, optimiser, loss_method, accuracy_method, &
       metrics, batch_size) result(network)
    implicit none
    type(container_layer_type), dimension(:), intent(in) :: layers
    class(base_optimiser_type), optional, intent(in) :: optimiser
    character(*), optional, intent(in) :: loss_method, accuracy_method
    class(*), dimension(..), optional, intent(in) :: metrics
    integer, optional, intent(in) :: batch_size

    type(network_type) :: network

    integer :: l


!!!-----------------------------------------------------------------------------
!!! handle optional arguments
!!!-----------------------------------------------------------------------------
    if(present(loss_method)) call network%set_loss(loss_method)
    if(present(accuracy_method)) call network%set_accuracy(accuracy_method)
    if(present(metrics)) call network%set_metrics(metrics)
    if(present(batch_size)) network%batch_size = batch_size


!!!-----------------------------------------------------------------------------
!!! add layers to network
!!!-----------------------------------------------------------------------------
    do l = 1, size(layers)
       call network%add(layers(l)%layer)
    end do


!!!-----------------------------------------------------------------------------
!!! compile network if optimiser present
!!!-----------------------------------------------------------------------------
    if(present(optimiser)) call network%compile(optimiser)

  end function network_setup
!!!#############################################################################


!!!#############################################################################
!!! set network metrics
!!!#############################################################################
  module subroutine set_metrics(this, metrics)
    use misc, only: to_lower
    implicit none
    class(network_type), intent(inout) :: this
    class(*), dimension(..), intent(in) :: metrics

    integer :: i


    this%metrics%active = .false.
    this%metrics(1)%key = "loss"
    this%metrics(2)%key = "accuracy"
    this%metrics%threshold = 1.E-1_real32
    select rank(metrics)
#if defined(GFORTRAN)
    rank(0)
       select type(metrics)
       type is(character(*))
          !! ERROR: ifort cannot identify that the rank of metrics has been ...
          !! ... identified as scalar here
          where(to_lower(trim(metrics)).eq.this%metrics%key)
             this%metrics%active = .true.
          end where
       end select
#endif
    rank(1)
       select type(metrics)
       type is(character(*))
          do i=1,size(metrics,1)
             where(to_lower(trim(metrics(i))).eq.this%metrics%key)
                this%metrics%active = .true.
             end where
          end do
       type is(metric_dict_type)
          if(size(metrics,1).eq.2)then
             this%metrics(:2) = metrics(:2)
          else
             stop "ERROR: invalid length array for metric_dict_type"
          end if
       end select
    rank default
       stop "ERROR: provided metrics rank in compile invalid"
    end select

  end subroutine set_metrics
!!!#############################################################################


!!!#############################################################################
!!! set network loss
!!!#############################################################################
  module subroutine set_loss(this, loss_method, verbose)
    use misc, only: to_lower
    use loss, only: &
         compute_loss_bce, compute_loss_cce, &
         compute_loss_mae, compute_loss_mse, &
         compute_loss_nll, compute_loss_hubber, &
         compute_loss_hubber_derivative
    implicit none
    class(network_type), intent(inout) :: this
    character(*), intent(in) :: loss_method
    integer, optional, intent(in) :: verbose

    integer :: verbose_
    character(len=:), allocatable :: loss_method_


    if(present(verbose))then
       verbose_ = verbose
    else
       verbose_ = 0
    end if

!!!-----------------------------------------------------------------------------
!!! handle analogous definitions
!!!-----------------------------------------------------------------------------
   loss_method_ = to_lower(loss_method)
   select case(loss_method)
   case("binary_crossentropy")
      loss_method_ = "bce"
   case("categorical_crossentropy")
      loss_method_ = "cce"
   case("mean_absolute_error")
      loss_method_ = "mae"
   case("mean_squared_error")
      loss_method_ = "mse"
   case("negative_log_likelihood")
      loss_method_ = "nll"
   case("hubber")
      loss_method_ = "hub"
   end select

!!!-----------------------------------------------------------------------------
!!! set loss method
!!!-----------------------------------------------------------------------------
   select case(loss_method_)
   case("bce")
      this%get_loss => compute_loss_bce
      this%get_loss_deriv => comp_loss_deriv
      if(verbose_.gt.0) write(*,*) "Loss method: Categorical Cross Entropy"
   case("cce")
      this%get_loss => compute_loss_cce
      this%get_loss_deriv => comp_loss_deriv
      if(verbose_.gt.0) write(*,*) "Loss method: Categorical Cross Entropy"
   case("mae")
      this%get_loss => compute_loss_mae
      this%get_loss_deriv => comp_loss_deriv
      if(verbose_.gt.0) write(*,*) "Loss method: Mean Absolute Error"
   case("mse")
      this%get_loss => compute_loss_mse
      this%get_loss_deriv => comp_loss_deriv
      if(verbose_.gt.0) write(*,*) "Loss method: Mean Squared Error"
   case("nll")
      this%get_loss => compute_loss_nll
      this%get_loss_deriv => comp_loss_deriv
      if(verbose_.gt.0) write(*,*) "Loss method: Negative Log Likelihood"
   case("hub")
      this%get_loss => compute_loss_hubber
      this%get_loss_deriv => compute_loss_hubber_derivative
      if(verbose_.gt.0) write(*,*) "Loss method: Hubber"
   case default
      write(0,*) "Failed loss method: "//trim(loss_method_)
      stop "ERROR: No loss method provided"
   end select

  end subroutine set_loss
!!!#############################################################################


!!!#############################################################################
!!! set network loss
!!!#############################################################################
  module subroutine set_accuracy(this, accuracy_method, verbose)
    use misc, only: to_lower
    use accuracy, only: categorical_score, mae_score, mse_score, rmse_score, &
         r2_score
    implicit none
    class(network_type), intent(inout) :: this
    character(*), intent(in) :: accuracy_method
    integer, optional, intent(in) :: verbose

    integer :: verbose_
    character(len=:), allocatable :: accuracy_method_


    if(present(verbose))then
       verbose_ = verbose
    else
       verbose_ = 0
    end if

!!!-----------------------------------------------------------------------------
!!! handle analogous definitions
!!!-----------------------------------------------------------------------------
   accuracy_method_ = to_lower(accuracy_method)
   select case(accuracy_method)
   case("categorical")
      accuracy_method_ = "cat"
   case("mean_absolute_error")
      accuracy_method_ = "mae"
   case("mean_squared_error")
      accuracy_method_ = "mse"
   case("root_mean_squared_error")
      accuracy_method_ = "rmse"
   case("r2", "r^2", "r squared")
      accuracy_method_ = "r2"
   end select

!!!-----------------------------------------------------------------------------
!!! set accuracy method
!!!-----------------------------------------------------------------------------
   select case(accuracy_method_)
   case("cat")
      this%get_accuracy => categorical_score
      if(verbose_.gt.0) write(*,*) "Accuracy method: Categorical "
   case("mae")
      this%get_accuracy => mae_score
      if(verbose_.gt.0) write(*,*) "Accuracy method: Mean Absolute Error"
   case("mse")
      this%get_accuracy => mse_score
      if(verbose_.gt.0) write(*,*) "Accuracy method: Mean Squared Error"
   case("rmse")
      this%get_accuracy => rmse_score
      if(verbose_.gt.0) write(*,*) "Accuracy method: Root Mean Squared Error"
   case("r2")
      this%get_accuracy => r2_score
      if(verbose_.gt.0) write(*,*) "Accuracy method: R^2"
   case default
      write(0,*) "Failed accuracy method: "//trim(accuracy_method_)
      stop "ERROR: No accuracy method provided"
   end select

  end subroutine set_accuracy
!!!#############################################################################


!!!#############################################################################
!!! reset network
!!!#############################################################################
  module subroutine reset(this)
    implicit none
    class(network_type), intent(inout) :: this

    this%accuracy = 0._real32
    this%loss = huge(1._real32)
    this%batch_size = 0
    this%num_layers = 0
    this%num_outputs = 0
    if(allocated(this%optimiser)) deallocate(this%optimiser)
    call this%set_metrics(["loss"])
    if(allocated(this%model)) deallocate(this%model)
    this%get_loss => null()
    this%get_loss_deriv => null()
    this%get_accuracy => null()

  end subroutine reset
!!!#############################################################################


!!!#############################################################################
!!! compile network
!!!#############################################################################
  module subroutine compile(this, optimiser, loss_method, accuracy_method, &
       metrics, batch_size, verbose)
    implicit none
    class(network_type), intent(inout) :: this
    class(base_optimiser_type), intent(in) :: optimiser
    character(*), optional, intent(in) :: loss_method, accuracy_method
    class(*), dimension(..), optional, intent(in) :: metrics
    integer, optional, intent(in) :: batch_size
    integer, optional, intent(in) :: verbose
    
    integer :: i
    integer :: verbose_ = 0, num_addit_inputs
    integer, dimension(:), allocatable :: input_shape
    class(base_layer_type), allocatable :: t_input_layer, t_flatten_layer


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose)) verbose_ = verbose

    
!!!-----------------------------------------------------------------------------
!!! initialise metrics
!!!-----------------------------------------------------------------------------
    if(present(metrics)) call this%set_metrics(metrics)


!!!-----------------------------------------------------------------------------
!!! initialise loss and accuracy methods
!!!-----------------------------------------------------------------------------
    if(present(loss_method)) call this%set_loss(loss_method, verbose_)
    if(present(accuracy_method)) &
         call this%set_accuracy(accuracy_method, verbose_)


!!!-----------------------------------------------------------------------------
!!! check for input layers at root vertices
!!!-----------------------------------------------------------------------------
    call this%calculate_root_vertices()
    do i = 1, size(this%root_vertices)
       if(.not.allocated(this%model(this%auto_graph%vertex(this%root_vertices(i)))%layer%input_shape))then
          write(0,*) "ERROR: input_shape of first layer not defined"
          stop 1
       end if
       select type(root => this%model(this%auto_graph%vertex(this%root_vertices(i)))%layer)
       class is(input_layer_type)
          cycle
       class is(conv_layer_type)
         input_shape = root%input_shape + [ 2 * root%pad, 0 ]
         root%calc_input_gradients = .false. !make this an option for all learnable layer types? !or just remove the option entirely
       class default
         input_shape = root%input_shape
       end select
       t_input_layer = input_layer_type(&
             input_shape = input_shape, &
             batch_size = this%batch_size, &
             verbose=verbose_ &
       )
       call this%add(t_input_layer, output_list=[this%model(this%auto_graph%vertex(this%root_vertices(i)))%layer%id])
       ! NEED TO CALL layer%init?
       deallocate(input_shape)
       deallocate(t_input_layer)
       this%root_vertices(i) = this%num_layers
    end do


!!!-----------------------------------------------------------------------------
!!! initialise layers
!!!-----------------------------------------------------------------------------
    if(verbose_.gt.0)then
       write(*,*) "layer:",1, this%model(1)%name
       write(*,*) this%model(1)%layer%input_shape
       write(*,*) this%model(1)%layer%output%shape
    end if
    do i=2,size(this%model,dim=1)
       if(.not.allocated(this%model(i)%layer%output)) &
            call this%model(i)%layer%init( &
                 input_shape = this%model(i-1)%layer%output%shape, &
                 batch_size = this%batch_size, &
                 verbose = verbose_ &
            )
       if(verbose_.gt.0)then
          write(*,*) "layer:",i, this%model(i)%name
          write(*,*) this%model(i)%layer%input_shape
          write(*,*) this%model(i)%layer%output%shape
       end if
    end do


!!!-----------------------------------------------------------------------------
!!! check for required flatten layers
!!!-----------------------------------------------------------------------------
    i = 0
    vertex_loop: do
       i = i + 1
       if(i.gt.size(this%auto_graph%num_vertices,dim=1)) exit vertex_loop
       id = this%auto_graph%vertex(i)%id

! NEED AN if(allocated()) STATEMENT FOR output%shape?

       ! get all child vertices
       allocate(child_vertices(0))
       do j = 1, size(this%auto_graph%adjacency(i,:))
          if(this%auto_graph%adjacency(i,j).eq.0) cycle
          child_vertices = [child_vertices, j]
       end do
       child_loop: do j = 1, size(child_vertices)
          child_id = this%auto_graph%vertex(child_vertices(j))%id
          if(trim(this%model(id)%name).eq."flat") cycle

          ! get all parent vertices of the child vertex
          allocate(parent_vertices(0))
          do k = 1, size(this%auto_graph%adjacency(:,child_vertices(j)))
             if(this%auto_graph%adjacency(k,child_vertices(j)).eq.0) cycle
             parent_vertices = [parent_vertices, k]
          end do
          !check if ranks match, rather than input and output shapes
          if(all(this%model(parent_vertices(:))%layer%rank.eq.&
               this%model(child_vertices(j))%layer%rank))then
             cycle child_loop
          end if
          t_flatten_layer = flatten_layer_type(&
               input_shape = this%model(id)%layer%output%shape, &
               batch_size = this%batch_size &
          )

          if(all(this%model(parent_vertices(:))%layer%rank.eq.&
               this%model(id)%layer%rank))then
             ! add flatten layer in the place of the child layer
             call this%auto_graph%remove_edges(indices=[this%auto_graph%adjacency(parent_vertices(:),child_vertices(j))])
             call this%add(t_flatten_layer, input_list=[parent_vertices(:)], output_list=[child_id])
          else
             ! add flatten layer between the current layer and the child layer
             call this%auto_graph%remove_edges(indices=[this%auto_graph%adjacency(i,child_vertices(j))])
             call this%add(t_flatten_layer, input_list=[i], output_list=[child_id])
          end if
          i = i - 1
          deallocate(t_flatten_layer)
          deallocate(child_vertices)
          cycle vertex_loop
       end do child_loop
       deallocate(child_vertices)
    end do vertex_loop
    call this%generate_vertex_order()
    
    !! update number of layers
    !!--------------------------------------------------------------------------
    this%num_layers = size(this%model,dim=1)


    !! set number of outputs
    !!--------------------------------------------------------------------------
    this%num_outputs = 0
    call this%calculate_output_vertices()
    do i = 1, this%output_vertices
       this%num_outputs = this%num_outputs + &
            product(this%model(this%auto_graph%vertex(this%output_vertices(i))%id)%layer%output%shape)
    end do


!!!-----------------------------------------------------------------------------
!!! initialise optimiser
!!!-----------------------------------------------------------------------------
    this%optimiser = optimiser
    call this%optimiser%init(num_params=this%get_num_params())


!!!-----------------------------------------------------------------------------
!!! set batch size, if provided
!!!-----------------------------------------------------------------------------
    if(present(batch_size)) this%batch_size = batch_size
    if(this%batch_size.ne.0)then
       if(this%model(1)%layer%batch_size.ne.0.and.&
            this%model(1)%layer%batch_size.ne.this%batch_size)then
          write(*,*) "WARNING: &
               &batch_size in compile differs from batch_size of input layer"
          write(*,*) "         &
               &batch_size of input layer will be set to network batch_size"
       end if
       call this%set_batch_size(this%batch_size)
    elseif(this%model(1)%layer%batch_size.ne.0)then
       call this%set_batch_size(this%model(1)%layer%batch_size)
    end if

  end subroutine compile
!!!#############################################################################


!!!#############################################################################
!!! set batch size
!!!#############################################################################
  module subroutine set_batch_size(this, batch_size)
     implicit none
     class(network_type), intent(inout) :: this
     integer, intent(in) :: batch_size

     integer :: l

     this%batch_size = batch_size
     do l=1,this%num_layers
        call this%model(l)%layer%set_batch_size(this%batch_size)
     end do

  end subroutine set_batch_size
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! return sample from any rank
!!!#############################################################################
  function get_sample(input, start_index, end_index) result(output)
    implicit none
    integer, intent(in) :: start_index, end_index
    real(real32), dimension(..), intent(in), target :: input
    real(real32), allocatable, dimension(:,:) :: output

    real(real32), pointer :: sample_ptr(:,:) => null()

    select rank(input)
    rank(2)
       sample_ptr(1:size(input(:,1)),1:end_index-start_index+1) => &
            input(:,start_index:end_index)
    rank(3)
       sample_ptr(1:size(input(:,:,1)),1:end_index-start_index+1) => &
            input(:,:,start_index:end_index)
    rank(4)
       sample_ptr(1:size(input(:,:,:,1)),1:end_index-start_index+1) => &
            input(:,:,:,start_index:end_index)
    rank(5)
       sample_ptr(1:size(input(:,:,:,:,1)),1:end_index-start_index+1) => &
            input(:,:,:,:,start_index:end_index)
    rank(6)
       sample_ptr(1:size(input(:,:,:,:,:,1)),1:end_index-start_index+1) => &
            input(:,:,:,:,:,start_index:end_index)
    end select
    output = sample_ptr

  end function get_sample
!!!#############################################################################


!!!#############################################################################
!!! get number of parameters
!!!#############################################################################
  pure module function get_num_params(this) result(num_params)
   implicit none
   class(network_type), intent(in) :: this
   integer :: num_params

   integer :: l

   num_params = 0
   do l = 1, this%num_layers
      num_params = num_params + this%model(l)%layer%get_num_params()
   end do

  end function get_num_params
!!!#############################################################################


!!!#############################################################################
!!! get learnable parameters
!!!#############################################################################
  pure module function get_params(this) result(params)
    implicit none
    class(network_type), intent(in) :: this
    real(real32), allocatable, dimension(:) :: params
  
    integer :: l, start_idx, end_idx
  
    start_idx = 0
    end_idx   = 0
    allocate(params(this%get_num_params()), source=0._real32)
    do l = 1, this%num_layers
       select type(current => this%model(l)%layer)
       class is(learnable_layer_type)
          start_idx = end_idx + 1
          end_idx = end_idx + current%get_num_params()
          params(start_idx:end_idx) = current%get_params()
       end select
    end do
  
  end function get_params
!!!#############################################################################


!!!#############################################################################
!!! set learnable parameters
!!!#############################################################################
  module subroutine set_params(this, params)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(:), intent(in) :: params
  
    integer :: l, start_idx, end_idx
  
    start_idx = 0
    end_idx   = 0
    do l = 1, this%num_layers
       select type(current => this%model(l)%layer)
       class is(learnable_layer_type)
          start_idx = end_idx + 1
          end_idx = end_idx + current%get_num_params()
          call current%set_params(params(start_idx:end_idx))
       end select
    end do
  
  end subroutine set_params
!!!#############################################################################


!!!#############################################################################
!!! get gradients
!!!#############################################################################
  pure module function get_gradients(this) result(gradients)
    implicit none
    class(network_type), intent(in) :: this
    real(real32), allocatable, dimension(:) :: gradients

    integer :: l, start_idx, end_idx

    start_idx = 0
    end_idx   = 0
    allocate(gradients(this%get_num_params()), source=0._real32)
    do l = 1, this%num_layers
       select type(current => this%model(l)%layer)
       class is(learnable_layer_type)
          start_idx = end_idx + 1
          end_idx = end_idx + current%get_num_params()
          gradients(start_idx:end_idx) = &
               current%get_gradients(clip_method=this%optimiser%clip_dict)
       end select
    end do

  end function get_gradients
!!!#############################################################################


!!!#############################################################################
!!! set gradients
!!!#############################################################################
  module subroutine set_gradients(this, gradients)
   implicit none
   class(network_type), intent(inout) :: this
   real(real32), dimension(..), intent(in) :: gradients
 
   integer :: l, start_idx, end_idx
 
   start_idx = 0
   end_idx   = 0
   do l = 1, this%num_layers
      select type(current => this%model(l)%layer)
      class is(learnable_layer_type)
         start_idx = end_idx + 1
         end_idx = end_idx + current%get_num_params()
         select rank(gradients)
         rank(0)
            call current%set_gradients(gradients)
         rank(1)
            call current%set_gradients(gradients(start_idx:end_idx))
         end select
      end select
   end do
 
 end subroutine set_gradients
!!!#############################################################################


!!!#############################################################################
!!! reset gradients
!!!#############################################################################
  module subroutine reset_gradients(this)
   implicit none
   class(network_type), intent(inout) :: this
 
   integer :: l

   do l = 1, this%num_layers
      select type(current => this%model(l)%layer)
      class is(learnable_layer_type)
         call current%set_gradients(0._real32)
      end select
   end do
 
 end subroutine reset_gradients
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! forward pass
!!!#############################################################################
  pure module subroutine forward_1d(this, input, addit_input, layer)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input

    real(real32), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: layer
    
    integer :: i


    !!--------------------------------------------------------------------------
    !! initialise optional arguments
    !!--------------------------------------------------------------------------
    if(present(layer).and.present(addit_input))then
       select type(previous => this%model(layer-1)%layer)
       type is(flatten_layer_type)
          call previous%set_addit_input(addit_input)
       end select
    end if


    !! Forward pass (first layer)
    !!--------------------------------------------------------------------------
    select type(current => this%model(1)%layer)
    class is(input_layer_type)
       call current%set(input)
    end select

    !! Forward pass
    !!--------------------------------------------------------------------------
    !!! GOING TO BE ISSUES WITH INSERTED LAYERS, such as flatten, reshape, input
    !!! these will mess up the vertex and edge indices, and adjacency matrix
    !!! edge id will define operation
    do i = 1, size(this%vertex_order,1)
       

       !! MAKE THIS A FUNCTION, get_input_autodiff
       input = 0._real32
       if(all(this%auto_graph%adjacency(:,this%vertex_order(i)).eq.0))then
          input = input
       else
          do j = 1, this%auto_graph%num_vertices
             if(this%auto_graph%adjacency(j,this%vertex_order(i)).ne.0)then
                select case(this%auto_graph%edge(this%auto_graph%adjacency(j,this%vertex_order(i)))%id)
                case(1)
                  input = input + this%model(this%auto_graph%vertex(j)%id)%layer%output%val
                end select
             end if
          end do
       end if

       call this%model(i)%layer%forward(input)
    end do

  end subroutine forward_1d
!!!#############################################################################


!!!#############################################################################
!!! backward pass
!!!#############################################################################
  pure module subroutine backward_1d(this, output)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: output

    integer :: i
    real(real32), allocatable, dimension(:,:) :: predicted


    !! Backward pass (final layer)
    !!-------------------------------------------------------------------
    call this%model(this%num_layers)%layer%get_output(predicted)
    call this%model(this%num_layers)%backward( &
         this%model(this%num_layers-1), &
         this%get_loss_deriv(predicted, output))


    !! Backward pass
    !!-------------------------------------------------------------------
    do i=this%num_layers-1,2,-1
       select type(gradient => this%model(i+1)%layer%di)
       type is (array1d_type)
          call this%model(i)%backward(this%model(i-1),gradient%val)
       type is (array2d_type)
          call this%model(i)%backward(this%model(i-1),gradient%val)
       type is (array3d_type)
          call this%model(i)%backward(this%model(i-1),gradient%val)
       type is (array4d_type)
          call this%model(i)%backward(this%model(i-1),gradient%val)
      type is (array5d_type)
          call this%model(i)%backward(this%model(i-1),gradient%val)
       end select
    end do

    do i = size(this%vertex_order,1), 1, -1
       
      !! MAKE THIS A FUNCTION, get_input_autodiff
      input = 0._real32
      if(all(this%auto_graph%adjacency(:,this%vertex_order(i)).eq.0))then
         input = input
      else
         do j = 1, this%auto_graph%num_vertices
            if(this%auto_graph%adjacency(j,this%vertex_order(i)).ne.0)then
               select case(this%auto_graph%edge(this%auto_graph%adjacency(j,this%vertex_order(i)))%id)
               case(1)
                 input = input + this%model(this%auto_graph%vertex(j)%id)%layer%output%val
               end select
            end if
         end do
      end if

      gradient = 0._real32
      if(all(this%auto_graph%adjacency(this%vertex_order(i)).eq.0))then
         input = input
      else
         do j = 1, this%auto_graph%num_vertices
            if(this%auto_graph%adjacency(this%vertex_order(i),j).ne.0)then
               select case(this%auto_graph%edge(this%auto_graph%adjacency(this%vertex_order(i),j))%id)
               case(1)
                  gradient = gradient + this%model(this%auto_graph%vertex(j)%id)%layer%di%val
               end select
            end if
         end do
      end if


      call this%model(i)%layer%backward(input, gradient)
   end do

  end subroutine backward_1d
!!!#############################################################################


!!!#############################################################################
!!! update weights and biases
!!!#############################################################################
  module subroutine update(this)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), allocatable, dimension(:) :: params, gradients

    integer :: i
    

    !!-------------------------------------------------------------------
    !! Update layers of learnable layer types
    !!-------------------------------------------------------------------
    params = this%get_params()
    gradients = this%get_gradients()
    call this%optimiser%minimise(params, gradients)
    call this%set_params(params)
    call this%reset_gradients()

    !! Increment optimiser iteration counter
    !!-------------------------------------------------------------------
    this%optimiser%iter = this%optimiser%iter + 1

  end subroutine update
!!!#############################################################################


!!!##########################################################################!!!
!!! * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * !!!
!!!##########################################################################!!!


!!!#############################################################################
!!! training loop
!!! ... loops over num_epoch number of epochs
!!! ... i.e. it trains on the same datapoints num_epoch times
!!!#############################################################################
  module subroutine train(this, input, output, num_epochs, batch_size, &
       addit_input, addit_layer, &
       plateau_threshold, shuffle_batches, batch_print_step, verbose)
    use infile_tools, only: stop_check
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    class(*), dimension(:,:), intent(in) :: output
    integer, intent(in) :: num_epochs
    integer, optional, intent(in) :: batch_size !! deprecated

    real(real32), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: addit_layer

    real(real32), optional, intent(in) :: plateau_threshold
    logical, optional, intent(in) :: shuffle_batches
    integer, optional, intent(in) :: batch_print_step
    integer, optional, intent(in) :: verbose
    
    !! training and testing monitoring
    real(real32) :: batch_loss, batch_accuracy, avg_loss, avg_accuracy
    real(real32), allocatable, dimension(:,:) :: y_true

    !! learning parameters
    integer :: l, num_samples
    integer :: num_batches
    integer :: converged
    integer :: history_length
    integer :: verbose_
    integer :: batch_print_step_
    real(real32) :: plateau_threshold_
    logical :: shuffle_batches_

    !! training loop variables
    integer :: epoch, batch, start_index, end_index
    integer, allocatable, dimension(:) :: batch_order

    integer :: i, time, time_old, clock_rate

#ifdef _OPENMP
    type(network_type) :: this_copy
    real(real32), allocatable, dimension(:,:) :: input_slice, addit_input_slice
#endif
    integer :: timer_start = 0, timer_stop = 0, timer_sum = 0, timer_tot = 0


!!!-----------------------------------------------------------------------------
!!! check loss and accuracy methods are set
!!!-----------------------------------------------------------------------------
    if(.not.associated(this%get_loss)) stop "ERROR: loss method not set"
    if(.not.associated(this%get_accuracy)) stop "ERROR: accuracy method not set"


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    verbose_ = 0
    batch_print_step_ = 20 
    plateau_threshold_ = 1.E-2_real32
    shuffle_batches_ = .true.
    if(present(plateau_threshold)) plateau_threshold_ = plateau_threshold
    if(present(shuffle_batches)) shuffle_batches_ = shuffle_batches
    if(present(batch_print_step)) batch_print_step_ = batch_print_step
    if(present(verbose)) verbose_ = verbose
    if(present(batch_size)) this%batch_size = batch_size


!!!-----------------------------------------------------------------------------
!!! initialise monitoring variables
!!!-----------------------------------------------------------------------------
    history_length = max(ceiling(500._real32/this%batch_size),1)
    do i=1,size(this%metrics,dim=1)
       if(allocated(this%metrics(i)%history)) &
            deallocate(this%metrics(i)%history)
       allocate(this%metrics(i)%history(history_length))
       this%metrics(i)%history = -huge(1._real32)
    end do


!!!-----------------------------------------------------------------------------
!!! allocate predicted and true label sets
!!!-----------------------------------------------------------------------------
    allocate(y_true(this%num_outputs,this%batch_size), source = 0._real32)


!!!-----------------------------------------------------------------------------
!!! if parallel, initialise slices
!!!-----------------------------------------------------------------------------
    num_batches = size(output,dim=2) / this%batch_size
    allocate(batch_order(num_batches))
    do batch = 1, num_batches
       batch_order(batch) = batch
    end do


!!!-----------------------------------------------------------------------------
!!! get number of samples
!!!-----------------------------------------------------------------------------
    select rank(input)
    rank(1)
       write(*,*) "Cannot check number of samples in rank 1 input"
    rank default
       num_samples = size(input,rank(input))
       if(size(output,2).ne.num_samples)then
          write(0,*) "ERROR: number of samples in input and output do not match"
          stop "Exiting..."
       elseif(size(output,1).ne.this%num_outputs)then
          write(0,*) "ERROR: number of outputs in output does not match network"
          stop "Exiting..."
       end if
     end select


!!!-----------------------------------------------------------------------------
!!! set/reset batch size for training
!!!-----------------------------------------------------------------------------
    call this%set_batch_size(this%batch_size)


!!!-----------------------------------------------------------------------------
!!! turn off inference booleans
!!!-----------------------------------------------------------------------------
    do l=1,this%num_layers
       select type(current => this%model(l)%layer)
       class is(drop_layer_type)
          current%inference = .false.
       end select
    end do


!!!-----------------------------------------------------------------------------
!!! query system clock
!!!-----------------------------------------------------------------------------
    call system_clock(time, count_rate = clock_rate)


    epoch_loop: do epoch = 1, num_epochs
       !!-----------------------------------------------------------------------
       !! shuffle batch order at the start of each epoch
       !!-----------------------------------------------------------------------
       if(shuffle_batches_)then
          call shuffle(batch_order)
       end if

       avg_loss     = 0._real32
       avg_accuracy = 0._real32

       !!-----------------------------------------------------------------------
       !! batch loop
       !! ... split data up into minibatches for training
       !!-----------------------------------------------------------------------
       batch_loop: do batch = 1, num_batches


          !! set batch start and end index
          !!--------------------------------------------------------------------
          start_index = (batch_order(batch) - 1) * this%batch_size + 1
          end_index = batch_order(batch) * this%batch_size
          

          !! reinitialise variables
          !!--------------------------------------------------------------------
          select type(output)
          type is(integer)
             y_true(:,:) = real(output(:,start_index:end_index:1),real32)
          type is(real)
             y_true(:,:) = output(:,start_index:end_index:1)
          end select


          !! Forward pass
          !!--------------------------------------------------------------------
          if(present(addit_input).and.present(addit_layer))then
             call this%forward(get_sample(input,start_index,end_index),&
                addit_input(:,start_index:end_index),addit_layer)
          else
             call this%forward(get_sample(input,start_index,end_index))
          end if


          !! Backward pass and store predicted output
          !!--------------------------------------------------------------------
          call this%backward(y_true(:,:))
          select type(output => this%model(this%num_layers)%layer%output)
          type is(array2d_type)
             !! compute loss and accuracy (for monitoring)
             !!-----------------------------------------------------------------
             batch_loss = sum( &
                  this%get_loss( &
                  output%val(:,1:this%batch_size), &
                  y_true(:,1:this%batch_size)))
             batch_accuracy = sum( &
                  this%get_accuracy( &
                  output%val(:,1:this%batch_size), &
                  y_true(:,1:this%batch_size)))
          class default
             stop "ERROR: final layer output not 2D"
          end select



          !! Average metric over batch size and store
          !! Check metric convergence
          !!--------------------------------------------------------------------
          avg_loss = avg_loss + batch_loss
          avg_accuracy = avg_accuracy + batch_accuracy
          this%metrics(1)%val = batch_loss / this%batch_size
          this%metrics(2)%val = batch_accuracy / this%batch_size
          do i = 1, size(this%metrics,dim=1)
             call this%metrics(i)%check(plateau_threshold_, converged)
             if(converged.ne.0)then
                exit epoch_loop
             end if
          end do


          !! update weights and biases using optimization algorithm
          !! ... (gradient descent)
          !!--------------------------------------------------------------------
          !! STORE ADAM VALUES IN OPTIMISER
          call this%update()


          !! print batch results
          !!--------------------------------------------------------------------
          if(abs(verbose_).gt.0.and.&
               (batch.eq.1.or.mod(batch,batch_print_step_).eq.0.E0))then
             write(6,'("epoch=",I0,", batch=",I0,&
                  &", learning_rate=",F0.3,", loss=",F0.3,", accuracy=",F0.3)')&
                  epoch, batch, &
                  this%optimiser%learning_rate, &
                  avg_loss/(batch*this%batch_size), &
                  avg_accuracy/(batch*this%batch_size)
          end if
          
          
!!! TESTING
!#ifdef _OPENMP
!          call system_clock(timer_start)
!          call system_clock(timer_stop)
!          timer_sum = timer_sum + timer_stop - timer_start
!          timer_tot = timer_tot + timer_sum / omp_get_max_threads()
!#else
!          timer_tot = timer_tot + timer_sum
!#endif
!          timer_sum = 0
!           if(batch.gt.200)then
!              time_old = time
!              call system_clock(time)
!              write(*,'("time check: ",F8.3," seconds")') real(time-time_old)/clock_rate
!              !write(*,'("update timer: ",F8.3," seconds")') real(timer_tot)/clock_rate
!              exit epoch_loop
!              stop "THIS IS FOR TESTING PURPOSES"
!           end if
!!!


          !! time check
          !!--------------------------------------------------------------------
          if(verbose_.eq.-2)then
             time_old = time
             call system_clock(time)
             write(*,'("time check: ",F5.3," seconds")') &
                  real(time-time_old)/clock_rate
             time_old = time
          end if


          !! check for user-name stop file
          !!--------------------------------------------------------------------
          if(stop_check())then
             write(0,*) "STOPCAR ENCOUNTERED"
             write(0,*) "Exiting training loop..."
             exit epoch_loop
          end if

       end do batch_loop


       !! print epoch summary results
       !!-----------------------------------------------------------------------
       if(verbose_.eq.0)then
          write(6,'("epoch=",I0,&
               &", learning_rate=",F0.3,", val_loss=",F0.3,&
               &", val_accuracy=",F0.3)') &
               epoch, &
               this%optimiser%learning_rate, &
               this%metrics(1)%val, this%metrics(2)%val
       end if


    end do epoch_loop

  end subroutine train
!!!#############################################################################


!!!#############################################################################
!!! testing loop
!!!#############################################################################
  module subroutine test(this, input, output, &
       addit_input, addit_layer, &
       verbose)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    class(*), dimension(:,:), intent(in) :: output

    real(real32), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: addit_layer

    integer, optional, intent(in) :: verbose

    integer :: l, sample, num_samples
    integer :: verbose_, unit
    real(real32) :: acc_val, loss_val
    real(real32), allocatable, dimension(:) :: accuracy_list
    real(real32), allocatable, dimension(:,:) :: predicted, y_true


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
    if(present(verbose))then
       verbose_ = verbose
    else
       verbose_ = 0
    end if
    num_samples = size(output, dim=2)
    allocate(predicted(size(output,1), num_samples))

    this%metrics%val = 0._real32
    acc_val  = 0._real32
    loss_val = 0._real32
    allocate(accuracy_list(num_samples))


    select type(output)
    type is(integer)
       y_true = real(output(:,:),real32)
    type is(real)
       y_true = output(:,:)
    end select


!!!-----------------------------------------------------------------------------
!!! reset batch size for testing
!!!-----------------------------------------------------------------------------
    call this%set_batch_size(1)


!!!-----------------------------------------------------------------------------
!!! turn on inference booleans
!!!-----------------------------------------------------------------------------
    do l=1,this%num_layers
       select type(current => this%model(l)%layer)
       class is(drop_layer_type)
          current%inference = .true.
       end select
    end do


!!!-----------------------------------------------------------------------------
!!! testing loop
!!!-----------------------------------------------------------------------------
    test_loop1: do sample = 1, num_samples

       !! Forward pass
       !!-----------------------------------------------------------------------
       if(present(addit_input).and.present(addit_layer))then
          call this%forward(get_sample(input,sample,sample),&
               addit_input(:,sample:sample),addit_layer)
       else
          call this%forward(get_sample(input,sample,sample))
       end if


       !! compute loss and accuracy (for monitoring)
       !!-----------------------------------------------------------------------
       select type(output_arr => this%model(this%num_layers)%layer%output)
       type is(array2d_type)
          loss_val = sum(this%get_loss( &
               predicted = output_arr%val, &
               !!!! JUST REPLACE y_true(:,sample) WITH output(:,sample) !!!!
               !!!! THERE IS NO REASON TO USE y_true, as it is just a copy !!!!
               !!!! get_loss should handle both integers and reals !!!!
               !!!! it does not. Instead just wrap real(output(:,sample),real32) !!!!
               expected  = y_true(:,sample:sample)))
          acc_val = sum(this%get_accuracy( &
               predicted = output_arr%val, &
               expected  = y_true(:,sample:sample)))
          this%metrics(2)%val = this%metrics(2)%val + acc_val
          this%metrics(1)%val = this%metrics(1)%val + loss_val
          accuracy_list(sample) = acc_val
          predicted(:,sample) = output_arr%val(:,1)
       end select

    end do test_loop1

    
    !! print testing results
    !!--------------------------------------------------------------------
    if(abs(verbose_).gt.1)then
       open(file="test_output.out",newunit=unit)
       test_loop2: do concurrent(sample = 1:num_samples)
          select type(output)
          type is(integer)
             write(unit,'(I4," Expected=",I3,", Got=",I3,", Accuracy=",F0.3)') &
                  sample, &
                  maxloc(output(:,sample)), maxloc(predicted(:,sample),dim=1)-1, &
                  accuracy_list(sample)
          type is(real)
             write(unit,'(I4," Expected=",F0.3,", Got=",F0.3,", Accuracy=",F0.3)') &
                  sample, &
                  output(:,sample), predicted(:,sample), &
                  accuracy_list(sample)
          end select
       end do test_loop2
       close(unit)
    end if


    !! normalise metrics by number of samples
    !!--------------------------------------------------------------------
    this%accuracy = this%metrics(2)%val/real(num_samples)
    this%loss     = this%metrics(1)%val/real(num_samples)

  end subroutine test
!!!#############################################################################


!!!#############################################################################
!!! predict outputs from input data using trained network
!!!#############################################################################
  module function predict_1d(this, input, &
       addit_input, addit_layer, &
       verbose) result(output)
    implicit none
    class(network_type), intent(inout) :: this
    real(real32), dimension(..), intent(in) :: input
    
    real(real32), dimension(:,:), optional, intent(in) :: addit_input
    integer, optional, intent(in) :: addit_layer
    
    integer, optional, intent(in) :: verbose

    real(real32), dimension(:,:), allocatable :: output
    
    integer :: verbose_, batch_size


!!!-----------------------------------------------------------------------------
!!! initialise optional arguments
!!!-----------------------------------------------------------------------------
   if(present(verbose))then
      verbose_ = verbose
   else
      verbose_ = 0
   end if

   select rank(input)
   rank(2)
      batch_size = size(input,dim=2)
   rank(3)
      batch_size = size(input,dim=3)
   rank(4)
      batch_size = size(input,dim=4)
   rank(5)
      batch_size = size(input,dim=5)
   rank(6)
      batch_size = size(input,dim=6)
   rank default
      batch_size = size(input,dim=rank(input))
   end select
   allocate(output(this%num_outputs,batch_size))


!!!-----------------------------------------------------------------------------
!!! reset batch size for testing
!!!-----------------------------------------------------------------------------
   call this%set_batch_size(batch_size)


!!!-----------------------------------------------------------------------------
!!! predict
!!!-----------------------------------------------------------------------------
   if(present(addit_input).and.present(addit_layer))then
      call this%forward(get_sample(input,1,batch_size),&
           addit_input(:,1:batch_size),addit_layer)
   else
      call this%forward(get_sample(input,1,batch_size))
   end if

   select type(output_arr => this%model(this%num_layers)%layer%output)
   type is(array2d_type)
      output = output_arr%val(:,1:batch_size)
   end select

  end function predict_1d
!!!#############################################################################

end submodule network_submodule
!!!#############################################################################
