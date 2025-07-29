module read_chemical_graphs_extd
  use constants_mnist, only: real32
  use misc_linalg, only: modu
  use rw_geom, only: bas_type, geom_read, igeom_input
  use athena, only: graph_type, edge_type, array2d_type
  use read_chemical_graphs, only: get_elements_masses_and_charges
  implicit none

  private

  public :: read_extxyz_db

contains



!!!#############################################################################
!!! read extxyz
!!!#############################################################################
  subroutine read_extxyz_db(file,graphs,labels)
    implicit none
    character(1024), intent(in) :: file
    type(graph_type), allocatable, dimension(:,:), intent(out) :: graphs
    type(array2d_type), dimension(1,1), intent(out) :: labels

    integer :: ierror, unit
    real(real32) :: label
    type(graph_type) :: graph
    character(100) :: iomsg
    character(1024) :: buffer

    integer :: num_samples
    type(bas_type) :: basis
    real(real32), dimension(3,3) :: lattice
    real(real32), allocatable, dimension(:) :: labels_tmp
    type(graph_type), allocatable, dimension(:) :: graphs_tmp


    open(newunit=unit, file=file, status='old', action='read', iostat=ierror)

    write(*,*) "Reading data from file: ", trim(file)

    igeom_input = 6
    allocate(graphs_tmp(0))
    allocate(labels_tmp(0))
    do
       read(unit,'(A)',iostat=ierror) buffer
       if(ierror .ne. 0) exit
       if(trim(buffer).eq."") cycle
       backspace(unit)
       call geom_read(unit, lattice, basis)
       call get_elements_masses_and_charges(basis)
       graphs_tmp = [ graphs_tmp, get_input_graph_from_basis(lattice, basis) ]
       labels_tmp = [ labels_tmp, basis%energy ]
    end do
    close(unit)
    allocate(graphs(1, size(graphs_tmp)))
    graphs(1,:) = graphs_tmp
    call labels(1,1)%allocate(array_shape=[1,size(graphs_tmp)])
    labels(1,1)%val(1,:) = labels_tmp

  end subroutine read_extxyz_db
!!!#############################################################################


!!!#############################################################################
!!! convert basis to graph
!!!#############################################################################
  function get_input_graph_from_basis(lattice, basis) result(graph)
    implicit none
    type(bas_type), intent(in) :: basis
    real(real32), dimension(3,3), intent(in) :: lattice
    type(graph_type) :: graph

    integer :: is, ia, js, ja, i, j, k, degree
    integer :: iatom, jatom
    integer :: amax, bmax, cmax
    type(edge_type) :: edge
    real(real32) :: rtmp1, cutoff_min, cutoff_max
    real(real32), dimension(3) :: diff, vtmp1


    graph%num_vertices = basis%natom
    graph%num_vertex_features = 9
    graph%num_edge_features = 1


    allocate(graph%vertex(graph%num_vertices))

    iatom = 0
    do is = 1, basis%nspec
       do ia = 1, basis%spec(is)%num
          iatom = iatom + 1
          allocate(graph%vertex(iatom)%feature(graph%num_vertex_features))
          graph%vertex(iatom)%feature = [ basis%spec(is)%charge / 100._real32, &
               basis%spec(is)%mass / 52._real32 ]
          graph%vertex(iatom)%id = iatom
       end do
    end do

    cutoff_min = 0.5_real32
    cutoff_max = 3.0_real32
    amax = ceiling(cutoff_max/modu(lattice(1,:)))
    bmax = ceiling(cutoff_max/modu(lattice(2,:)))
    cmax = ceiling(cutoff_max/modu(lattice(3,:)))

    iatom = 0
    allocate(graph%edge(0))
    spec_loop1: do is=1,basis%nspec
       atom_loop1: do ia=1,basis%spec(is)%num
          iatom = iatom + 1
          jatom = 0
          degree = 0
          spec_loop2: do js=is,basis%nspec
             atom_loop2: do ja=1,basis%spec(js)%num
                jatom = jatom + 1
                if(is.eq.js.and.ja.lt.ia) cycle atom_loop2
                diff = basis%spec(is)%atom(ia,:3) -  basis%spec(js)%atom(ja,:3)
                diff = diff - ceiling(diff - 0.5_real32)
                do i=-amax,amax+1,1
                   vtmp1(1) = diff(1) + real(i, real32)
                   do j=-bmax,bmax+1,1
                      vtmp1(2) = diff(2) + real(j, real32)
                      do k=-cmax,cmax+1,1
                         vtmp1(3) = diff(3) + real(k, real32)
                         rtmp1 = modu(matmul(vtmp1,lattice))
                         if( rtmp1 .gt. cutoff_min .and. &
                              rtmp1 .lt. cutoff_max )then
                            degree = degree + 1
                            edge%index = [iatom,jatom]
                            edge%feature = [ rtmp1 / cutoff_max ]
                            graph%edge = [ graph%edge, edge ]
                            graph%edge(size(graph%edge))%id = size(graph%edge)
                         end if
                      end do
                   end do
                end do
             end do atom_loop2
          end do spec_loop2
          graph%vertex(iatom)%feature = [ &
               basis%spec(is)%atom(ia,:), &
               basis%spec(is)%force(ia,:), &
               graph%vertex(iatom)%feature, &
               real(degree, real32) / 6._real32 &
          ]
       end do atom_loop1
    end do spec_loop1
    graph%num_edges = size(graph%edge)
    call graph%convert_to_sparse()
    call graph%generate_adjacency()

  end function get_input_graph_from_basis
!!!#############################################################################

end module read_chemical_graphs_extd
