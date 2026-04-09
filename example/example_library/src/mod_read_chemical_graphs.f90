module read_chemical_graphs
  use coreutils, only: real32
  use misc_linalg, only: modu
  use atomstruc, only: basis_type, geom_read, igeom_input
  use graphstruc, only: graph_type, edge_type
  use diffstruc, only: array_type
  implicit none

  private

  public :: read_mp_db, read_extxyz_db

contains

!!!#############################################################################
!!! get line of unknown length
!!!#############################################################################
!!! read line of unknown length from file
!!! implemented as written by IanH in stackoverflow:
!!! https://stackoverflow.com/questions/34746461/how-to-read-text-files-with-possibly-very-long-lines-in-fortran
  subroutine get_line(lun, line, iostat, iomsg)
    integer, intent(in)           :: lun
    character(len=:), intent(out), allocatable :: line
    integer, intent(out)          :: iostat
    character(*), intent(inout)   :: iomsg

    integer, parameter            :: buffer_len = 1024
    character(len=buffer_len)     :: buffer
    integer                       :: size_read

    line = ''
    do
       read ( lun, '(A)',  &
            iostat = iostat,  &
            iomsg = iomsg,  &
            advance = 'no',  &
            size = size_read ) buffer
       if (is_iostat_eor(iostat)) then
          line = line // buffer(:size_read)
          iostat = 0
          exit
       else if (iostat == 0) then
          line = line // buffer
       else
          exit
       end if
       !   write(*,*) line
    end do
  end subroutine get_line
!!!#############################################################################


!!!#############################################################################
!!! read mnist dataset
!!!#############################################################################
  subroutine read_mp_db(file,graphs,labels)
    use coreutils, only: icount
    implicit none
    integer ::v, e, pos_i, pos_f, length, Reason, unit
    character(:), allocatable :: line
    real(real32) :: label
    type(graph_type) :: graph
    character(100) :: iomsg

    integer :: num_samples
    character(1024), intent(in) :: file
    type(graph_type), allocatable, dimension(:), intent(out) :: graphs
    real(real32), allocatable, dimension(:), intent(out) :: labels


    open(newunit=unit, file=file, status='old', action='read', iostat=Reason)

    write(*,*) "Reading data from file: ", trim(file)
    call get_line(unit, line, Reason, iomsg)
    write(*,*) "Line read"

    num_samples = 0
    pos_i = 1
    graph%num_vertex_features = 1
    graph%num_edge_features = 1
    length = len(line)
    do
       if (index(line(pos_i:), 'graph') .eq. 0) exit
       if(num_samples.gt.10) exit
       write(*,*) "pos_i: ", pos_i

       !! get graph name
       pos_i = pos_i + index(line(pos_i:), '"material_id": "') - 1 + 16
       pos_f = pos_i + index(line(pos_i:), '"') - 2
       graph%name = line(pos_i:pos_i+pos_f-1)
       pos_i = pos_f

       pos_i = pos_i + index(line(pos_i:), 'graph') - 1

       pos_i = pos_i + index(line(pos_i:), '"atom": [') - 1 + 9
       pos_f = pos_i + index(line(pos_i:), ']') - 2
       graph%num_vertices = icount(line(pos_i:pos_f), ',')
       allocate(graph%vertex(graph%num_vertices))
       do v = 1, graph%num_vertices
          allocate(graph%vertex(v)%feature(graph%num_vertex_features))
       end do
       read(line(pos_i:pos_f),*) &
            (graph%vertex(v)%feature(1), v=1,graph%num_vertices)
       pos_i = pos_f + 1

       pos_i = pos_i + index(line(pos_i:), '"bond": [') - 1 + 9
       pos_f = pos_i + index(line(pos_i:), ']') - 2
       graph%num_edges = icount(line(pos_i:pos_f), ',')
       allocate(graph%edge(graph%num_edges))
       do e = 1, graph%num_edges
          allocate(graph%edge(e)%feature(graph%num_edge_features))
       end do
       read(line(pos_i:pos_f),*) (graph%edge(e)%feature(1), e=1,graph%num_edges)
       pos_i = pos_f + 1

       pos_i = pos_i + index(line(pos_i:), '"index1": [') - 1 + 11
       pos_f = pos_i + index(line(pos_i:), ']') - 2
       read(line(pos_i:pos_f),*) (graph%edge(e)%index(1), e=1,graph%num_edges)
       pos_i = pos_f + 1

       pos_i = pos_i + index(line(pos_i:), '"index2": [') - 1 + 11
       pos_f = pos_i + index(line(pos_i:), ']') - 2
       read(line(pos_i:pos_f),*) (graph%edge(e)%index(2), e=1,graph%num_edges)

       pos_i = pos_i + index(line(pos_i:), '"formation_energy_per_atom": ') - &
            1 + 30
       pos_f = pos_i + index(line(pos_i:), ',') - 2
       read(line(pos_i:),*) label

       graphs = [graphs, graph]
       labels = [labels, label]
       num_samples = num_samples + 1

    end do

    write(6,*) "Data read"

    return
  end subroutine read_mp_db
!!!#############################################################################


!!!#############################################################################
!!! read extxyz
!!!#############################################################################
  subroutine read_extxyz_db(file,graphs,labels)
    implicit none
    character(1024), intent(in) :: file
    type(graph_type), allocatable, dimension(:,:), intent(out) :: graphs
    type(array_type), dimension(1,1), intent(out) :: labels

    integer :: ierror, unit
    real(real32) :: label
    type(graph_type) :: graph
    character(100) :: iomsg
    character(1024) :: buffer

    integer :: num_samples
    type(basis_type) :: basis
    type(graph_type) :: graph_tmp
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
       call geom_read(unit, basis)
       call basis%set_element_properties_to_default()
       graph_tmp = get_graph_from_basis(basis)
       graphs_tmp = [ graphs_tmp, graph_tmp ]
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
  function get_graph_from_basis(basis) result(graph)
    implicit none
    type(basis_type), intent(in) :: basis
    type(graph_type) :: graph

    integer :: is, ia, js, ja, i, j, k, degree
    integer :: iatom, jatom
    integer :: amax, bmax, cmax
    type(edge_type) :: edge
    real(real32) :: rtmp1, cutoff_min, cutoff_max
    real(real32), dimension(3) :: diff, vtmp1


    graph%directed = .false.
    graph%num_vertices = basis%natom
    graph%num_vertex_features = 6
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
    amax = ceiling(cutoff_max/modu(basis%lat(1,:)))
    bmax = ceiling(cutoff_max/modu(basis%lat(2,:)))
    cmax = ceiling(cutoff_max/modu(basis%lat(3,:)))

    iatom = 0
    allocate(graph%edge(0))
    spec_loop1: do is = 1, basis%nspec
       atom_loop1: do ia = 1, basis%spec(is)%num
          iatom = iatom + 1
          jatom = 0
          degree = 0
          spec_loop2: do js = is, basis%nspec
             atom_loop2: do ja = 1, basis%spec(js)%num
                jatom = jatom + 1
                if(is.eq.js.and.ja.lt.ia) cycle atom_loop2
                diff = basis%spec(is)%atom(:3,ia) -  basis%spec(js)%atom(:3,ja)
                diff = diff - ceiling(diff - 0.5_real32)
                do i=-amax,amax+1,1
                   vtmp1(1) = diff(1) + real(i, real32)
                   do j=-bmax,bmax+1,1
                      vtmp1(2) = diff(2) + real(j, real32)
                      do k=-cmax,cmax+1,1
                         vtmp1(3) = diff(3) + real(k, real32)
                         rtmp1 = modu(matmul(vtmp1,basis%lat))
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
               basis%spec(is)%force(:,ia), &
               graph%vertex(iatom)%feature, &
               real(degree, real32) / 6._real32 &
          ]
       end do atom_loop1
    end do spec_loop1
    graph%num_edges = size(graph%edge)
    call graph%convert_to_sparse()
    call graph%generate_adjacency()

  end function get_graph_from_basis
!!!#############################################################################

end module read_chemical_graphs
