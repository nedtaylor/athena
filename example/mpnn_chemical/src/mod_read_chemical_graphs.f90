module read_chemical_graphs
  use constants_mnist, only: real32
  use misc_linalg, only: modu
  use rw_geom, only: bas_type, geom_read, igeom_input
  use athena, only: graph_type, edge_type
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
     write(*,*) line
   end do
 end subroutine get_line
!!!#############################################################################


!!!#############################################################################
!!! read mnist dataset
!!!#############################################################################
  subroutine read_mp_db(file,graphs,labels)
    use misc_mnist, only: icount
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
       read(line(pos_i:pos_f),*) (graph%vertex(v)%feature(1), v=1,graph%num_vertices)
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

       pos_i = pos_i + index(line(pos_i:), '"formation_energy_per_atom": ') - 1 + 30
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
    type(graph_type), allocatable, dimension(:), intent(out) :: graphs
    real(real32), allocatable, dimension(:), intent(out) :: labels

    integer :: ierror, unit
    real(real32) :: label
    type(graph_type) :: graph
    character(100) :: iomsg
    character(1024) :: buffer

    integer :: num_samples
    type(bas_type) :: basis
    real(real32), dimension(3,3) :: lattice


    open(newunit=unit, file=file, status='old', action='read', iostat=ierror)

    write(*,*) "Reading data from file: ", trim(file)

    igeom_input = 6
    allocate(graphs(0))
    allocate(labels(0))
    do
       read(unit,'(A)',iostat=ierror) buffer
       if(ierror .ne. 0) exit
       if(trim(buffer).eq."") cycle
       backspace(unit)
       write(*,*) trim(buffer)
       call geom_read(unit, lattice, basis)
       call get_elements_masses_and_charges(basis)
       graphs = [ graphs, get_graph_from_basis(lattice, basis) ]
       labels = [ labels, basis%energy ]
    end do
    close(unit)

  end subroutine read_extxyz_db
!!!#############################################################################


!!!#############################################################################
!!! convert basis to graph
!!!#############################################################################
  function get_graph_from_basis(lattice, basis) result(graph)
    implicit none
    type(bas_type), intent(in) :: basis
    real(real32), dimension(3,3), intent(in) :: lattice
    type(graph_type) :: graph

    integer :: is, ia, js, ja, i, j, k
    integer :: iatom, jatom
    integer :: amax, bmax, cmax
    type(edge_type) :: edge
    real(real32) :: rtmp1, cutoff_min, cutoff_max
    real(real32), dimension(3) :: diff, vtmp1

    
    graph%num_vertices = basis%natom
    graph%num_vertex_features = 2
    graph%num_edge_features = 1


    allocate(graph%vertex(graph%num_vertices))

    iatom = 0
    do is = 1, basis%nspec
       do ia = 1, basis%spec(is)%num
          iatom = iatom + 1
          allocate(graph%vertex(iatom)%feature(graph%num_vertex_features))
          graph%vertex(iatom)%feature = [ basis%spec(is)%charge / 100._real32, &
               basis%spec(is)%mass / 52._real32 ]
       end do
    end do

    cutoff_min = 0.5_real32
    cutoff_max = 6.0_real32
    amax = ceiling(cutoff_max/modu(lattice(1,:)))
    bmax = ceiling(cutoff_max/modu(lattice(2,:)))
    cmax = ceiling(cutoff_max/modu(lattice(3,:)))

    iatom = 0
    allocate(graph%edge(0))
    spec_loop1: do is=1,basis%nspec
       atom_loop1: do ia=1,basis%spec(is)%num
          iatom = iatom + 1
          jatom = 0
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
                            edge%index = [iatom,jatom]
                            edge%feature = [rtmp1]
                            graph%edge = [ graph%edge, edge ]
                         end if
                      end do
                   end do
                end do
             end do atom_loop2
          end do spec_loop2
       end do atom_loop1
    end do spec_loop1
    graph%num_edges = size(graph%edge)
    call graph%generate_adjacency()
    call graph%calculate_degree()


  end function get_graph_from_basis
!!!#############################################################################


!!!#############################################################################
!!! get elements masses and charges
!!!#############################################################################
  subroutine get_elements_masses_and_charges(basis)
    implicit none
    type(bas_type), intent(inout) :: basis

    integer :: i
    real(real32) :: mass, charge

    do i = 1, basis%nspec
       select case(basis%spec(i)%name)
       case('H')
          mass = 1.00784_real32
          charge = 1.0_real32
       case('He')
          mass = 4.0026_real32
          charge = 2.0_real32
       case('Li')
          mass = 6.94_real32
          charge = 3.0_real32
       case('Be')
          mass = 9.0122_real32
          charge = 4.0_real32
       case('B')
          mass = 10.81_real32
          charge = 5.0_real32
       case('C')
          mass = 12.011_real32
          charge = 4.0_real32
       case('N')
          mass = 14.007_real32
          charge = 5.0_real32
       case('O')
          mass = 15.999_real32
          charge = 6.0_real32
       case('F')
          mass = 18.998_real32
          charge = 7.0_real32
       case('Na')
          mass = 22.989_real32
          charge = 1.0_real32
       case('Mg')
          mass = 24.305_real32
          charge = 2.0_real32
       case('Al')
          mass = 26.982_real32
          charge = 3.0_real32
       case('Si')
          mass = 28.085_real32
          charge = 4.0_real32
       case('P')
          mass = 30.974_real32
          charge = 5.0_real32
       case('S')  
          mass = 32.06_real32
          charge = 6.0_real32
       case('Cl')
          mass = 35.453_real32
          charge = 8.0_real32
       case('K')
          mass = 39.098_real32
          charge = 1.0_real32
       case('Ca')
          mass = 40.078_real32
          charge = 2.0_real32
       case('Sc')
          mass = 44.956_real32
          charge = 3.0_real32
       case('Ti')
          mass = 47.867_real32
          charge = 4.0_real32
       case('V')
          mass = 50.942_real32
          charge = 5.0_real32
       case('Cr')
          mass = 51.996_real32
          charge = 6.0_real32
       case('Mn')
          mass = 54.938_real32
          charge = 7.0_real32
       case('Fe')
          mass = 55.845_real32
          charge = 8.0_real32
       case('Co')
          mass = 58.933_real32
          charge = 9.0_real32
       case('Ni')
          mass = 58.693_real32
          charge = 10.0_real32
       case('Cu')
          mass = 63.546_real32
          charge = 11.0_real32
       case('Zn')
          mass = 65.38_real32
          charge = 12.0_real32
       case('Ga')
          mass = 69.723_real32
          charge = 13.0_real32
       case('Ge')
          mass = 72.63_real32
          charge = 14.0_real32
       case('As')
          mass = 74.922_real32
          charge = 15.0_real32
       case('Se')
          mass = 78.971_real32
          charge = 16.0_real32
       case('Br')
          mass = 79.904_real32
          charge = 17.0_real32
       case('Kr')
          mass = 83.798_real32
          charge = 18.0_real32
       case('Rb')
          mass = 85.468_real32
          charge = 19.0_real32
       case('Sr')
          mass = 87.62_real32
          charge = 20.0_real32
       case('Y')
          mass = 88.906_real32
          charge = 21.0_real32
       case('Zr')
          mass = 91.224_real32
          charge = 22.0_real32
       case('Nb')
          mass = 92.906_real32
          charge = 23.0_real32
       case('Mo')
          mass = 95.95_real32
          charge = 24.0_real32
       case('Tc')
          mass = 98.0_real32
          charge = 25.0_real32
       case('Ru')
          mass = 101.07_real32
          charge = 26.0_real32
       case('Rh')
          mass = 102.91_real32
          charge = 27.0_real32
       case('Pd')
          mass = 106.42_real32
          charge = 28.0_real32
       case('Ag')
          mass = 107.87_real32
          charge = 29.0_real32
       case('Cd')
          mass = 112.41_real32
          charge = 30.0_real32
       case('In')
          mass = 114.82_real32
          charge = 31.0_real32
       case('Sn')
          mass = 118.71_real32
          charge = 32.0_real32
       case('Sb')
          mass = 121.76_real32
          charge = 33.0_real32
       case('Te')
          mass = 127.6_real32
          charge = 34.0_real32
       case('I')
          mass = 126.9_real32
          charge = 35.0_real32
       case('Xe')
          mass = 131.29_real32
          charge = 36.0_real32
       case('Cs')
          mass = 132.91_real32
          charge = 37.0_real32
       case('Ba')
          mass = 137.33_real32
          charge = 38.0_real32
       case('La')
          mass = 138.91_real32
          charge = 39.0_real32
       case('Ce')
          mass = 140.12_real32
          charge = 40.0_real32
       case('Pr')
          mass = 140.91_real32
          charge = 41.0_real32
       case('Nd')
          mass = 144.24_real32
          charge = 42.0_real32
       case('Pm')
          mass = 145.0_real32
          charge = 43.0_real32
       case('Sm')
          mass = 150.36_real32
          charge = 44.0_real32
       case('Eu')
          mass = 152.0_real32
          charge = 45.0_real32
       case('Gd')
          mass = 157.25_real32
          charge = 46.0_real32
       case('Tb')
          mass = 158.93_real32
          charge = 47.0_real32
       case('Dy')
          mass = 162.5_real32
          charge = 48.0_real32
       case('Ho')
          mass = 164.93_real32
          charge = 49.0_real32
       case('Er')
          mass = 167.26_real32
          charge = 50.0_real32
       case('Tm')
          mass = 168.93_real32
          charge = 51.0_real32
       case('Yb')
          mass = 173.05_real32
          charge = 52.0_real32
       case('Lu')
          mass = 174.97_real32
          charge = 53.0_real32
       case('Hf')
          mass = 178.49_real32
          charge = 54.0_real32
       case('Ta')
          mass = 180.95_real32
          charge = 55.0_real32
       case('W')
          mass = 183.84_real32
          charge = 56.0_real32
       case('Re')
          mass = 186.21_real32
          charge = 57.0_real32
       case('Os')
          mass = 190.23_real32
          charge = 58.0_real32
       case('Ir')
          mass = 192.22_real32
          charge = 59.0_real32
       case('Pt')
          mass = 195.08_real32
          charge = 60.0_real32
       case('Au')
          mass = 196.97_real32
          charge = 61.0_real32
       case('Hg')
          mass = 200.59_real32
          charge = 62.0_real32
       case('Tl')
          mass = 204.38_real32
          charge = 63.0_real32
       case('Pb')
          mass = 207.2_real32
          charge = 64.0_real32
       case('Bi')
          mass = 208.98_real32
          charge = 65.0_real32
       case('Th')
          mass = 232.04_real32
          charge = 66.0_real32
       case('Pa')
          mass = 231.04_real32
          charge = 67.0_real32
       case('U')
          mass = 238.03_real32
          charge = 68.0_real32
       case('Np')
          mass = 237.0_real32
          charge = 69.0_real32
       case('Pu')
          mass = 244.0_real32
          charge = 70.0_real32
       case('Am')
          mass = 243.0_real32
          charge = 71.0_real32
       case('Cm')
          mass = 247.0_real32
          charge = 72.0_real32
       case('Bk')
          mass = 247.0_real32
          charge = 73.0_real32
       case('Cf')
          mass = 251.0_real32
          charge = 74.0_real32
       case('Es')
          mass = 252.0_real32
          charge = 75.0_real32
       case('Fm')
          mass = 257.0_real32
          charge = 76.0_real32
       case('Md')
          mass = 258.0_real32
          charge = 77.0_real32
       case('No')
          mass = 259.0_real32
          charge = 78.0_real32
       case('Lr')
          mass = 262.0_real32
          charge = 79.0_real32
       case('Rf')
          mass = 267.0_real32
          charge = 80.0_real32
       case('Db')
          mass = 270.0_real32
          charge = 81.0_real32
       case('Sg')
          mass = 271.0_real32
          charge = 82.0_real32
       case('Bh')
          mass = 270.0_real32
          charge = 83.0_real32
       case('Hs')
          mass = 277.0_real32
          charge = 84.0_real32
       case('Mt')
          mass = 276.0_real32
          charge = 85.0_real32
       case('Ds')
          mass = 281.0_real32
          charge = 86.0_real32
       case('Rg')
          mass = 280.0_real32
          charge = 87.0_real32
       case('Cn')
          mass = 285.0_real32
          charge = 88.0_real32
       case('Nh')
          mass = 284.0_real32
          charge = 89.0_real32
       case('Fl')
          mass = 289.0_real32
          charge = 90.0_real32
       case('Mc')
          mass = 288.0_real32
          charge = 91.0_real32
       case('Lv')
          mass = 293.0_real32
          charge = 92.0_real32
       case('Ts')
          mass = 294.0_real32
          charge = 93.0_real32
       case('Og')
          mass = 294.0_real32
          charge = 94.0_real32
       case default
          ! handle unknown element
          mass = 0.0_real32
          charge = 0.0_real32
       end select
       basis%spec(i)%mass = mass
       basis%spec(i)%charge = charge
    end do

  end subroutine get_elements_masses_and_charges
!!!#############################################################################

end module read_chemical_graphs          