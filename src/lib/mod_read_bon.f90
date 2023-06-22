module read_bon
  use constants, only: real12
  use misc, only: icount,set_str_output_order
  use misc_maths, only: gauss_array
  !use gvec
  implicit none

  private

  public :: bon_read

contains

!!!#############################################################################
!!! read bon data from RAFFLE filespace
!!!#############################################################################
  subroutine bon_read(dataset,dir)
    implicit none
    integer :: i,j,itmp1,itmp2,istruc,is1,is2,ipair
    integer :: Reason,unit1,unit2
    integer :: n_spec,n_spec_unique,n_pair_unique,n_bond,n_bin,ns2,n_struc,n_atom
    real(real12) :: rtmp1,tol
    real(real12) :: sigma, cutoff, bin_size
    logical :: filefound
    
    character(128) :: buffer,tstring1,tstring2
    character(1024) :: file,file_pattern,ls_file

    integer, allocatable, dimension(:) :: ivec1,n_atom_per_spec,order, bond_count, struc_list
    real(real12), allocatable, dimension(:) :: rvec1, bin_centre
    real(real12), allocatable, dimension(:) :: bond_dist
    character(5), allocatable, dimension(:) :: tlist
    
    real(real12), allocatable, dimension(:,:) :: g_vec, tarr1
    character(5), allocatable, dimension(:,:) :: spec_pair_list

    character(1024), intent(in) :: dir
    real(real12), allocatable, dimension(:,:), intent(inout) :: dataset
    

!!!-----------------------------------------------------------------------------
!!! read first pos file to get number of species
!!! WARNING: NEED TO FIND ANOTHER WAY OF IDENTIFYING ALL SPECIES
!!!-----------------------------------------------------------------------------
    unit1 = 10
    open(unit=unit1,file=trim(adjustl(dir))//"/pos/POSCAR_001/POSCAR",action='read')
    do i=1,5
       read(unit1,*)
    end do
    read(unit1,'(A)',iostat=Reason) buffer
    n_spec = icount(buffer)
    allocate(n_atom_per_spec(n_spec))
    read(unit1,*) n_atom_per_spec
    close(unit1)
    
    allocate(tlist(n_spec))
    read(buffer,*) (tlist(i),i=1,n_spec)

    write(0,*) tlist
    write(0,*) n_atom_per_spec
    
    allocate(order(n_spec))
    order=set_str_output_order(tlist,.true.)
    n_spec_unique = size(tlist)

    allocate(ivec1(n_spec_unique))
    ivec1 = 0

    do i=1,n_spec
       ivec1(order(i)) = ivec1(order(i)) + n_atom_per_spec(i)
    end do
    n_atom_per_spec = ivec1
    

!!!-----------------------------------------------------------------------------
!!! determine the unqiue pairs of species
!!!-----------------------------------------------------------------------------
    itmp1 = 0
    n_pair_unique = nint(n_spec_unique * (n_spec_unique - (n_spec_unique-1.E0)/2.E0))
    allocate(spec_pair_list(n_pair_unique,2))
    do i=1,n_spec_unique
       do j=i,n_spec_unique
          itmp1 = itmp1 + 1
          spec_pair_list(itmp1,1) = tlist(i)
          spec_pair_list(itmp1,2) = tlist(j)
          write(0,*) spec_pair_list(itmp1,:)
       end do
    end do

    write(0,*) "DONE"

!! for each pair, there will be two sets of angles
!! A-B-A, and B-A-B !! OR WILL THERE JUST BE ONE? IS A-B-A THE SAME AS B-A-B?
!! I THINK SO, SO JUST USE THE FIRST ONE
!! A = spec_pair_list(:,1)
!! B = spec_pair_list(:,2)
!! GET THIS DATA FROM BAD/ directory (bond angle distribution)
!! READ ALL SETS, C_C_C, C_C_Mg, C_Mg_Mg
!! MAYBE JUST LOOK FOR X_Y_X AND Y_X_Y

    
!!!-----------------------------------------------------------------------------
!!! set up gaussians for g_vector bond lengths
!!!-----------------------------------------------------------------------------
    tol = 1.E-3
    sigma = 0.1E0
    bin_size = 0.25E0
    cutoff = 8.E0
    n_bin = nint(cutoff/bin_size)
    allocate(bin_centre(n_bin))
    do i=1,n_bin
       bin_centre(i) = i*bin_size
    end do
    allocate(rvec1(n_bin))
    allocate(g_vec(n_pair_unique,n_bin))
    g_vec = 0.E0

    
!!!-----------------------------------------------------------------------------
!!! get the energies for each structure
!!!-----------------------------------------------------------------------------
    n_struc = 0
    open(unit1,file=trim(adjustl(dir))//"/tmp_energies.txt",action='read')
    n_struc_count_loop: do
       read(unit1,'(A)',iostat=Reason) buffer
       if(Reason.ne.0) exit
       if(icount(buffer).eq.1) cycle
       read(buffer,*) tstring1, tstring2
       if(verify('.1',tstring1).ne.0.or.trim(tstring2).eq."") cycle n_struc_count_loop
       n_struc = n_struc + 1
    end do n_struc_count_loop
    allocate(struc_list(n_struc), dataset(n_struc,n_bin*n_pair_unique+1))
    dataset=0.E0
    struc_list=0
    rewind(unit1)
    
    
!!!-----------------------------------------------------------------------------
!!! determine the number of usable structures (that have all data present)
!!!-----------------------------------------------------------------------------
    n_struc = 0
    energy_read_loop: do
       read(unit1,'(A)',iostat=Reason) buffer
       if(Reason.ne.0) exit
       if(icount(buffer).eq.1) cycle
       read(buffer,*) tstring1, tstring2
       if(index(tstring1,'.1').eq.0.or.trim(tstring2).eq."") cycle energy_read_loop
       n_struc = n_struc + 1
       read(tstring1(:scan(tstring1,".")-1),*) struc_list(n_struc)
       read(tstring2,*) dataset(n_struc,n_bin*n_pair_unique+1)     
    end do energy_read_loop
    close(unit1)
    write(0,*) "TOTAL NUMBER OF USABLE STRUCTURES:", n_struc

    

!!!-----------------------------------------------------------------------------
!!! read the structure files and determine g_vectors
!!!-----------------------------------------------------------------------------
    itmp2 = 0
    unit2=11
    ls_file="fileContents.txt"
    struc_loop: do istruc=1,n_struc
       itmp2 = itmp2 + 1
       dataset(itmp2,:n_bin*n_pair_unique) = 0.E0
       g_vec = 0.E0
       write(0,*) "structure",itmp2, struc_list(istruc)
       spec_loop: do is1=1,n_spec_unique
          !ns2=findloc(tlist,spec_pair_list(is1,1),dim=1)
          write(file_pattern,'(A,"/bon/BON_",I0.3,"/",A,"_*")') &
               trim(adjustl(dir)),struc_list(istruc),trim(adjustl(tlist(is1)))
          call execute_command_line('ls '//trim(adjustl(file_pattern))//' > '//trim(adjustl(ls_file)),exitstat=Reason)
          !write(0,*) "spec",is1,trim(file_pattern)
          if(Reason.ne.0) cycle spec_loop
          open(unit=unit2,file=ls_file,action="read")
          file_read_loop: do
             !write(file,'(A,"/bon/BON_",I0.3,"/",A,"_",I0.3,A,"_",A,"_bond_distribution")') &
             !     trim(adjustl(dir)),struc_list(istruc),trim(adjustl(spec_pair_list(is1,1))),is2,&
             !     trim(adjustl(spec_pair_list(is1,1))),trim(adjustl(spec_pair_list(is1,2)))
             read(unit2,'(A)',iostat=Reason) file
             !write(0,*) trim(file)
             if(Reason.ne.0.or.trim(file).eq."")then !end of file
                !write(0,*) "Exit"
                exit file_read_loop
             end if
             !write(0,*) "continue"

             if(verify('gaussian',file).eq.0.or.verify('distribution',file).eq.0) cycle
             !! does this need to do anything else if the species isn't there at all?

             inquire(file=trim(file),exist=filefound)
             if(.not.filefound)then
                itmp2 = itmp2 - 1
                close(unit2)
                cycle struc_loop
             end if
             open(unit1,file=trim(adjustl(file)),action='read')

             n_atom = 0
             n_bond = 0
             n_bond_loop: do
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0.or.buffer.eq."") exit n_bond_loop
                if(scan(trim(adjustl(buffer)),'.').eq.0)then
                   n_atom = n_atom + 1
                   cycle n_bond_loop
                end if
                n_bond = n_bond + 1
             end do n_bond_loop
             if(n_bond.eq.0) cycle
             allocate(bond_dist(n_bond))
             allocate(bond_count(n_bond))
             bond_dist = -huge(1.E0)
             bond_count = 0
             itmp1 = 0
             rewind(unit1)

             bon_read_loop: do i=1,n_bond+n_atom,1
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0) exit bon_read_loop

                if(scan(trim(adjustl(buffer)),'.').eq.0)then
                   !! need to determine what species we are connecting to?
                   !! so, in is1 loop, only go over each nspec, not each n_pair_unique ...
                   !! ... that way the file defines the pair we are on and we use that later?
                   !! do we even use the pair value at all? Yes, this is all what is1 is about.
                   !! so we use our new is1 (nspec)
                   do j=1,n_pair_unique
                      if(spec_pair_list(j,1).eq.tlist(is1).and.&
                           spec_pair_list(j,2).eq.trim(adjustl(buffer)))then
                         if(i.ne.1)then
                            !write(0,*) "here",i,itmp1
                            !write(0,*) bond_dist(:itmp1)
                            !write(0,*) bond_count(:itmp1)
                            g_vec(j,:) = g_vec(j,:) + gauss_array(bin_centre,bond_dist(:itmp1),sigma,&
                                 multiplier=bond_count(:itmp1))
                         end if
                         bond_dist = -huge(1.E0)
                         bond_count = 0
                         itmp1 = 0
                         ipair = j
                         !write(0,*) "here",ipair,tlist(is1),trim(buffer)
                         cycle bon_read_loop
                      end if
                   end do
                   cycle bon_read_loop
                end if
                read(buffer,*) rtmp1
                
                if(rtmp1.gt.cutoff+2.50*sigma) cycle bon_read_loop
                if(any(abs(rtmp1-bond_dist(:itmp1)).lt.tol))then
                   where(abs(rtmp1-bond_dist(:itmp1)).lt.tol)
                      bond_count(:itmp1) = bond_count(:itmp1) + 1
                   end where
                   cycle bon_read_loop
                end if
                itmp1 = itmp1 + 1
                bond_dist(itmp1) = rtmp1
                bond_count(itmp1) = 1

             end do bon_read_loop
             n_bond = itmp1
             close(unit1)

             !! do this for the final pair that doesn't get picked
             g_vec(ipair,:) = g_vec(ipair,:) + gauss_array(bin_centre,bond_dist(:itmp1),sigma,multiplier=bond_count(:itmp1))

             !MAKE GVEC FROM THIS BOND_DIST AND BOND_COUNT CAN JUST MULTIPLY EACH TIME
             !g_vec(ipair,:) = g_vec(ipair,:) + gauss_array(bin_centre,bond_dist,sigma,multiplier=bond_count)
             !gvec_gen_loop: do i=1,n_bond
             !   rvec1 = 0.E0
             !   call gauss(rvec1, bond_dist(i), n_bin, bin_centre, eta, cutoff)
             !   rvec1 = rvec1 * bond_count(i)
             !   g_vec(is1,:) = g_vec(is1,:) + rvec1
             !end do gvec_gen_loop
             !write(0,*) g_vec(is1,:)

             deallocate(bond_dist,bond_count)

          end do file_read_loop
          close(unit2)
       end do spec_loop
       do i=1,n_pair_unique
          dataset(itmp2,n_bin*(i-1)+1:n_bin*i) = g_vec(i,:)
       end do
       !write(0,*) size(g_vec,dim=2), n_bin*(is1-1)+1,n_bin*is1
       !write(0,*) struc_list(istruc)
       !write(0,*) dataset(itmp2,:)
       !write(0,*)
    end do struc_loop
    write(0,*) "DONE"
    

!!!-----------------------------------------------------------------------------
!!! resize dataset
!!!-----------------------------------------------------------------------------
    allocate(tarr1(itmp2,size(dataset,dim=2)))
    tarr1 = dataset(:itmp2,:)
    call move_alloc(tarr1,dataset)


  end subroutine bon_read
!!!#############################################################################

end module read_bon
