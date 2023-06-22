module read_gvec
  use constants, only: real12,pi
  use misc, only: icount
  use misc_maths, only: gauss_array
  implicit none

  private

  public :: gvec_read


!!!updated  2023/03/15


contains

!!!#############################################################################
!!! read bon and bad data from RAFFLE filespace
!!! bon = bond
!!! bad = bond angle distribution
!!! dscr = discrete/discretised
!!!#############################################################################
  subroutine gvec_read(dataset,dir,&
       bond_cutoff,bond_sigma,bond_bin_size,bond_tol,&
       angle_sigma,angle_bin_size,angle_tol)
    implicit none
    integer :: i,j,k,itmp1,itmp2,istruc,is1,is2
    integer :: Reason,unit1,unit2,n_col
    integer :: fs_1,fs_2
    integer :: ipair,itrip
    integer :: n_spec,n_spec_unique
    integer :: n_pair_unique,n_trip_unique
    integer :: n_bad,n_bon,n_struc,n_atom
    integer :: n_bin_bad,n_bin_bon
    real(real12) :: rtmp1
    real(real12) :: tol_bad,tol_bon
    real(real12) :: cutoff
    real(real12) :: sigma_bad,sigma_bon,bin_size_bad,bin_size_bon
    logical :: filefound
    
    character(5) :: spec1,spec3
    character(128) :: buffer,tstring1,tstring2,fmt
    character(1024) :: file,file_pattern,ls_file

    integer, allocatable, dimension(:) :: bad_count,bon_count
    integer, allocatable, dimension(:) :: struc_list
    real(real12), allocatable, dimension(:) :: bin_centre_bad,bin_centre_bon
    real(real12), allocatable, dimension(:) :: bad_dscr,bon_dscr 
    character(5), allocatable, dimension(:) :: spec_list
    
    real(real12), allocatable, dimension(:,:) :: g_vec_bad,g_vec_bon
    real(real12), allocatable, dimension(:,:) :: tarr1
    character(5), allocatable, dimension(:,:) :: spec_pair_list,spec_trip_list

    real(real12), optional, intent(in) :: bond_tol,angle_tol
    real(real12), optional, intent(in) :: bond_cutoff,bond_sigma,bond_bin_size
    real(real12), optional, intent(in) :: angle_sigma,angle_bin_size

    character(1024), intent(in) :: dir
    real(real12), allocatable, dimension(:,:), intent(inout) :: dataset
    

!!!-----------------------------------------------------------------------------
!!! ls Devolved directory to identify list of unique species
!!!-----------------------------------------------------------------------------
    ls_file="fileContents.txt"
    call execute_command_line(&
         'ls '//trim(adjustl(dir))//'/Devolved/*_evolved_bondlength_gauss'//&
         '|awk ''{gsub("'//trim(adjustl(dir))//'/Devolved/","",$0);print$0}'''//&
         '|awk ''/_.*_.*_.*_/{gsub("_.*_evolved_bondlength_gauss","");print$0}'''//&
         '|sort -u|tr ''\n'' '' '' >'//trim(adjustl(ls_file)),exitstat=Reason)
    if(Reason.ne.0)then
       stop "Error with Devolved directory"
    end if

    unit1 = 10
    open(unit=unit1,file=trim(adjustl(ls_file)),action='read')
    read(unit1,'(A)',iostat=Reason) buffer
    close(unit1)
    n_spec_unique = icount(buffer)
    allocate(spec_list(n_spec_unique))
    read(buffer,*) (spec_list(i),i=1,n_spec_unique)
    
    write(fmt,'("(1X,""Species list:"",",I0,"(1X,A))")') n_spec_unique
    write(6,trim(fmt)) (trim(spec_list(i)),i=1,n_spec_unique)
    write(6,*)


!!!-----------------------------------------------------------------------------
!!! determine the unique pairs of species
!!!-----------------------------------------------------------------------------
    itmp1 = 0
    n_pair_unique = nint(&
         n_spec_unique * (n_spec_unique - (n_spec_unique-1.E0)/2.E0))
    allocate(spec_pair_list(n_pair_unique,2))
    write(6,'(1X,"ipair spec1 spec2")')
    do i=1,n_spec_unique
       do j=i,n_spec_unique
          itmp1 = itmp1 + 1
          spec_pair_list(itmp1,1) = spec_list(i)
          spec_pair_list(itmp1,2) = spec_list(j)
          write(6,'(1X,I3,4X,2(1X,A5))') itmp1,spec_pair_list(itmp1,:)
       end do
    end do
    write(6,'(1X,"Number of unique species pairs: ",I0)') n_pair_unique
    write(6,*)


!!!-----------------------------------------------------------------------------
!!! determine the unique triplets of species
!!!-----------------------------------------------------------------------------
    itmp1 = 0
    n_trip_unique = nint(&
         n_spec_unique**2.E0 * &
         (n_spec_unique - (n_spec_unique-1.E0)/2.E0) )
    allocate(spec_trip_list(n_trip_unique,3))
    write(6,'(1X,"itriplet spec1 spec2 spec3")')
    do i=1,n_spec_unique
       do j=1,n_spec_unique
          do k=j,n_spec_unique
             itmp1 = itmp1 + 1
             spec_trip_list(itmp1,2) = spec_list(i)
             spec_trip_list(itmp1,1) = spec_list(j)
             spec_trip_list(itmp1,3) = spec_list(k)
             write(6,'(3X,I3,5X,3(1X,A5))') itmp1,spec_trip_list(itmp1,:)
          end do
       end do
    end do
    write(6,'(1X,"Number of unique species triplets: ",I0)') n_trip_unique
    write(6,*)

!!!-----------------------------------------------------------------------------
!!! set up gaussians for g_vector bond lengths
!!!-----------------------------------------------------------------------------
    if(present(bond_tol))then
       tol_bon = bond_tol
    else
       tol_bon = 1.E-3
    end if
    if(present(bond_cutoff))then
       cutoff = bond_cutoff
    else
       cutoff = 8.E0
    end if
    if(present(bond_sigma))then
       sigma_bon = bond_sigma
    else
       sigma_bon = 0.1E0
    end if
    if(present(bond_bin_size))then
       bin_size_bon = bond_bin_size
    else
       bin_size_bon = 0.25E0
    end if
    n_bin_bon = nint(cutoff/bin_size_bon)
    allocate(bin_centre_bon(n_bin_bon))
    do i=1,n_bin_bon,1
       bin_centre_bon(i) = i*bin_size_bon
    end do
    allocate(g_vec_bon(n_pair_unique,n_bin_bon))
    g_vec_bon = 0.E0

    
!!!-----------------------------------------------------------------------------
!!! set up gaussians for g_vector bond angles
!!!-----------------------------------------------------------------------------
    if(present(angle_tol))then
       tol_bad = angle_tol
    else
       tol_bad = 1.E-4
    end if
    if(present(angle_sigma))then
       sigma_bad = angle_sigma
    else
       sigma_bad = 0.01E0
    end if
    if(present(angle_bin_size))then
       bin_size_bad = angle_bin_size
    else
       bin_size_bad = pi/18.E0
    end if
    n_bin_bad = nint(pi/bin_size_bad)
    allocate(bin_centre_bad(n_bin_bad))
    do i=1,n_bin_bad,1
       bin_centre_bad(i) = (i-0.5E0)*bin_size_bad
    end do
    allocate(g_vec_bad(n_trip_unique,n_bin_bad))
    g_vec_bad = 0.E0

    
!!!-----------------------------------------------------------------------------
!!! get the energies for each structure
!!!-----------------------------------------------------------------------------
    n_struc = 0
    n_col = n_bin_bon*n_pair_unique + n_bin_bad*n_trip_unique
    open(unit1,file=trim(adjustl(dir))//"/tmp_energies.txt",action='read')
    n_struc_count_loop: do
       read(unit1,'(A)',iostat=Reason) buffer
       if(Reason.ne.0) exit
       if(icount(buffer).eq.1) cycle
       read(buffer,*) tstring1, tstring2
       if(verify('.1',tstring1).ne.0.or.trim(tstring2).eq."") &
            cycle n_struc_count_loop
       n_struc = n_struc + 1
    end do n_struc_count_loop
    allocate(struc_list(n_struc), dataset(n_struc,n_col+1))
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
       if(index(tstring1,'.1').eq.0.or.trim(tstring2).eq."") &
            cycle energy_read_loop
       n_struc = n_struc + 1
       read(tstring1(:scan(tstring1,".")-1),*) struc_list(n_struc)
       read(tstring2,*) dataset(n_struc,n_col+1)     
    end do energy_read_loop
    close(unit1)
    write(6,'(1X,"Total number of valid structures: ",I0)') n_struc
    write(6,*)
    

!!!-----------------------------------------------------------------------------
!!! read the structure files and determine g_vectors
!!!-----------------------------------------------------------------------------
    itmp2 = 0
    unit2=11
    struc_loop: do istruc=1,n_struc
       itmp2 = itmp2 + 1
       dataset(itmp2,:n_col) = 0.E0
       g_vec_bon = 0.E0
       g_vec_bad = 0.E0
       write(6,*) "structure",itmp2, struc_list(istruc)
       spec_loop: do is1=1,n_spec_unique
          write(file_pattern,'(A,"/bon/BON_",I0.3,"/",A,"_*")') &
               trim(adjustl(dir)),struc_list(istruc),&
               trim(adjustl(spec_list(is1)))
          call execute_command_line(&
               'ls '//trim(adjustl(file_pattern))//' > '//&
               trim(adjustl(ls_file)),exitstat=Reason)
          if(Reason.ne.0) cycle spec_loop
          open(unit=unit2,file=ls_file,action="read")
          
          !!--------------------------------------------------------------------
          !! read through each of the bon files for structure 'is'
          !!--------------------------------------------------------------------
          file_read_loop1: do
             read(unit2,'(A)',iostat=Reason) file
             if(Reason.ne.0.or.trim(file).eq."")then !end of file
                exit file_read_loop1
             end if

             !! NOTE: does this need to do anything else if the species isn't there at all?
             if(verify('gaussian',file).eq.0.or.&
                  verify('distribution',file).eq.0) cycle

             !! check if file exists, else skip structure
             inquire(file=trim(file),exist=filefound)
             if(.not.filefound)then
                itmp2 = itmp2 - 1
                close(unit2)
                cycle struc_loop
             end if
             open(unit1,file=trim(adjustl(file)),action='read')

             !! count number of pairs in file and number of bonds
             n_atom = 0
             n_bon = 0
             n_bon_loop1: do
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0.or.buffer.eq."") exit n_bon_loop1
                if(scan(trim(adjustl(buffer)),'.').eq.0)then
                   n_atom = n_atom + 1
                   cycle n_bon_loop1
                end if
                n_bon = n_bon + 1
             end do n_bon_loop1
             if(n_bon.eq.0) cycle file_read_loop1
             allocate(bon_dscr(n_bon))
             allocate(bon_count(n_bon))
             bon_dscr = -huge(1.E0)
             bon_count = 0
             itmp1 = 0
             rewind(unit1)

             !! read through each line in bon file 'file' and store the bonds
             bon_read_loop: do i=1,n_bon+n_atom,1
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0) exit bon_read_loop

                !! check if there is no decimal point on the line
                !! ... if there is none, then it's a species title line
                !! ... so start a bond list for new species pair
                if(scan(trim(adjustl(buffer)),'.').eq.0)then
                   !! need to determine what species we are connecting to?
                   !! so, in is1 loop, only go over each nspec, not each n_pair_unique ...
                   !! ... that way the file defines the pair we are on and we use that later?
                   !! do we even use the pair value at all? Yes, this is all what is1 is about.
                   !! so we use our new is1 (nspec)
                   do j=1,n_pair_unique
                      if(spec_pair_list(j,1).eq.spec_list(is1).and.&
                           spec_pair_list(j,2).eq.trim(adjustl(buffer)))then
                         if(i.ne.1)then
                            g_vec_bon(j,:) = g_vec_bon(j,:) + &
                                 gauss_array(bin_centre_bon,bon_dscr(:itmp1),&
                                 sigma_bon,multiplier=bon_count(:itmp1))
                         end if
                         bon_dscr = -huge(1.E0)
                         bon_count = 0
                         itmp1 = 0
                         ipair = j
                         cycle bon_read_loop
                      end if
                   end do
                   cycle bon_read_loop
                end if
                read(buffer,*) rtmp1
                
                !! if bond is over cutoff, skip the bond
                if(rtmp1.gt.cutoff+2.50*sigma_bon) cycle bon_read_loop
                !! if bond length is within tolerance of other stored, ...
                !! ... increase duplicate count and cycle
                if(any(abs(rtmp1-bon_dscr(:itmp1)).lt.tol_bon))then
                   where(abs(rtmp1-bon_dscr(:itmp1)).lt.tol_bon)
                      bon_count(:itmp1) = bon_count(:itmp1) + 1
                   end where
                   cycle bon_read_loop
                end if
                itmp1 = itmp1 + 1
                bon_dscr(itmp1) = rtmp1
                bon_count(itmp1) = 1

             end do bon_read_loop
             n_bon = itmp1
             close(unit1)

             !! store g_vec for final pair, as it is not caught by the earlier check
             !! ... the earlier check is for 'species title' lines in the file
             !! ... as the final line has no title line after, it is skipped by ...
             !! ... the earlier check
             g_vec_bon(ipair,:) = g_vec_bon(ipair,:) + gauss_array(&
                  bin_centre_bon,bon_dscr(:itmp1),&
                  sigma_bon,multiplier=bon_count(:itmp1))

             deallocate(bon_dscr,bon_count)

          end do file_read_loop1
          close(unit2)

          
          !!--------------------------------------------------------------------
          !! read through each of the bad files for structure 'is'
          !!--------------------------------------------------------------------
          write(file_pattern,'(A,"/bad/BAD_",I0.3,"/*_",A,"_*")') &
               trim(adjustl(dir)),struc_list(istruc),&
               trim(adjustl(spec_list(is1)))
          call execute_command_line('ls '//trim(adjustl(file_pattern))//' > '//&
               trim(adjustl(ls_file)),exitstat=Reason)
          !! if no files match 'file_pattern' skip structure
          if(Reason.ne.0) cycle spec_loop
          open(unit=unit2,file=ls_file,action="read")

          !! read through ls output and select each bad file for reading
          file_read_loop2: do
             read(unit2,'(A)',iostat=Reason) file
             if(is_iostat_end(Reason).or.trim(file).eq."")then !end of file
                exit file_read_loop2
             end if
          
             !! check if file exists, else, skip
             inquire(file=trim(file),exist=filefound)
             if(.not.filefound)then
                itmp2 = itmp2 - 1
                close(unit2)
                cycle struc_loop
             end if
             open(unit1,file=trim(adjustl(file)),action='read')
             
             !! determine number of angles listed in file
             n_bad = 0
             n_bad_loop2: do
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0.or.buffer.eq."") exit n_bad_loop2
                if(scan(trim(adjustl(buffer)),'.').ne.0)then
                   n_bad = n_bad + 1
                end if
             end do n_bad_loop2
             if(n_bad.eq.0) cycle file_read_loop2
             allocate(bad_dscr(n_bad))
             allocate(bad_count(n_bad))
             bad_dscr = -huge(1.E0)
             bad_count = 0
             itmp1 = 0
             rewind(unit1)

             !! deteremine the angle triplet identifier, itrip
             itrip = 1
             buffer = adjustl(buffer)
             fs_1 = scan(buffer,"_")
             fs_2 = scan(buffer,"_",.true.)
             spec1 = trim(buffer(:fs_1-1))
             spec3 = trim(buffer(fs_2+1:))
             get_triplet_loop: do j=1,n_trip_unique
                if(&
                     spec_trip_list(j,2).eq.spec_list(is1).and.&
                     spec_trip_list(j,1).eq.spec1.and.&
                     spec_trip_list(j,3).eq.spec3)then
                   itrip = j
                   exit get_triplet_loop 
                end if
             end do get_triplet_loop

             !! read through bad file and set up list of angles and count
             bad_read_loop: do i=1,n_bad,1
                read(unit1,'(A)',iostat=Reason) buffer
                if(Reason.ne.0) exit bad_read_loop
                read(buffer,*) rtmp1

                !! if bond angle is within tolerance of other stored, increase duplicate count and cycle
                if(any(abs(rtmp1-bad_dscr(:itmp1)).lt.tol_bad))then
                   where(abs(rtmp1-bad_dscr(:itmp1)).lt.tol_bad)
                      bad_count(:itmp1) = bad_count(:itmp1) + 1
                   end where
                   cycle bad_read_loop
                end if
                itmp1 = itmp1 + 1
                bad_dscr(itmp1) = rtmp1
                bad_count(itmp1) = 1

             end do bad_read_loop
             n_bad = itmp1
             close(unit1)
             
             !! for each bond triplet, add discretise data and add to g_vec
             g_vec_bad(itrip,:) = g_vec_bad(itrip,:) + &
                  gauss_array_angle(&
                  bin_centre_bad,bad_dscr(:itmp1),&
                  sigma_bad,multiplier=bad_count(:itmp1))


             deallocate(bad_dscr,bad_count)
          end do file_read_loop2
          close(unit2)

       end do spec_loop


       !!-----------------------------------------------------------------------
       !! store successful structure bon data in dataset
       !!-----------------------------------------------------------------------
       do i=1,n_pair_unique,1
          dataset(itmp2,n_bin_bon*(i-1)+1:n_bin_bon*i) = g_vec_bon(i,:)
       end do


       !!-----------------------------------------------------------------------
       !! store successful structure bad data in dataset
       !!-----------------------------------------------------------------------
       itmp1 = n_bin_bon*n_pair_unique
       do i=1,n_trip_unique,1
          dataset(itmp2,itmp1+n_bin_bad*(i-1)+1:itmp1+n_bin_bad*i) = g_vec_bad(i,:)
       end do


    end do struc_loop


!!!-----------------------------------------------------------------------------
!!! print end of structure reading
!!!-----------------------------------------------------------------------------
    write(6,*) "All structures read in"
    write(6,*)
    

!!!-----------------------------------------------------------------------------
!!! resize dataset
!!!-----------------------------------------------------------------------------
    allocate(tarr1(itmp2,size(dataset,dim=2)))
    tarr1 = dataset(:itmp2,:)
    call move_alloc(tarr1,dataset)


  end subroutine gvec_read
!!!#############################################################################


!!!#############################################################################
!!! apply gaussians to a set of angles in an array
!!!#############################################################################
  function gauss_array_angle(angle,in_array,sigma,tol,norm,mask,multiplier) &
       result(gauss_func)
    implicit none
    integer :: i,n,init_step,n_bin
    real(real12) :: x,udef_tol,mult,multi,step_size
    logical :: lexit

    real(real12), intent(in) :: sigma
    real(real12), dimension(:), intent(in) :: in_array,angle

    real(real12), optional, intent(in) :: tol
    logical, optional, intent(in) :: norm
    integer, dimension(:), optional, intent(in) :: multiplier
    logical, dimension(size(angle)), optional, intent(in) :: mask

    real(real12), dimension(size(angle)) :: gauss_func

    udef_tol=16.0
    if(present(tol)) udef_tol=tol
    mult=(1.0/(sqrt(pi*2.0)*sigma))
    if(present(norm))then
       if(.not.norm) mult=1.0
    end if
    multi = mult
    
    n_bin = size(angle)
    step_size = angle(2) - angle(1)

    gauss_func=0.0
    do n=1,size(in_array)
       if(present(mask))then
          if(.not.mask(n)) cycle
       end if
       !! determine closest bin to data point
       init_step=minloc(abs( angle(:) - in_array(n) ),dim=1)
       if(present(multiplier))then
          multi = multiplier(n) * mult
       end if

       lexit = .false.
       !! iterate down the right-side of the gaussian until negligible
       forward1: do i=init_step,n_bin,1
          x=0.5*(( angle(i) - in_array(n) )/sigma)**2.0
          if(x.gt.udef_tol)then
             lexit = .true.
             exit forward1
          end if
          gauss_func(i) = gauss_func(i) + exp(-x) * multi
       end do forward1
       !! if angle close to pi, iterate from pi to 2pi (i.e. 0)
       !! ... this must be done as angles are mirrored about pi
       if(.not.lexit)then
          forward2: do i=n_bin,1,-1
             x=0.5*(( ( angle(n_bin) + step_size * (n_bin +1 - i) ) - &
                  in_array(n) )/sigma)**2.0
             if(x.gt.udef_tol) exit forward2
             gauss_func(i) = gauss_func(i) + exp(-x) * multi
          end do forward2
       end if

       lexit = .false.
       !! iterate down the left-side of the gaussian until negligible
       backward1: do i=init_step-1,1,-1
          x=0.5*(( angle(i) - in_array(n) )/sigma)**2.0
          if(x.gt.udef_tol)then
             lexit = .true.
             exit backward1
          end if
          gauss_func(i) = gauss_func(i) + exp(-x) * multi
       end do backward1
       !! if angle close to 0, iterate from 0 to -pi (i.e. pi)
       !! ... this must be done as angles are mirrored about 0
       if(.not.lexit)then
          backward2: do i=1,n_bin,1
             x=0.5*(( ( angle(1) - step_size * i ) - &
                  in_array(n) )/sigma)**2.0
             if(x.gt.udef_tol) exit backward2
             gauss_func(i) = gauss_func(i) + exp(-x) * multi
          end do backward2
       end if
    end do


    
  end function gauss_array_angle
!!!#############################################################################



end module read_gvec
