!!!#############################################################################
!!! Code written by Ned Thaddeus Taylor and Francis Huw Davies
!!! Code part of the ARTEMIS group (Hepplestone research group).
!!! Think Hepplestone, think HRG.
!!!#############################################################################
!!! module contains various miscellaneous functions and subroutines.
!!! module includes the following functions and subroutines:
!!! increment_list   (iteratively increment a list)
!!! alloc            (allocate using an 1D array as the shape) 
!!! find_loc         (version of findloc for pre fortran2008)
!!! closest_below    (returns closest element below input number)
!!! closest_above    (returns closest element above input number)
!!! sort1D           (sort 1st col of array by size. Opt:sort 2nd array wrt 1st)
!!! sort2D           (sort 1st two columns of an array by size)
!!! sort_str         (sort a list of strings)
!!! set              (return the sorted set of unique elements)
!!! sort_col         (sort array with respect to col column)
!!! swap             (swap two variables around)
!!! shuffle          (randomly shuffle a 2D array along one dimension)
!!!##################
!!! Icount           (counts words on line)
!!! readcl           (read string and separate into a char array using user fs)
!!! grep             (finds 1st line containing the pattern)
!!! count_occ        (count number of occurances of substring in string)
!!! flagmaker        (read flag inputs supplied and stores variable if present)
!!! loadbar          (writes out a loading bar to the terminal)
!!! jump             (moves file to specified line number)
!!! file_check       (checks whether file exists and prompts user otherwise)
!!! to_upper         (converts all characters in string to upper case)
!!! to_lower         (converts all characters in string to lower case)
!!!#############################################################################
module misc
  use constants, only: real12
  implicit none


  interface alloc
     procedure ralloc2D,ralloc3D
  end interface alloc

  interface sort1D
     procedure isort1D,rsort1D
  end interface sort1D

  interface set
     procedure iset,rset, cset
  end interface set

  interface swap
     procedure iswap, rswap, rswap_vec, cswap
  end interface swap


!!!updated 2021/12/08


contains
!!!#####################################################
!!! increment a list
!!!#####################################################
 recursive subroutine increment_list(list,max_list,dim,fixed_dim)
   implicit none
   integer, intent(in) :: dim,fixed_dim
   integer, dimension(:), intent(in) :: max_list
   integer, dimension(:), intent(inout) :: list

   if(dim.eq.fixed_dim)then
      call increment_list(list,max_list,dim-1,fixed_dim)
      return
   elseif(dim.gt.size(list))then
      call increment_list(list,max_list,size(list),fixed_dim)
   elseif(dim.le.0)then
      list = 0
      return
   end if

   list(dim) = list(dim) + 1
   
   if(list(dim).gt.max_list(dim))then
      list(dim) = 1
      call increment_list(list,max_list,dim-1,fixed_dim)
   end if

 end subroutine increment_list
!!!#####################################################


!!!#####################################################
!!! find location of true in vector
!!!#####################################################
  subroutine ralloc2D(arr,list)
    implicit none
    integer, dimension(2), intent(in) :: list
    real(real12), allocatable, dimension(:,:) :: arr
    
    allocate(arr(list(1),list(2)))

  end subroutine ralloc2D
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine ralloc3D(arr,list)
    implicit none
    integer, dimension(3), intent(in) :: list
    real(real12), allocatable, dimension(:,:,:) :: arr

    allocate(arr(list(1),list(2),list(3)))

  end subroutine ralloc3D
!!!#####################################################


!!!#####################################################
!!! find location of true in vector
!!!#####################################################
  function find_loc(array,dim,mask,back) result(idx)
    implicit none
    integer :: i,idx
    integer :: istep,istart,iend,idim
    integer, optional, intent(in) :: dim
    logical, dimension(:) :: array
    logical, optional, intent(in) :: back
    logical, dimension(:), optional, intent(in) :: mask

    idim=1
    if(present(dim)) idim=dim
    istep=1;istart=1;iend=size(array,dim=idim)
    if(present(back))then
       if(back)then
          istep=-1;istart=size(array,dim=idim);iend=1
       end if
    end if

    idx = 0
    do i=istart,iend,istep
       if(present(mask))then
          if(.not.mask(i)) cycle
       end if
       idx = idx + 1
       if(array(i)) exit
    end do

    return
  end function find_loc
!!!#####################################################


!!!#####################################################
!!! function to find closest -ve element in array
!!!#####################################################
  function closest_below(vec,val,optmask) result(int)
    implicit none
    integer :: i,int
    real(real12) :: val,best,dtmp1
    real(real12), dimension(:) :: vec
    logical, dimension(:), optional :: optmask

    int=0
    best=-huge(0._real12)
    do i=1,size(vec)
       dtmp1=vec(i)-val
       if(present(optmask))then
          if(.not.optmask(i)) cycle
       end if
       if(dtmp1.gt.best.and.dtmp1.lt.-1.D-8)then
          best=dtmp1
          int=i
       end if
    end do

    return
  end function closest_below
!!!#####################################################


!!!#####################################################
!!! function to find closest +ve element in array
!!!#####################################################
  function closest_above(vec,val,optmask) result(int)
    implicit none
    integer :: i,int
    real(real12) :: val,best,dtmp1
    real(real12), dimension(:) :: vec
    logical, dimension(:), optional :: optmask

    int=0
    best=huge(0._real12)
    do i=1,size(vec)
       dtmp1=vec(i)-val
       if(present(optmask))then
          if(.not.optmask(i)) cycle
       end if
       if(dtmp1.lt.best.and.dtmp1.gt.1.D-8)then
          best=dtmp1
          int=i
       end if
    end do

    return
  end function closest_above
!!!#####################################################


!!!#####################################################
!!! sorts a character list
!!!#####################################################
  subroutine sort_str(list,lcase)
    implicit none
    integer :: i,loc
    integer :: charlen
    logical :: ludef_case
    character(*), dimension(:), intent(inout) :: list
    character(:), allocatable, dimension(:) :: tlist
    logical, optional, intent(in) :: lcase !default is false

    charlen = len(list(1))
    if(present(lcase))then
       if(lcase)then
          ludef_case = lcase
          allocate(character(len=charlen) :: tlist(size(list)))
          tlist = list
          do i=1,size(tlist)
             list(i) = to_upper(list(i))
          end do
       end if
    else
       ludef_case = .false.
    end if
    do i=1,size(list)
       loc = minloc(list(i:),dim=1)
       if(loc.eq.1) cycle
       if(ludef_case) call cswap(tlist(i),tlist(loc+i-1))
       call cswap(list(i),list(loc+i-1))
    end do
    if(ludef_case) list=tlist
    
    return
  end subroutine sort_str
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  function sort_str_order(list,lcase) result(order)
    implicit none
    integer :: i,loc
    integer :: charlen
    logical :: ludef_case
    character(*), dimension(:), intent(inout) :: list
    character(:), allocatable, dimension(:) :: tlist
    logical, optional, intent(in) :: lcase !default is false

    integer, allocatable, dimension(:) :: torder,order

    charlen = len(list(1))
    if(present(lcase))then
       if(lcase)then
          ludef_case = lcase
          allocate(character(len=charlen) :: tlist(size(list)))
          tlist = list
          do i=1,size(tlist)
             list(i) = to_upper(list(i))
          end do
       end if
    else
       ludef_case = .false.
    end if

    allocate(torder(size(list)))
    do i=1,size(list)
       torder(i) = i
    end do
    
    do i=1,size(list)
       loc = minloc(list(i:),dim=1)
       if(loc.eq.1) cycle
       if(ludef_case) call cswap(tlist(i),tlist(loc+i-1))
       call cswap(list(i),list(loc+i-1))
       call iswap(torder(i),torder(loc+i-1))
    end do
    
    allocate(order(size(list)))
    do i=1,size(list)
       order(i) = findloc(torder,i,dim=1)
    end do
    
    if(ludef_case) list=tlist
    
    return
  end function sort_str_order
!!!#####################################################


!!!#####################################################
!!! sorts two arrays from min to max
!!! sorts the optional second array wrt the first array
!!!#####################################################
  subroutine isort1D(arr1,arr2,reverse)
    implicit none
    integer :: i,dim,loc
    integer :: ibuff
    logical :: udef_reverse
    integer, dimension(:) :: arr1
    integer, dimension(:),intent(inout),optional :: arr2
    logical, optional, intent(in) :: reverse

    if(present(reverse))then
       udef_reverse=reverse
    else
       udef_reverse=.false.
    end if

    dim=size(arr1,dim=1)
    do i=1,dim
       if(udef_reverse)then
          loc=maxloc(arr1(i:dim),dim=1)+i-1          
       else
          loc=minloc(arr1(i:dim),dim=1)+i-1
       end if
       ibuff=arr1(i)
       arr1(i)=arr1(loc)
       arr1(loc)=ibuff

       if(present(arr2)) then
          ibuff=arr2(i)
          arr2(i)=arr2(loc)
          arr2(loc)=ibuff
       end if
    end do

    return
  end subroutine isort1D
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine rsort1D(arr1,arr2,reverse)
    implicit none
    integer :: i,dim,loc
    real(real12) :: rbuff
    logical :: udef_reverse
    real(real12), dimension(:) :: arr1
    integer, dimension(:),intent(inout),optional :: arr2
    logical, optional, intent(in) :: reverse

    if(present(reverse))then
       udef_reverse=reverse
    else
       udef_reverse=.false.
    end if

    dim=size(arr1,dim=1)
    do i=1,dim
       if(udef_reverse)then
          loc=maxloc(arr1(i:dim),dim=1)+i-1          
       else
          loc=minloc(arr1(i:dim),dim=1)+i-1
       end if
       rbuff=arr1(i)
       arr1(i)=arr1(loc)
       arr1(loc)=rbuff

       if(present(arr2)) then
          rbuff=arr2(i)
          arr2(i)=arr2(loc)
          arr2(loc)=rbuff
       end if
    end do

    return
  end subroutine rsort1D
!!!#####################################################


!!!#####################################################
!!! sort an array from min to max
!!!#####################################################
  subroutine sort2D(arr,dim)
    implicit none
    integer :: i,j,dim,loc,istart
    integer, dimension(3) :: a123
    real(real12), dimension(3) :: buff
    real(real12), dimension(dim,3) :: arr

    a123(:)=(/1,2,3/)
    istart=1
    do j=1,3
       do i=j,dim
          loc=minloc(abs(arr(i:dim,a123(1))),dim=1,mask=(abs(arr(i:dim,a123(1))).gt.1.D-5))+i-1
          buff(:)=arr(i,:)
          arr(i,:)=arr(loc,:)
          arr(loc,:)=buff(:)
       end do

       scndrow: do i=j,dim
          if(abs(arr(j,a123(1))).ne.abs(arr(i,a123(1)))) exit scndrow
          loc=minloc(abs(arr(i:dim,a123(2)))+abs(arr(i:dim,a123(3))),dim=1,&
               mask=(abs(arr(j,a123(1))).eq.abs(arr(i:dim,a123(1)))))+i-1
          buff(:)=arr(i,:)
          arr(i,:)=arr(loc,:)
          arr(loc,:)=buff(:)
       end do scndrow

       a123=cshift(a123,1)
    end do

    return
  end subroutine sort2D
!!!#####################################################


!!!#####################################################
!!! return the sorted set of unique elements
!!!#####################################################
  subroutine iset(arr)
    implicit none
    integer :: i,n
    integer, allocatable, dimension(:) :: tmp_arr
    
    integer, allocatable, dimension(:) :: arr

    call sort1D(arr)
    allocate(tmp_arr(size(arr)))

    tmp_arr(1) = arr(1)
    n=1
    do i=2,size(arr)
       if(arr(i)==tmp_arr(n)) cycle
       n = n + 1
       tmp_arr(n) = arr(i)
    end do
    deallocate(arr); allocate(arr(n))
    arr(:n) = tmp_arr(:n)
    !call move_alloc(tmp_arr, arr)
    
  end subroutine iset
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine rset(arr, tol)
    implicit none
    integer :: i,n
    real(real12) :: tiny
    real(real12), allocatable, dimension(:) :: tmp_arr
    
    real(real12), allocatable, dimension(:) :: arr
    real(real12), optional :: tol

    if(present(tol))then
       tiny = tol
    else
       tiny = 1.D-4
    end if
    
    call sort1D(arr)
    allocate(tmp_arr(size(arr)))

    tmp_arr(1) = arr(1)
    n=1
    do i=2,size(arr)
       if(abs(arr(i)-tmp_arr(n)).lt.tiny) cycle
       n = n + 1
       tmp_arr(n) = arr(i)
    end do
    deallocate(arr); allocate(arr(n))
    arr(:n) = tmp_arr(:n)
    
  end subroutine rset
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine cset(arr,lcase,lkeep_size)
    implicit none
    integer :: i, n
    logical :: ludef_keep_size
    character(len=:), allocatable, dimension(:) :: tmp_arr
    character(*), allocatable, dimension(:) :: arr
    logical, intent(in), optional :: lcase, lkeep_size

    if(present(lcase))then
       call sort_str(arr,lcase)
    else
       call sort_str(arr)
    end if
    
    allocate(character(len=len(arr(1))) :: tmp_arr(size(arr)))
    tmp_arr(1) = arr(1)
    n=1
    
    do i=2,size(arr)
       if(trim(arr(i)).eq.trim(tmp_arr(n))) cycle
       n = n + 1
       tmp_arr(n) = arr(i)
    end do
    if(present(lkeep_size))then
       ludef_keep_size=lkeep_size
    else
       ludef_keep_size=.false.
    end if

    if(ludef_keep_size)then
       call move_alloc(tmp_arr,arr)!!!CONSISTENCY WITH OTHER SET FORMS
    else
       deallocate(arr)
       allocate(arr(n))
       arr(:n) = tmp_arr(:n)
    end if

  end subroutine cset
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  function set_str_output_order(arr,lcase,lkeep_size) result(order)
    implicit none
    integer :: i, n
    logical :: ludef_keep_size
    integer, allocatable, dimension(:) :: order
    character(len=:), allocatable, dimension(:) :: tmp_arr
    character(*), allocatable, dimension(:) :: arr
    logical, intent(in), optional :: lcase, lkeep_size

    allocate(order(size(arr)))
    if(present(lcase))then
       order=sort_str_order(arr,lcase)
    else
       order=sort_str_order(arr)
    end if
    
    allocate(character(len=len(arr(1))) :: tmp_arr(size(arr)))
    tmp_arr(1) = arr(1)
    n=1

    do i=2,size(arr)
       if(trim(arr(i)).eq.trim(tmp_arr(n)))then
          where(order.gt.order(n))
             order = order - 1
          end where
          cycle
       end if
       n = n + 1
       tmp_arr(n) = arr(i)
    end do
    write(0,*) order
    
    if(present(lkeep_size))then
       ludef_keep_size=lkeep_size
    else
       ludef_keep_size=.false.
    end if

    if(ludef_keep_size)then
       call move_alloc(tmp_arr,arr)!!!CONSISTENCY WITH OTHER SET FORMS
    else
       deallocate(arr)
       allocate(arr(n))
       arr(:n) = tmp_arr(:n)
    end if

  end function set_str_output_order
!!!#####################################################


!!!#####################################################
!!! sort an array over specified column
!!!#####################################################
!!! Have it optionally take in an integer vector that ...
!!! ... lists the order of imporance of columns
  subroutine sort_col(arr1,col,reverse)
    implicit none
    integer :: i,dim,loc
    logical :: udef_reverse
    real(real12), allocatable, dimension(:) :: dbuff
    real(real12), dimension(:,:) :: arr1

    integer, intent(in) :: col
    logical, optional, intent(in) :: reverse


    if(present(reverse))then
       udef_reverse=reverse
    else
       udef_reverse=.false.
    end if

    allocate(dbuff(size(arr1,dim=2)))

    dim=size(arr1,dim=1)
    do i=1,dim
       if(udef_reverse)then
          loc=maxloc(arr1(i:dim,col),dim=1)+i-1          
       else
          loc=minloc(arr1(i:dim,col),dim=1)+i-1
       end if
       dbuff=arr1(i,:)
       arr1(i,:)=arr1(loc,:)
       arr1(loc,:)=dbuff

    end do

    return
  end subroutine sort_col
!!!#####################################################


!!!#####################################################
!!! swap two ints
!!!#####################################################
  subroutine iswap(i1,i2)
    implicit none
    integer :: i1,i2,itmp

    itmp=i1
    i1=i2
    i2=itmp
  end subroutine iswap
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine rswap(d1,d2)
    implicit none
    real(real12) :: d1,d2,dtmp

    dtmp=d1
    d1=d2
    d2=dtmp
  end subroutine rswap
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine cswap(c1,c2)
    implicit none
    character(*) :: c1,c2
    character(len=:), allocatable :: ctmp

    ctmp=c1
    c1=c2
    c2=ctmp
  end subroutine cswap
!!!-----------------------------------------------------
!!!-----------------------------------------------------
  subroutine rswap_vec(vec1,vec2)
    implicit none
    real(real12),dimension(:)::vec1,vec2
    real(real12),allocatable,dimension(:)::tvec

    allocate(tvec(size(vec1)))
    tvec=vec1(:)
    vec1(:)=vec2(:)
    vec2(:)=tvec
  end subroutine rswap_vec
!!!#####################################################


!!!#####################################################
!!! shuffle an array along one dimension
!!!#####################################################
  subroutine shuffle(arr,dim,seed)
    implicit none
    integer :: iseed,istart
    integer :: i,j,k,n_data,iother
    integer :: i1s,i2s,i1e,i2e,j1s,j2s,j1e,j2e
    real(real12) :: r
    real(real12), allocatable, dimension(:,:) :: tlist

    integer, intent(in) :: dim
    real(real12), dimension(:,:), intent(inout) :: arr

    integer, optional, intent(in) :: seed

    if(present(seed)) iseed = seed

    call random_seed(iseed)
    n_data = size(arr,dim=dim)
    if(dim.eq.1)then
       iother = 2
       i2s=1;i2e=size(arr,dim=iother)
       j2s=1;j2e=size(arr,dim=iother)
    else
       iother = 1
       i1s=1;i1e=size(arr,dim=iother)
       j1s=1;j1e=size(arr,dim=iother)
    end if
    istart=1
    allocate(tlist(1,size(arr,dim=iother)))
    do k=1,2
       do i=1,n_data
          call random_number(r)
          j = istart + floor((n_data+1-istart)*r)
          if(dim.eq.1)then
             i1s=i;i1e=i
             j1s=j;j1e=j
          else
             i2s=i;i2e=i
             j2s=j;j2e=j
          end if
          tlist(1:1,:) = arr(i1s:i1e,i2s:i2e)
          arr(i1s:i1e,i2s:i2e) = arr(j1s:j1e,j2s:j2e)
          arr(j1s:j1e,j2s:j2e) = tlist(1:1,:)
       end do
    end do

  end subroutine shuffle
!!!#####################################################


!!!#############################################################################
!!!#############################################################################
!!!  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
!!!#############################################################################
!!!#############################################################################


!!!#####################################################
!!! counts the number of words on a line
!!!#####################################################
  integer function Icount(full_line,tmpchar)
    character(*) :: full_line
    !ONLY WORKS WITH IFORT COMPILER
    !      character(1) :: fs
    character(len=:),allocatable :: fs
    character(*),optional :: tmpchar
    integer ::items,pos,k,length
    items=0
    pos=1

    length=1
    if(present(tmpchar)) length=len(trim(tmpchar))
    allocate(character(len=length) :: fs)
    if(present(tmpchar)) then
       fs=trim(tmpchar)
    else
       fs=" "
    end if

    loop: do
       k=verify(full_line(pos:),fs)
       if (k.eq.0) exit loop
       items=items+1
       pos=k+pos-1
       k=scan(full_line(pos:),fs)
       if (k.eq.0) exit loop
       pos=k+pos-1
    end do loop
    Icount=items
  end function Icount
!!!#####################################################


!!!#####################################################
!!! counts the number of words on a line
!!!#####################################################
  subroutine readcl(full_line,store,tmpchar)
    character(*) :: full_line
    !ONLY WORKS WITH IFORT COMPILER
    !      character(1) :: fs
    character(len=:),allocatable :: fs
    character(*),optional :: tmpchar
    character(100),dimension(1000) :: tmp_store
    character(*),allocatable,dimension(:),optional :: store
    integer ::items,pos,k,length
    items=0
    pos=1

    length=1
    if(present(tmpchar)) length=len(trim(tmpchar))
    allocate(character(len=length) :: fs)
    if(present(tmpchar)) then
       fs=tmpchar
    else
       fs=" "
    end if

    loop: do
       k=verify(full_line(pos:),fs)
       if (k.eq.0) exit loop
       pos=k+pos-1
       k=scan(full_line(pos:),fs)
       if (k.eq.0) exit loop
       items=items+1
       tmp_store(items)=full_line(pos:pos+k-1)
       pos=k+pos-1
    end do loop

    if(present(store))then
       if(.not.allocated(store)) allocate(store(items))
       do k=1,items
          store(k)=trim(tmp_store(k))
       end do
    end if

  end subroutine readcl
!!!#####################################################

!!!#####################################################
!!! grep 
!!!#####################################################
!!! searches a file untill it finds the mattching patern
  subroutine grep(unit,input,lstart)
    integer :: unit,Reason
    character(*) :: input
    character(1024) :: buffer
    logical, optional, intent(in) :: lstart
    !  character(1024), intent(out), optional :: linechar
    if(present(lstart))then
       if(lstart) rewind(unit)
    else
       rewind(unit)
    end if

    greploop: do
       read(unit,'(A100)',iostat=Reason) buffer
       if(Reason.lt.0) return
       if(index(trim(buffer),trim(input)).ne.0) exit greploop
    end do greploop
  end subroutine grep
!!!#####################################################


!!!#####################################################
!!! count number of occurances of substring in string
!!!#####################################################
  function count_occ(string,substring)
    implicit none
    integer :: pos,i,count_occ
    character(*) :: string,substring

    pos=1
    count_occ=0
    countloop: do 
       i=verify(string(pos:), substring)
       if (i.eq.0) exit countloop
       if(pos.eq.len(string)) exit countloop
       count_occ=count_occ+1
       pos=i+pos-1
       i=scan(string(pos:), ' ')
       if (i.eq.0) exit countloop
       pos=i+pos-1
    end do countloop

    return
  end function count_occ
!!!#####################################################


!!!#####################################################
!!! Assigns variables of flags from getarg
!!!#####################################################
!!! SHOULD MAKE THIS A FUNCTION INSTEAD !!!
  subroutine flagmaker(buffer,flag,i,skip,empty)
    integer :: i
    logical :: skip,empty
    character(*) :: flag,buffer

    if(len(trim(buffer)).eq.len(trim(flag))) then
       call get_command_argument(i+1,buffer)
       if(scan(buffer,'-').eq.1.or.buffer.eq.'') then
          buffer=""
          empty=.true.
       else
          skip=.true.
       end if
    else
       buffer=buffer(len(trim(flag))+1:)
    end if

    return
  end subroutine flagmaker
!!!#####################################################


!!!#####################################################
!!! Writes out a loading bar to the terminal
!!!#####################################################
  subroutine loadbar(count,div,loaded)
    implicit none
    integer :: count,div !div=10
    real :: tiny=1.E-5
    character(1) :: yn,creturn = achar(13)
    character(1), optional :: loaded

    if(.not.present(loaded)) then
       yn='n'
    else
       yn=loaded
    end if

    if(yn.eq.'l'.or.yn.eq.'y') then
       write(*,'(A,20X,A)',advance='no') achar(13),achar(13)
       return
    end if

    if((real(count)/real(4*div)-floor(real(count)/real(4*div))).lt.tiny) then
       write(6,'(A,20X,A,"CALCULATING")',advance='no') creturn,creturn
    else if((real(count)/real(div)-floor(real(count)/real(div))).lt.tiny) then
       write(6,'(".")',advance='no')
    end if

    return
  end subroutine loadbar
!!!#####################################################


!!!#####################################################
!!! Jumps UNIT to input line number
!!!#####################################################
  subroutine jump(unit,linenum)
    integer :: unit, linenum, move
    rewind(unit)
    do move=1,(linenum)
       read(unit,*)
    end do
    return
  end subroutine jump
!!!#####################################################


!!!#####################################################
!!! File checker
!!!#####################################################
  subroutine file_check(UNIT,FILENAME,ACTION)
    implicit none
    integer :: i,UNIT,Reason
    character(len=*) :: FILENAME
    character(20) :: udef_action
    character(20), optional :: ACTION
    logical :: filefound

    udef_action="READWRITE"
    if(present(ACTION)) udef_action=ACTION
    udef_action=to_upper(udef_action)
    do i=1,5
       inquire(file=trim(FILENAME),exist=filefound)
       if(.not.filefound) then
          write(6,'("File name ",A," not found.")')&
               "'"//trim(FILENAME)//"'"
          write(6,'("Supply another filename: ")')
          read(*,*) FILENAME
       else
          write(6,'("Using file ",A)')  &
               "'"//trim(FILENAME)//"'"
          exit
       end if
       if(i.ge.4) then
          stop "Nope"
       end if
    end do
    if(trim(adjustl(udef_action)).eq.'NONE')then
       write(6,*) "File found, but not opened."
    else
       open(unit=UNIT,file=trim(FILENAME),action=trim(udef_action),iostat=Reason)
    end if


    return
  end subroutine file_check
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to upper case
!!!#####################################################
  function to_upper(buffer) result(upper)
    implicit none
    integer :: i,j
    character(*) :: buffer
    character(len=:),allocatable :: upper


    allocate(character(len=len(buffer)) :: upper)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("a").and.j.le.iachar("z"))then
          upper(i:i)=achar(j-32)
       else
          upper(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_upper
!!!#####################################################


!!!#####################################################
!!! converts all characters in string to lower case
!!!#####################################################
  function to_lower(buffer) result(lower)
    implicit none
    integer :: i,j
    character(*) :: buffer
    character(len=:),allocatable :: lower


    allocate(character(len=len(buffer)) :: lower)
    do i=1,len(buffer)
       j=iachar(buffer(i:i))
       if(j.ge.iachar("A").and.j.le.iachar("Z"))then
          lower(i:i)=achar(j+32)
       else
          lower(i:i)=buffer(i:i)
       end if
    end do

    return
  end function to_lower
!!!#####################################################

end module misc
